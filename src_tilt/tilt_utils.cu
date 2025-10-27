#include <cmath>
#include <chrono>
#include <boost/algorithm/string.hpp>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "Status.h"
#include "Netcdf_interface.h"
#include "Array.h"
#include "Gas_concs.h"
#include "Aerosol_optics_rt.h"
#include "tilt_utils.h"
#include "types.h"


namespace
{
    __device__
    Float interpolate_gpu(const int n_x, const int n_y, const int n_lay_in,
                          const Float* zf_in, const Float* zh_in,
                          const Float* vlay_in, const Float* vlev_in,
                          const int ix, const int iy,
                          const Float zp, const ijk offset)
    {

            const int n_col = n_x * n_y;
            const int n_lev_in = n_lay_in + 1;

            int posh_bot = 0;
            int posf_bot = 0;
            for (int ilev=0; ilev<n_lev_in-1; ++ilev)
                if (zh_in[ilev] < zp)
                    posh_bot = ilev;
            for (int ilay=0; ilay<n_lay_in; ++ilay)
                if (zf_in[ilay] < zp)
                    posf_bot = ilay;
            const Float* v_top;
            const Float* v_bot;
            Float  z_top;
            Float  z_bot;

            const int zh_top = zh_in[posh_bot+1];
            const int zh_bot = zh_in[posh_bot];
            const int zf_top = (posf_bot+1 < n_lay_in) ? zf_in[posf_bot+1] : zh_top+1;
            const int zf_bot = zf_in[posf_bot];

            // tilted layers between the lowest level and layer
            if (posf_bot == 0 && posh_bot == 0 && zp < zf_in[0])
            {
                v_top = &vlay_in[posf_bot*n_col];
                z_top = zf_in[posf_bot];
                v_bot = &vlev_in[posh_bot*n_col];
                z_bot = zh_in[posh_bot];
            }
            // tilted layers between the top layer and level
            else if (posf_bot == n_lay_in - 1 && posh_bot == n_lev_in -2)
            {
                v_top = &vlev_in[(posh_bot + 1) * n_col];
                z_top = zh_in[posh_bot + 1];
                v_bot = &vlay_in[posf_bot * n_col];
                z_bot = zf_in[posf_bot];
            }
            // all layers in between
            else
            {
                if (zh_top > zf_top)
                {
                    v_top = &vlay_in[(posf_bot + 1) * n_col];
                    z_top = zf_in[posf_bot + 1];
                }
                else
                {
                    v_top = &vlev_in[(posh_bot + 1) * n_col];
                    z_top = zh_in[posh_bot + 1];
                }
                if (zh_bot < zf_bot)
                {
                    v_bot = &vlay_in[posf_bot * n_col];
                    z_bot = zf_in[posf_bot];
                }
                else
                {
                    v_bot = &vlev_in[posh_bot * n_col];
                    z_bot = zh_in[posh_bot];
                }
            }

            Float dz = z_top-z_bot;

            const int idx = (ix + offset.i)%n_x + (iy+offset.j)%n_y * n_x;
            const Float value = (zp-z_bot)/dz*v_top[idx] + (z_top-zp)/dz*v_bot[idx];
            return value;
    }



    __global__
    void tilt_plev_gpu(const int n_x, const int n_y, const int n_z, const int n_z_tilt,
                       const ijk* tilted_path,
                       const Float* zh_tilt,
                       const Float* z, const Float* zh,
                       const Float* p_lay, const Float* p_lev,
                       Float* p_lev_tilt)
    {
        const int i = blockIdx.x*blockDim.x + threadIdx.x;

        if ( (i < (n_z_tilt+1) ) )
        {
            const Float pos_z = zh_tilt[i];
            const Float p = interpolate_gpu(n_x, n_y, n_z, z, zh, p_lay, p_lev, 0, 0, pos_z, tilted_path[i]);
            p_lev_tilt[i] = p;
        }
    }

    __global__
    void tilt_tlay_gpu(const int n_x, const int n_y, const int n_z, const int n_z_tilt,
                       const ijk* tilted_path,
                       const Float* z_tilt,
                       const Float* z, const Float* zh,
                       const Float* p_lay, const Float* p_lev,
                       Float* t_lay_tilt)
    {
        const int i = blockIdx.x*blockDim.x + threadIdx.x;
        const int j = blockIdx.y*blockDim.y + threadIdx.y;
        const int k = blockIdx.z*blockDim.z + threadIdx.z;

        if ( (i < n_x) && (j< n_y) && (k < n_z_tilt) )
        {
            const int idx  = i + j*n_y + k*n_y*n_x;
            const Float pos_z = (z_tilt[k] + z_tilt[k+1])/Float(2.);
            const Float t = interpolate_gpu(n_x, n_y, n_z, z, zh, p_lay, p_lev, i, j, pos_z, tilted_path[k]);
            t_lay_tilt[idx] = t;
        }
    }

    __global__
    void tilt_layers_gpu(const int n_x, const int n_y, const int n_z,
                         const ijk* tilted_path,
                         const Float* v_in, Float* v_tilt)
    {
        const int ix = blockIdx.x*blockDim.x + threadIdx.x;
        const int iy = blockIdx.y*blockDim.y + threadIdx.y;
        const int iz = blockIdx.z*blockDim.z + threadIdx.z;

        if ( ( ix < n_x) && ( iy < n_y) && (iz < n_z) )
        {
            const ijk offset = tilted_path[iz];

            const int idx_out  = ix + iy*n_y + iz*n_y*n_x;
            const int idx_in = (ix + offset.i)%n_x + (iy+offset.j)%n_y * n_x + offset.k*n_y*n_x;

            v_tilt[idx_out] = v_in[idx_in];

        }
    }

    __global__
    void tilt_clouds_gpu(const int n_x, const int n_y, const int n_z,
                                      const ijk* tilted_path, const Float* zh,
                                      const Float* water_path, const Float* effective_size,
                                      Float* water_path_tmp, Float* effective_size_tmp)
    {
        const int ix = blockIdx.x*blockDim.x + threadIdx.x;
        const int iy = blockIdx.y*blockDim.y + threadIdx.y;
        const int iz = blockIdx.z*blockDim.z + threadIdx.z;

        if ( ( ix < n_x) && ( iy < n_y) && (iz < n_z) )
        {
            const ijk offset = tilted_path[iz];

            const int idx_out  = ix + iy*n_y + iz*n_y*n_x;
            const int idx_in = (ix + offset.i)%n_x + (iy+offset.j)%n_y * n_x + offset.k*n_y*n_x;

            const Float dz_local =  zh[offset.k+1] - zh[offset.k];

            water_path_tmp[idx_out] = water_path[idx_in]/dz_local;
            effective_size_tmp[idx_out] = effective_size[idx_in];

        }
    }

    __global__
    void compress_tilted_clouds_gpu(const int n_x, const int n_y, const int n_z,
                                      const int* tilted_path_bounds, const Float* zh,
                                      const Float eff_size_min, const Float eff_size_max,
                                      const Float* water_path_tmp, const Float* effective_size_tmp,
                                      Float* water_path, Float* effective_size)
    {
        const int ix = blockIdx.x*blockDim.x + threadIdx.x;
        const int iy = blockIdx.y*blockDim.y + threadIdx.y;
        const int iz = blockIdx.z*blockDim.z + threadIdx.z;

        if ( ( ix < n_x) && ( iy < n_y) && (iz < n_z) )
        {
            const int i_lwr = tilted_path_bounds[iz];
            const int i_upr = tilted_path_bounds[iz+1];

            Float t_sum = 0.0;
            Float sum = 0.0;
            Float w_sum = 0.0;
            Float avg = 0.0;

            const int idx_out  = ix + iy*n_y + iz*n_y*n_x;

            for (int i = i_lwr; i<i_upr; ++i)
            {
                int in_idx = ix + iy * n_x + i * n_x * n_y;

                const Float dz_local =  zh[i+1] - zh[i];

                sum += water_path_tmp[in_idx] * dz_local;

                t_sum += effective_size_tmp[in_idx] * water_path_tmp[in_idx];
                w_sum += water_path_tmp[in_idx];
            }

            water_path[idx_out] = sum;
            if (w_sum > Float(1e-6))
                effective_size[idx_out] = max(eff_size_min, min(t_sum / w_sum, eff_size_max));
            else
                effective_size[idx_out] = eff_size_min;

        }
    }

    __global__
    void compress_tilted_columns_gpu(const int n_x, const int n_y, const int n_z,
                                      const int* tilted_path_bounds,
                                      const Float* v_tilt, const Float* p_lev_tilt,
                                      Float* v_out)
    {
        const int ix = blockIdx.x*blockDim.x + threadIdx.x;
        const int iy = blockIdx.y*blockDim.y + threadIdx.y;
        const int iz = blockIdx.z*blockDim.z + threadIdx.z;

        if ( ( ix < n_x) && ( iy < n_y) && (iz < n_z) )
        {
            const int i_lwr = tilted_path_bounds[iz];
            const int i_upr = tilted_path_bounds[iz+1];

            Float v_sum = Float(0.);
            Float p_sum = Float(0.);

            const int idx_out  = ix + iy*n_y + iz*n_y*n_x;

            for (int i = i_lwr; i<i_upr; ++i)
            {
                int in_idx = ix + iy * n_x + i * n_x * n_y;

                const Float dp_local =  p_lev_tilt[i+1] - p_lev_tilt[i];

                v_sum += v_tilt[in_idx] * dp_local;
                p_sum += dp_local;
            }

            v_out[idx_out] = v_sum / p_sum;

        }
    }

    __global__
    void compute_rh(const int n_x, const int n_y, const int n_z,
                    const Float* t_lay,
                    const Float* p_lay,
                    const Float* h2o_vmr,
                    Float* rh)
    {
        const int ix = blockIdx.x*blockDim.x + threadIdx.x;
        const int iy = blockIdx.y*blockDim.y + threadIdx.y;
        const int iz = blockIdx.z*blockDim.z + threadIdx.z;

        if ( ( ix < n_x) && ( iy < n_y) && (iz < n_z) )
        {
            const int idx  = ix + iy*n_y + iz*n_y*n_x;

            const Float qsat = t_lay[idx] > Float(273.15) ? qsat_liq(p_lay[idx], t_lay[idx]) : qsat_ice(p_lay[idx], t_lay[idx]);

            constexpr Float eps = Float(0.62185985592);

            const Float q = h2o_vmr[idx] * eps / (1 + h2o_vmr[idx] * eps);

            rh[idx] = min(max(Float(0.), q / qsat), Float(1.));

        }
    }




}

void tica_tilt_gpu(
        const Float sza, const Float azi,
        const int n_col_x, const int n_col_y, const int n_col,
        const int n_lay, const int n_lev, const int n_z_in, const int n_zh_in ,
        Array<Float,1>& xh, Array<Float,1>& yh, Array<Float,1>& zh, Array<Float,1>& z,
        Array_gpu<Float,2>& p_lay, Array_gpu<Float,2>& t_lay, Array_gpu<Float,2>& p_lev, Array_gpu<Float,2>& t_lev,
        Array_gpu<Float,2>& lwp, Array_gpu<Float,2>& iwp, Array_gpu<Float,2>& rel, Array_gpu<Float,2>& dei, Array_gpu<Float,2>& rh,
        Gas_concs_gpu& gas_concs, Aerosol_concs_gpu& aerosol_concs,
        std::vector<std::string>& gas_names, std::vector<std::string>& aerosol_names,
        bool switch_cloud_optics, bool switch_liq_cloud_optics, bool switch_ice_cloud_optics, bool switch_aerosol_optics,
        int rnd_seed)
{
    ////// SETUP FOR CENTER START POINT TILTING //////
    // Finding tilted path can stay on CPU
    Array<ijk,1> center_path;
    Array<Float,1> center_zh_tilt;
    create_tilted_path(xh.v(),yh.v(),zh.v(),z.v(),sza,azi, 0.5, 0.5, center_path.v(), center_zh_tilt.v());


    int n_zh_tilt_center = center_zh_tilt.v().size();
    int n_z_tilt_center = n_zh_tilt_center - 1;

    center_path.set_dims({n_zh_tilt_center});
    center_zh_tilt.set_dims({n_zh_tilt_center});

    Array<int,1> center_path_bounds({n_zh_in});
    get_tilted_path_bounds(n_zh_tilt_center, center_path.v(), center_path_bounds.v());

    Array_gpu<ijk,1> center_path_gpu(center_path);
    Array_gpu<Float,1> center_zh_tilt_gpu(center_zh_tilt);
    Array_gpu<int,1> center_path_bounds_gpu(center_path_bounds);
    Array_gpu<Float,1> zh_gpu(zh);
    Array_gpu<Float,1> z_gpu(z);

    // p_lay and p_lev remain unchanged after tiltint and compression.
    // However, we need to obtain the pressures at each cell crossing of the tilted path to use for weighted averaging
    Array_gpu<Float,1> p_lev_tilt({n_zh_tilt_center});

    const int block_col_x = 8;
    const int block_col_y = 8;
    const int block_z_in = 4;
    const int block_z_tilt = 4;
    const int block_zh_tilt = 4;

    const int grid_col_x = n_col_x/block_col_x + (n_col_x%block_col_x > 0);
    const int grid_col_y = n_col_y/block_col_y + (n_col_y%block_col_y > 0);
    const int grid_z_in = n_z_in/block_z_in + (n_z_in%block_z_in > 0);
    const int grid_z_tilt = n_z_tilt_center/block_z_tilt + (n_z_tilt_center%block_z_tilt > 0);
    const int grid_zh_tilt = n_zh_tilt_center/block_zh_tilt + (n_zh_tilt_center%block_zh_tilt > 0);

    dim3 grid_1d(grid_zh_tilt);
    dim3 block_1d(block_zh_tilt);

    dim3 grid_3d_1(grid_col_x, grid_col_y, grid_z_in);
    dim3 block_3d_1(block_col_x, block_col_y, block_z_in);

    dim3 grid_3d_2(grid_col_x, grid_col_y, grid_z_tilt);
    dim3 block_3d_2(block_col_x, block_col_y, block_z_tilt);

    tilt_plev_gpu<<<grid_1d, block_1d>>>(
            n_col_x, n_col_y, n_z_in, n_z_tilt_center,
            center_path_gpu.ptr(),
            center_zh_tilt_gpu.ptr(), z_gpu.ptr(), zh_gpu.ptr(),
            p_lay.ptr(), p_lev.ptr(),
            p_lev_tilt.ptr() );

    // T_lev remains unchanged after tilting and compression.
    // T_lay is first to z_tilt and then compressed

    Array_gpu<Float,2> t_lay_tilt({n_col, n_z_tilt_center});

    tilt_tlay_gpu<<<grid_3d_2, block_3d_2>>>(
        n_col_x, n_col_y, n_z_in, n_z_tilt_center,
        center_path_gpu.ptr(),
        center_zh_tilt_gpu.ptr(), z_gpu.ptr(), zh_gpu.ptr(),
        t_lay.ptr(), t_lev.ptr(),
        t_lay_tilt.ptr() );

    compress_tilted_columns_gpu<<<grid_3d_1, block_3d_1>>>(
        n_col_x, n_col_y, n_z_in,
        center_path_bounds_gpu.ptr(),
        t_lay_tilt.ptr(), p_lev_tilt.ptr(),
        t_lay.ptr());

    //// tilt and compress gasses ////
    Array_gpu<Float,2> gas_tilt({n_col, n_z_tilt_center});

    for (const auto& gas_name : gas_names)
    {
        if (!gas_concs.exists(gas_name))
        {
            continue;
        }

        Array_gpu<Float,2> gas(gas_concs.get_vmr(gas_name));
        if (gas.size() > 1)
        {
            if (gas.get_dims()[0] > 1)
            { // checking: do we have 3D field?
                tilt_layers_gpu<<<grid_3d_2, block_3d_2>>>(
                        n_col_x, n_col_y, n_z_tilt_center,
                        center_path_gpu.ptr(),
                        gas.ptr(), gas_tilt.ptr());

                compress_tilted_columns_gpu<<<grid_3d_1, block_3d_1>>>(
                        n_col_x, n_col_y, n_z_in,
                        center_path_bounds_gpu.ptr(),
                        gas_tilt.ptr(), p_lev_tilt.ptr(),
                        gas.ptr());

                gas_concs.set_vmr(gas_name, gas);

            }
            else
            {
                throw std::runtime_error("No tilted column implementation for single column profiles.");
            }
            if ( (gas_name == "h2o") && switch_aerosol_optics )
            {
                compute_rh<<<grid_3d_1, block_3d_1>>>(
                        n_col_x, n_col_y, n_z_in,
                        t_lay.ptr(),
                        p_lay.ptr(),
                        gas.ptr(),
                        rh.ptr());
            }
        }
    }

    if (switch_aerosol_optics)
    {
        Array_gpu<Float,2> aerosol_tilt({n_col, n_z_tilt_center});

        for (const auto& aerosol_name : aerosol_names)
        {
            if (!aerosol_concs.exists(aerosol_name))
            {
                continue;
            }

            Array_gpu<Float,2> aerosol(aerosol_concs.get_vmr(aerosol_name));
            if (aerosol.size() > 1)
            {
                if (aerosol.get_dims()[0] > 1)
                {
                    //  Only tilt if we have a 3D aerosol field
                    tilt_layers_gpu<<<grid_3d_2, block_3d_2>>>(
                            n_col_x, n_col_y, n_z_tilt_center,
                            center_path_gpu.ptr(),
                            aerosol.ptr(), aerosol_tilt.ptr());

                    compress_tilted_columns_gpu<<<grid_3d_1, block_3d_1>>>(
                            n_col_x, n_col_y, n_z_in,
                            center_path_bounds_gpu.ptr(),
                            aerosol_tilt.ptr(), p_lev_tilt.ptr(),
                            aerosol.ptr());

                    aerosol_concs.set_vmr(aerosol_name, aerosol);
                }
            }
        }
    }


    if (switch_cloud_optics)
    {

        Array_gpu<Float,2> lwp_tmp({n_col, n_z_tilt_center});
        Array_gpu<Float,2> rel_tmp({n_col, n_z_tilt_center});
        Array_gpu<Float,2> iwp_tmp({n_col, n_z_tilt_center});
        Array_gpu<Float,2> dei_tmp({n_col, n_z_tilt_center});

        if (switch_liq_cloud_optics)
        {
            tilt_clouds_gpu<<<grid_3d_2, block_3d_2>>>(n_col_x, n_col_y, n_z_tilt_center,
                                         center_path_gpu.ptr(), zh_gpu.ptr(),
                                         lwp.ptr(), rel.ptr(),
                                         lwp_tmp.ptr(), rel_tmp.ptr());

            compress_tilted_clouds_gpu<<<grid_3d_1, block_3d_1>>>(n_col_x, n_col_y, n_z_in,
                                               center_path_bounds_gpu.ptr(), center_zh_tilt_gpu.ptr(),
                                               Float(2.5), Float(21.5),
                                               lwp_tmp.ptr(), rel_tmp.ptr(),
                                               lwp.ptr(), rel.ptr());

        }

        if (switch_ice_cloud_optics)
        {
            tilt_clouds_gpu<<<grid_3d_2, block_3d_2>>>(n_col_x, n_col_y, n_z_tilt_center,
                                         center_path_gpu.ptr(), zh_gpu.ptr(),
                                         iwp.ptr(), dei.ptr(),
                                         iwp_tmp.ptr(), dei_tmp.ptr());

            compress_tilted_clouds_gpu<<<grid_3d_1, block_3d_1>>>(n_col_x, n_col_y, n_z_in,
                                               center_path_bounds_gpu.ptr(), center_zh_tilt_gpu.ptr(),
                                               Float(10), Float(180.),
                                               iwp_tmp.ptr(), dei_tmp.ptr(),
                                               iwp.ptr(), dei.ptr());

        }

    }
}

