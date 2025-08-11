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


void tilted_path(std::vector<Float>& xh, std::vector<Float>& yh,
                 std::vector<Float>& zh, std::vector<Float>& z,
                 Float sza, Float azi,
                 Float x_start, Float y_start,
                 std::vector<ijk>& tilted_path,
                 std::vector<Float>& zh_tilted)
{
    const Float dx = xh[1]-xh[0];
    const Float dy = yh[1]-yh[0];
    const Float z_top = *std::max_element(zh.begin(), zh.end());
    const int n_x = xh.size()-1;
    const int n_y = yh.size()-1;
    int i = 0;
    int j = 0;
    int k = 0;

    Float xp = xh[0] + x_start*dx;
    Float yp = yh[0] + y_start*dy;
    Float zp = 0.;
    Float dl = 0.;

    tilted_path.clear();
    zh_tilted.clear();
    std::vector<Float> dz_tilted;

    Float dir_x = std::sin(sza) * std::sin(azi); // azi 0 is from the north
    Float dir_y = std::sin(sza) * std::cos(azi);
    Float dir_z = std::cos(sza);

    int z_idx = -1;

    const Float epsilon = 1e-8; // Small value to handle floating-point precision
    const Float min_step = 1e-2; // Minimum step size in meters

    tilted_path.push_back({i, j, k}); // Add starting point
    dz_tilted.push_back(0.0);
    z_idx = 0;

    while (zp < z_top)
    {
        // Check bounds before accessing arrays
        if (k + 1 >= zh.size()) {
            std::cerr << "Error: k+1 (" << k + 1 << ") out of bounds for zh (size=" << zh.size() << ")" << std::endl;
            break;
        }
        if (j + 1 >= yh.size()) {
            std::cerr << "Error: j+1 (" << j + 1 << ") out of bounds for yh (size=" << yh.size() << ")" << std::endl;
            break;
        }
        if (i + 1 >= xh.size()) {
            std::cerr << "Error: i+1 (" << i + 1 << ") out of bounds for xh (size=" << xh.size() << ")" << std::endl;
            break;
        }

        // Calculate distances to next cell boundaries
        Float lz = (std::abs(dir_z) < epsilon) ?
                    100000.: (zh[k+1] - zp) / dir_z;

        // Handle cases where we're extremely close to a boundary
        if (std::abs(zp - zh[k+1]) < epsilon && dir_z > 0) {
            k += 1;
            zp = zh[k];
            if (k + 1 >= zh.size()) {
                break; // We've reached the top
            }
            continue;
        }

        Float ly;
        if (std::abs(dir_y) < epsilon) {
            ly = 100000.;
        } else if (dir_y < 0) {
            if (std::abs(yp - yh[j]) < epsilon) {
                // Already at the lower boundary, move to the previous cell
                j = (j == 0) ? n_y - 1 : j - 1;
                yp = yh[j+1] - epsilon; // Position just inside the cell
                continue;
            }
            ly = (yp - yh[j]) / (-dir_y);
        } else { // dir_y > 0
            if (std::abs(yp - yh[j+1]) < epsilon) {
                // Already at the upper boundary, move to the next cell
                j = (j + 1) % n_y;
                yp = yh[j] + epsilon;
                continue;
            }
            ly = (yh[j+1] - yp) / dir_y;
        }

        Float lx;
        if (std::abs(dir_x) < epsilon) {
            lx = 100000.;
        } else if (dir_x < 0) {
            if (std::abs(xp - xh[i]) < epsilon) {
                // Already at the lower boundary, move to the previous cell
                i = (i == 0) ? n_x - 1 : i - 1;
                xp = xh[i+1] - epsilon;
                continue;
            }
            lx = (xp - xh[i]) / (-dir_x);
        } else { // dir_x > 0
            if (std::abs(xp - xh[i+1]) < epsilon) {
                // Already at the upper boundary, move to the next cell
                i = (i + 1) % n_x;
                xp = xh[i] + epsilon;
                continue;
            }
            lx = (xh[i+1] - xp) / dir_x;
        }

        Float l = std::min({lx, ly, lz});
        Float dx0 = l * dir_x;
        Float dy0 = l * dir_y;
        Float dz0 = l * dir_z;
        // Move along axes:
        xp += dx0;
        yp += dy0;
        zp += dz0;

        // Record the path segment
        dz_tilted[z_idx] += dz0;

        // Record boundary crossing
        // if path is larger than 1 cm
        if (l > min_step)
        {
            if ((std::abs(l - lx) < epsilon) || (std::abs(l - ly) < epsilon) ||
                ((std::abs(l - lz) < epsilon || zp >= zh[k + 1]))) {
                // Create a new path segment after crossing boundary
                tilted_path.push_back({i, j, k});
                dz_tilted.push_back(0.0);
                z_idx += 1;
            }
        }

        // Check z boundary crossing
        if ((std::abs(l - lz) < epsilon || zp >= zh[k+1])) {
            k += 1;
        }

        // Check y boundary crossing
        if (std::abs(l - ly) < epsilon) {

            j = int(j + sign(dy0));
            j = (j == -1) ? n_y - 1 : j%n_y;
            yp = dy0 < 0 ? yh[j+1] : yh[j];

        }

        // Check x boundary crossing
        if (std::abs(l - lx) < epsilon) {
            i = int(i + sign(dx0));
            i = (i == -1) ? n_x - 1 : i%n_x;
            xp = dx0 < 0 ? xh[i+1] : xh[i];
        }
    }
    // Construct final zh_tilted
    zh_tilted.clear();
    zh_tilted.push_back(0.);
    for (int iz = 0; iz < dz_tilted.size(); ++iz) {
        zh_tilted.push_back(zh_tilted[iz] + dz_tilted[iz]);
    }
}

void restore_bkg_profile(const int n_x, const int n_y,
                      const int n_full,
                      const int n_tilt,
                      const int bkg_start,
                      std::vector<Float>& var,
                      std::vector<Float>& var_w_bkg)
{
    const int n_out = n_tilt + (n_full - bkg_start);

    std::vector<Float> var_tmp(n_out * n_x * n_y);
    #pragma omp parallel for
    for (int ilay = 0; ilay < n_tilt; ++ilay)
    {
        for (int iy = 0; iy < n_y; ++iy)
        {
            for (int ix = 0; ix < n_x; ++ix)
            {
                const int out_idx = ix + iy * n_x + ilay * n_x * n_y;
                var_tmp[out_idx] = var[out_idx];
            }
        }
    }

    #pragma omp parallel for
    for (int ilay = n_tilt; ilay < n_out; ++ilay)
    {
        int ilay_in = ilay - (n_tilt - bkg_start);
        for (int iy = 0; iy < n_y; ++iy)
        {
            for (int ix = 0; ix < n_x; ++ix)
            {
                const int out_idx = ix + iy * n_x + ilay * n_x * n_y;
                const int in_idx = ix + iy * n_x + ilay_in * n_x * n_y;
                var_tmp[out_idx] = var_w_bkg[in_idx];
            }
        }
    }
    var = var_tmp;
}

void restore_bkg_profile_bundle(const int n_col_x, const int n_col_y,
    const int n_lay, const int n_lev,
    const int n_lay_tot, const int n_lev_tot,
    const int n_z_in, const int n_zh_in,
    const int bkg_start_z, const int bkg_start_zh,
    Array<Float,2>* p_lay_copy, Array<Float,2>* t_lay_copy, Array<Float,2>* p_lev_copy, Array<Float,2>* t_lev_copy,
    Array<Float,2>* lwp_copy, Array<Float,2>* iwp_copy, Array<Float,2>* rel_copy, Array<Float,2>* dei_copy, Array<Float,2>* rh_copy,
    Gas_concs& gas_concs_copy, Aerosol_concs& aerosol_concs_copy,
    Array<Float,2>* p_lay, Array<Float,2>* t_lay, Array<Float,2>* p_lev, Array<Float,2>* t_lev,
    Array<Float,2>* lwp, Array<Float,2>* iwp, Array<Float,2>* rel, Array<Float,2>* dei, Array<Float,2>* rh,
    Gas_concs& gas_concs, Aerosol_concs& aerosol_concs,
    const std::vector<std::string>& gas_names, const std::vector<std::string>& aerosol_names,
    bool switch_liq_cloud_optics, bool switch_ice_cloud_optics, bool switch_aerosol_optics
)
{
    const int n_col = n_col_x*n_col_y;

    if (switch_liq_cloud_optics)
    {
        restore_bkg_profile(n_col_x, n_col_y, n_lay, n_z_in, bkg_start_z, lwp_copy->v(), lwp->v());
        lwp_copy->expand_dims({n_col, n_lay_tot});
        restore_bkg_profile(n_col_x, n_col_y, n_lay, n_z_in, bkg_start_z, rel_copy->v(), rel->v());
        rel_copy->expand_dims({n_col, n_lay_tot});
    }
    if (switch_ice_cloud_optics)
    {
        restore_bkg_profile(n_col_x, n_col_y, n_lay, n_z_in, bkg_start_z, iwp_copy->v(), iwp->v());
        iwp_copy->expand_dims({n_col, n_lay_tot});
        restore_bkg_profile(n_col_x, n_col_y, n_lay, n_z_in, bkg_start_z, dei_copy->v(), dei->v());
        dei_copy->expand_dims({n_col, n_lay_tot});
    }

    for (const auto& gas_name : gas_names) {
        if (!gas_concs_copy.exists(gas_name)) {
            continue;
        }
        const Array<Float,2>& gas = gas_concs_copy.get_vmr(gas_name);
        const Array<Float,2>& gas_full = gas_concs.get_vmr(gas_name);

        std::string var_name = "vmr_" + gas_name;
        if (gas.size() > 1) {
            std::vector<Float> gas_copy = gas.v();
            std::vector<Float> gas_full_copy = gas_full.v();
            restore_bkg_profile(n_col_x, n_col_y,
                                n_lay, n_z_in,
                                bkg_start_z,
                                gas_copy, gas_full_copy);

            Array<Float,2> gas_tmp({n_col, n_lay_tot});
            gas_tmp = std::move(gas_copy);
            gas_tmp.expand_dims({n_col, n_lay_tot});

            gas_concs_copy.set_vmr(gas_name, gas_tmp);

        }
    }
    if (switch_aerosol_optics)
    {
        restore_bkg_profile(n_col_x, n_col_y, n_lay, n_z_in, bkg_start_z, rh_copy->v(), rh->v());
        rh_copy->expand_dims({n_col, n_lay_tot});

        for (const auto& aerosol_name : aerosol_names) {
            if (!aerosol_concs_copy.exists(aerosol_name)) {
                continue;
            }
            const Array<Float,2>& gas = aerosol_concs_copy.get_vmr(aerosol_name);
            const Array<Float,2>& gas_full = aerosol_concs.get_vmr(aerosol_name);

            std::string var_name = "vmr_" + aerosol_name;
            if (gas.size() > 1) {
                std::vector<Float> gas_copy = gas.v();
                std::vector<Float> gas_full_copy = gas_full.v();
                restore_bkg_profile(n_col_x, n_col_y,
                                    n_lay, n_z_in,
                                    bkg_start_z,
                                    gas_copy, gas_full_copy);

                Array<Float,2> gas_tmp({n_col, n_lay_tot});
                gas_tmp = std::move(gas_copy);
                gas_tmp.expand_dims({n_col, n_lay_tot});

                aerosol_concs_copy.set_vmr(aerosol_name, gas_tmp);

            }
        }
    }

    restore_bkg_profile(n_col_x, n_col_y,
                        n_lay, n_z_in,
                        bkg_start_z,
                        p_lay_copy->v(), p_lay->v());
    p_lay_copy->expand_dims({n_col, n_lay_tot});

    restore_bkg_profile(n_col_x, n_col_y,
                        n_lay, n_z_in,
                        bkg_start_z,
                        t_lay_copy->v(), t_lay->v());
    t_lay_copy->expand_dims({n_col, n_lay_tot});

    restore_bkg_profile(n_col_x, n_col_y,
                        n_lev, n_zh_in,
                        bkg_start_zh,
                        p_lev_copy->v(), p_lev->v());
    p_lev_copy->expand_dims({n_col, n_lev_tot});
    restore_bkg_profile(n_col_x, n_col_y,
                        n_lev, n_zh_in,
                        bkg_start_zh,
                        t_lev_copy->v(), t_lev->v());
    t_lev_copy->expand_dims({n_col, n_lev_tot});
}

void post_process_output(const std::vector<ColumnResult>& col_results,
                         const int n_col_x, const int n_col_y,
                         const int n_z, const int n_zh,
                         Array<Float,2>* lwp_out,
                         Array<Float,2>* iwp_out,
                         Array<Float,2>* rel_out,
                         Array<Float,2>* dei_out,
                         const bool switch_liq_cloud_optics,
                         const bool switch_ice_cloud_optics)
{
    const int total_cols = n_col_x * n_col_y;
    for (int idx_x = 0; idx_x < n_col_x; ++idx_x) {
        for (int idx_y = 0; idx_y < n_col_y; ++idx_y) {
            int col_idx = idx_x + idx_y * n_col_x;
            const ColumnResult& col = col_results[col_idx];
            for (int j = 0; j < n_z; ++j) {
                int out_idx = col_idx + j * total_cols;
                if (switch_liq_cloud_optics) {
                    lwp_out->v()[out_idx] = col.lwp.v()[j];
                    rel_out->v()[out_idx] = col.rel.v()[j];
                }
                if (switch_ice_cloud_optics) {
                    iwp_out->v()[out_idx] = col.iwp.v()[j];
                    dei_out->v()[out_idx] = col.dei.v()[j];
                }
            }
        }
    }
}

void compress_columns_weighted_avg(const int n_x, const int n_y,
                      const int n_out,
                      const int n_tilt,
                      const Array<ijk,1>& path,
                      std::vector<Float>& var, std::vector<Float>& p_lev)
{
    std::vector<Float> var_tmp(n_out * n_x * n_y);
    const std::vector<ijk>& tilted_path_v = path.v();

    #pragma omp parallel for
    for (int ilay = 0; ilay < n_out; ++ ilay)
    {
        for (int iy = 0; iy < n_y; ++iy)
        {
            for (int ix = 0; ix < n_x; ++ix)
            {
                const int out_idx = ix + iy * n_x + ilay * n_x * n_y;
                Float t_sum = 0.0;
                Float w_sum = 0.0;
                Float avg = 0.0;
                int num_inputs = 0;

                for (int ilay_tilt = 0; ilay_tilt < n_tilt; ++ilay_tilt)
                {
                    const ijk offset = tilted_path_v[ilay_tilt];
                    if (offset.k == ilay)
                    {
                        int in_idx = ix + iy * n_x + ilay_tilt * n_x * n_y;
                        int in_idx_top = ix + iy * n_x + (ilay_tilt + 1) * n_x * n_y;
                        const Float dp_local =  abs(p_lev[in_idx] - p_lev[in_idx_top]);
                        t_sum += var[in_idx] * dp_local;
                        w_sum += dp_local;
                        avg += var[in_idx];
                        num_inputs += 1;
                    }
                }

                if (w_sum > 1e-6)
                    var_tmp[out_idx] = t_sum / w_sum;
                else
                    var_tmp[out_idx] = avg / num_inputs;
            }
        }
    }
    var = var_tmp;
}

void compress_columns_p_or_t(const int n_x, const int n_y, 
                      const int n_out_lay,  const int n_tilt,
                      const Array<ijk,1>& path,
                      const Array<Float,1>& zh_tilt, const Array<Float,1>& zh,
                      const Array<Float,1>& z,
                      std::vector<Float>& var_lev, std::vector<Float>& var_lay)
{
    std::vector<Float> var_tmp_lay(n_out_lay * n_x * n_y);
    std::vector<Float> var_tmp_lev((n_out_lay + 1) * n_x * n_y);
    const std::vector<ijk>& tilted_path_v = path.v();

    // temperature and pressure at original levels can be taken directly from the same height in the tilted column
    #pragma omp parallel for
    for (int ilev = 0; ilev < n_out_lay + 1; ++ilev)
    {
        for (int iy = 0; iy < n_y; ++iy)
        {
            for (int ix = 0; ix < n_x; ++ix)
            {
                for (int ilev_tilt = 0; ilev_tilt < (n_tilt + 1); ++ilev_tilt)
                {
                    if (std::abs(zh.v()[ilev] - zh_tilt.v()[ilev_tilt]) < 1e-2)  // minimum step size
                    {
                        const int out_idx = ix + iy * n_x + ilev * n_x * n_y;
                        const int in_idx = ix + iy * n_x + ilev_tilt * n_x * n_y;
                        var_tmp_lev[out_idx] = var_lev[in_idx];
                    }
                }
            }
        }
    }

    // height of layer center differs between the original column and the tilted column, so we interpolate between the levels
    // weighted interpolation based on z, as in the interpolation function
    #pragma omp parallel for
    for (int ilay = 0; ilay < n_out_lay; ++ilay)
    {
        for (int iy = 0; iy < n_y; ++iy)
        {
            for (int ix = 0; ix < n_x; ++ix)
            {
                const int out_idx = ix + iy * n_x + ilay * n_x * n_y;
                const float dz =  zh.v()[ilay+1] - zh.v()[ilay];
                var_tmp_lay[out_idx] = (var_tmp_lev[out_idx] * (z.v()[ilay] - zh.v()[ilay]) / dz
                                        + var_tmp_lev[out_idx + 1] * (zh.v()[ilay + 1] - z.v()[ilay]) / dz);
            }
        }
    }
    var_lev = var_tmp_lev;
    var_lay = var_tmp_lay;
}

void tilt_and_compress_fields(const int n_z_in, const int n_zh_in, const int n_col_x, const int n_col_y,
    const int n_z_tilt, const int n_zh_tilt, const int n_col,
    const Array<Float,1>& zh, const Array<Float,1>& z,
    const Array<Float,1>& zh_tilt, const Array<ijk,1>& path,
    Array<Float,2>* p_lay_copy, Array<Float,2>* t_lay_copy, Array<Float,2>* p_lev_copy, Array<Float,2>* t_lev_copy, 
    Array<Float,2>* rh_copy, 
    Gas_concs& gas_concs_copy, const std::vector<std::string>& gas_names,
    Aerosol_concs& aerosol_concs_copy, const std::vector<std::string>& aerosol_names, const bool switch_aerosol_optics)
{
    // Create tilted columns for T and p. Important: create T first!!
    create_tilted_columns_levlay(n_col_x, n_col_y, n_z_in, n_zh_in, zh.v(), z.v(), zh_tilt.v(), path.v(), t_lay_copy->v(), t_lev_copy->v());
    t_lay_copy->expand_dims({n_col, n_z_tilt});
    t_lev_copy->expand_dims({n_col, n_zh_tilt});
    compress_columns_p_or_t(n_col_x, n_col_y, n_z_in, n_z_tilt,
                            path,
                            zh_tilt, zh,
                            z,
                            t_lev_copy->v(), t_lay_copy->v());
    t_lay_copy->expand_dims({n_col, n_z_in});
    t_lev_copy->expand_dims({n_col, n_zh_in});

    create_tilted_columns_levlay(n_col_x, n_col_y, n_z_in, n_zh_in, zh.v(), z.v(), zh_tilt.v(), path.v(), p_lay_copy->v(), p_lev_copy->v());
    p_lay_copy->expand_dims({n_col, n_z_tilt});
    p_lev_copy->expand_dims({n_col, n_zh_tilt});
    // do not compress P here, as pressure at tilted levels is required for weighting of gasses and aerosol

    for (int ilay=0; ilay<n_zh_tilt; ++ilay) {
        for (int iy=0; iy<n_col_y; ++iy) {
            for (int ix=0; ix<n_col_x; ++ix) {
                if (ilay > 0) {
                    const int curr_idx = ix + iy*n_col_x + ilay*n_col_x*n_col_y;
                    const int prev_idx = ix + iy*n_col_x + (ilay-1)*n_col_x*n_col_y;
                    if (p_lev_copy->v()[curr_idx] == p_lev_copy->v()[prev_idx]) {
                        p_lev_copy->v()[curr_idx] = p_lev_copy->v()[prev_idx] * 0.99999;
                    }
                    else if (p_lev_copy->v()[curr_idx] > p_lev_copy->v()[prev_idx])
                    {
                        throw std::runtime_error("Pressure INCREASED at layer " + std::to_string(ilay));
                    }
                }
            }
        }
    }

    // gasses
    for (const auto& gas_name : gas_names) {
        if (!gas_concs_copy.exists(gas_name)) {
            continue;
        }
        const Array<Float,2>& gas = gas_concs_copy.get_vmr(gas_name);
        if (gas.size() > 1) {
            if (gas.get_dims()[0] > 1) { // checking: do we have 3D field?
                Array<Float,2> gas_tmp(gas);
                create_tilted_columns(n_col_x, n_col_y, n_z_in, n_zh_in, zh_tilt.v(), path.v(), gas_tmp.v());
                gas_tmp.expand_dims({n_col, n_z_tilt});
                compress_columns_weighted_avg(n_col_x, n_col_y,
                                              n_z_in, n_z_tilt,
                                              path,
                                              gas_tmp.v(),
                                              p_lev_copy->v());
                gas_tmp.expand_dims({n_col, n_z_in});
                gas_concs_copy.set_vmr(gas_name, gas_tmp);
            }
            else {
                throw std::runtime_error("No tilted column implementation for single column profiles.");
            }
        } 
    }

    // aerosols
    if (switch_aerosol_optics)
    {
        // should we just recompute relative humidity after tilting water vapour and temperature?
        create_tilted_columns(n_col_x, n_col_y, n_z_in, n_zh_in, zh_tilt.v(), path.v(), rh_copy->v());
        rh_copy->expand_dims({n_col, n_z_tilt});
        compress_columns_weighted_avg(n_col_x, n_col_y,
                                      n_z_in, n_z_tilt,
                                      path,
                                      rh_copy->v(),
                                      p_lev_copy->v());
        rh_copy->expand_dims({n_col, n_z_in});

        for (const auto& aerosol_name : aerosol_names) {
            if (!aerosol_concs_copy.exists(aerosol_name)) {
                continue;
            }
            const Array<Float,2>& aerosol = aerosol_concs_copy.get_vmr(aerosol_name);

            if (aerosol.size() > 1) {
                if (aerosol.get_dims()[0] > 1) { // checking: do we have 3D field?
                    Array<Float,2> gas_tmp(aerosol);
                    create_tilted_columns(n_col_x, n_col_y, n_z_in, n_zh_in, zh_tilt.v(), path.v(), gas_tmp.v());
                    gas_tmp.expand_dims({n_col, n_z_tilt});
                    compress_columns_weighted_avg(n_col_x, n_col_y,
                                                                  n_z_in, n_z_tilt,
                                                                  path,
                                                                  gas_tmp.v(),
                                                                  p_lev_copy->v());
                    gas_tmp.expand_dims({n_col, n_z_in});
                    aerosol_concs_copy.set_vmr(aerosol_name, gas_tmp);
                }
                else {
                    throw std::runtime_error("No tilted column implementation for single column profiles.");
                }
            } 
        }
    }

    // compression of P at the end, as pressure at tilted levels is required for weighting of gasses and aerosol
    compress_columns_p_or_t(n_col_x, n_col_y, n_z_in, n_z_tilt,
                            path,
                            zh_tilt, zh,
                            z,
                            p_lev_copy->v(), p_lay_copy->v());
    p_lay_copy->expand_dims({n_col, n_z_in});
    p_lev_copy->expand_dims({n_col, n_zh_in});
}

void create_tilted_columns(const int n_x, const int n_y, const int n_lay_in, const int n_lev_in,
                           const std::vector<Float>& zh_tilted, const std::vector<ijk>& tilted_path,
                           std::vector<Float>& var)
{
    const int n_lay = tilted_path.size();
    const int n_lev = zh_tilted.size();

    std::vector<Float> var_tmp(n_lay*n_x*n_y);

    #pragma omp parallel for
    for (int ilay=0; ilay<n_lay; ++ilay)
    {
        const ijk offset = tilted_path[ilay];
        for (int iy=0; iy<n_y; ++iy)
            for (int ix=0; ix<n_x; ++ix)
            {
                const int idx_out  = ix + iy*n_y + ilay*n_y*n_x;
                const int idx_in = (ix + offset.i)%n_x + (iy+offset.j)%n_y * n_x + offset.k*n_y*n_x;
                var_tmp[idx_out] = var[idx_in];
            }
    }

    var.resize(var_tmp.size());
    var = var_tmp;
}

void interpolate(const int n_x, const int n_y, const int n_lay_in, const int n_lev_in,
                 const std::vector<Float>& zh_in, const std::vector<Float>& zf_in,
                 const std::vector<Float>& play_in, const std::vector<Float>& plev_in,
                 const Float zp, const ijk offset,
                 Float* p_out)
{
        int posh_bot = 0;
        int posf_bot = 0;
        for (int ilev=0; ilev<n_lev_in-1; ++ilev)
            if (zh_in[ilev] < zp)
                posh_bot = ilev;
        for (int ilay=0; ilay<n_lay_in; ++ilay)
            if (zf_in[ilay] < zp)
                posf_bot = ilay;
        const Float* p_top;
        const Float* p_bot;
        Float  z_top;
        Float  z_bot;

        const int zh_top = zh_in[posh_bot+1];
        const int zh_bot = zh_in[posh_bot];
        const int zf_top = (posf_bot+1 < n_lay_in) ? zf_in[posf_bot+1] : zh_top+1;
        const int zf_bot = zf_in[posf_bot];

        // tilted layers between the lowest level and layer
        if (posf_bot == 0 && posh_bot == 0 && zp < zf_in[0])
        {
            p_top = &play_in.data()[(posf_bot)*n_x*n_y];
            z_top = zf_in[posf_bot];
            p_bot = &plev_in.data()[(posh_bot)*n_x*n_y];
            z_bot = zh_in[posh_bot];
        }
        // tilted layers between the top layer and level
        else if (posf_bot == n_lay_in - 1 && posh_bot == n_lev_in -2)
        {
            p_top = &plev_in.data()[(posh_bot + 1) * n_x * n_y];
            z_top = zh_in[posh_bot + 1];
            p_bot = &play_in.data()[(posf_bot) * n_x * n_y];
            z_bot = zf_in[posf_bot];
        }
        // all layers in between
        else
        {
            if (zh_top > zf_top)
            {
                p_top = &play_in.data()[(posf_bot + 1) * n_x * n_y];
                z_top = zf_in[posf_bot + 1];
            }
            else
            {
                p_top = &plev_in.data()[(posh_bot + 1) * n_x * n_y];
                z_top = zh_in[posh_bot + 1];
            }
            if (zh_bot < zf_bot)
            {
                p_bot = &play_in.data()[(posf_bot) * n_x * n_y];
                z_bot = zf_in[posf_bot];
            }
            else
            {
                p_bot = &plev_in.data()[(posh_bot) * n_x * n_y];
                z_bot = zh_in[posh_bot];
            }
        }

        Float dz = z_top-z_bot;

        for (int iy=0; iy<n_y; ++iy)
            for (int ix=0; ix<n_x; ++ix)
            {
                const int idx_out  = ix + iy*n_y;
                const int idx_in = (ix + offset.i)%n_x + (iy+offset.j)%n_y * n_x;
                const Float pres = (zp-z_bot)/dz*p_top[idx_in] + (z_top-zp)/dz*p_bot[idx_in];
                p_out[idx_out] = pres;
            }
}


void create_tilted_columns_levlay(const int n_x, const int n_y, const int n_lay_in, const int n_lev_in,
                                 const std::vector<Float>& zh_in, const std::vector<Float>& z_in,
                                 const std::vector<Float>& zh_tilted, const std::vector<ijk>& tilted_path,
                                 std::vector<Float>& var_lay, std::vector<Float>& var_lev)

{
    const int n_lay = tilted_path.size();
    const int n_lev = zh_tilted.size();
    std::vector<Float> z_tilted(n_lay);
    for (int ilay=0; ilay<n_lay; ++ilay)
        z_tilted[ilay] = (zh_tilted[ilay]+zh_tilted[ilay+1])/Float(2.);

    std::vector<Float> var_lay_tmp(n_lay*n_x*n_y);
    std::vector<Float> var_lev_tmp(n_lev*n_x*n_y);
    for (int iy=0; iy<n_y; ++iy)
        for (int ix=0; ix<n_x; ++ix)
        {
            const int idx = ix + iy*n_y;
            var_lev_tmp[idx] = var_lev[idx];
        }

    #pragma omp parallel for
    for (int ilev=1; ilev<n_lev; ++ilev)
        interpolate(n_x, n_y, n_lay_in, n_lev_in, zh_in, z_in, var_lay, var_lev, zh_tilted[ilev], tilted_path[ilev-1], &var_lev_tmp.data()[ilev*n_y*n_x]);
    #pragma omp parallel for
    for (int ilay=0; ilay<n_lay; ++ilay)
        interpolate(n_x, n_y, n_lay_in, n_lev_in, zh_in, z_in, var_lay, var_lev, z_tilted[ilay], tilted_path[ilay], &var_lay_tmp.data()[ilay*n_y*n_x]);

    var_lay.resize(n_lay*n_y*n_x);
    var_lev.resize(n_lev*n_y*n_x);
    var_lay = var_lay_tmp;
    var_lev = var_lev_tmp;
}


void tica_tilt(
        const Float sza, const Float azi,
        const int n_col_x, const int n_col_y, const int n_col,
        const int n_lay, const int n_lev, const int n_z_in, const int n_zh_in ,
        Array<Float,1> xh, Array<Float,1> yh, Array<Float,1> zh, Array<Float,1> z,
        Array<Float,2> p_lay, Array<Float,2> t_lay, Array<Float,2> p_lev, Array<Float,2> t_lev,
        Array<Float,2> lwp, Array<Float,2> iwp, Array<Float,2> rel, Array<Float,2> dei, Array<Float,2> rh,
        Gas_concs gas_concs, Aerosol_concs aerosol_concs,
        Array<Float,2>& p_lay_out, Array<Float,2>& t_lay_out, Array<Float,2>& p_lev_out, Array<Float,2>& t_lev_out,
        Array<Float,2>& lwp_out, Array<Float,2>& iwp_out, Array<Float,2>& rel_out, Array<Float,2>& dei_out, Array<Float,2>& rh_out,
        Gas_concs& gas_concs_out, Aerosol_concs& aerosol_concs_out,
        std::vector<std::string>& gas_names, std::vector<std::string>& aerosol_names,
        bool switch_cloud_optics, bool switch_liq_cloud_optics, bool switch_ice_cloud_optics, bool switch_aerosol_optics,
        int rnd_seed
)
{
    // if t lev all 0, interpolate from t lay
    if (*std::max_element(t_lev_out.v().begin(), t_lev_out.v().end()) <= 0) {
        for (int i = 1; i <= n_col; ++i) {
            for (int j = 2; j <= n_lay; ++j) {
                t_lev_out({i, j}) = (t_lay_out({i, j}) + t_lay_out({i, j - 1})) / 2.0;
            }
            t_lev_out({i, n_lev}) = 2 * t_lay_out({i, n_lay}) - t_lev_out({i,n_lay});
            t_lev_out({i, 1}) = 2 * t_lay_out({i, 1}) - t_lev_out({i,2});
        }
    }
    // copy interpolated values into t_lev too
    t_lev = t_lev_out;

    ////// SETUP FOR CENTER START POINT TILTING //////
    Array<ijk,1> center_path;
    Array<Float,1> center_zh_tilt;
    tilted_path(xh.v(),yh.v(),zh.v(),z.v(),sza,azi, 0.5, 0.5, center_path.v(), center_zh_tilt.v());

    int n_zh_tilt_center = center_zh_tilt.v().size();
    int n_z_tilt_center = n_zh_tilt_center - 1;

    center_path.set_dims({n_z_tilt_center});
    center_zh_tilt.set_dims({n_zh_tilt_center});

    tilt_and_compress_fields(n_z_in, n_zh_in, n_col_x, n_col_y,
                n_z_tilt_center, n_zh_tilt_center, n_col,
                zh, z,
                center_zh_tilt, center_path,
                &p_lay_out, &t_lay_out, &p_lev_out, &t_lev_out, &rh_out,
                gas_concs_out, gas_names, aerosol_concs_out, aerosol_names, switch_aerosol_optics
    );

    ////// SETUP FOR RANDOM START POINT TILTING //////
    if (switch_cloud_optics)
    {
        std::vector<std::pair<Float, Float>> x_y_start_arr(n_col);
        std::vector<Array<ijk,1>> by_col_paths(n_col);
        std::vector<Array<Float,1>> by_col_zh_tilt(n_col);
        std::vector<int> by_col_n_zh_tilt(n_col);
        std::vector<int> by_col_n_z_tilt(n_col);

        std::mt19937_64 rng;
        // uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        // std::seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed>>32)};
        std::seed_seq ss{rnd_seed};
        rng.seed(ss);
        std::uniform_real_distribution<double> unif(0.001, 0.999);
        for (int i = 0; i < n_col; i++)
        {
            Float x_point = unif(rng);
            Float y_point = unif(rng);
            x_y_start_arr[i] = std::make_pair(x_point, y_point);
        }

#pragma omp parallel for
        for (int i = 0; i < n_col; i++) {
            Float x_start = x_y_start_arr[i].first;
            Float y_start = x_y_start_arr[i].second;
            Array<ijk,1> path;
            Array<Float,1> zh_tilt;

            tilted_path(xh.v(), yh.v(), zh.v(), z.v(), sza, azi, x_start, y_start, path.v(), zh_tilt.v());
            int n_zh_tilt = zh_tilt.v().size();
            int n_z_tilt = n_zh_tilt - 1;

            path.set_dims({n_z_tilt});
            zh_tilt.set_dims({n_zh_tilt});

            by_col_paths[i] = path;
            by_col_zh_tilt[i] = zh_tilt;
            by_col_n_zh_tilt[i] = n_zh_tilt;
            by_col_n_z_tilt[i] = n_zh_tilt - 1;
        }

        const int total_iterations = n_col_x * n_col_y;
        std::vector<ColumnResult> col_results(total_iterations);

        // Do Normalization and Reshaping OUTSIDE OF LOOP
        Array<Float,2> lwp_norm_reshaped;
        lwp_norm_reshaped.set_dims({n_z_in, n_col});
        Array<Float,2> rel_reshaped;
        rel_reshaped.set_dims({n_z_in, n_col});
        Array<Float,2> iwp_norm_reshaped;
        iwp_norm_reshaped.set_dims({n_z_in, n_col});
        Array<Float,2> dei_reshaped;
        dei_reshaped.set_dims({n_z_in, n_col});

        for (int icol = 1; icol <= n_col; ++icol)
        {
            for (int ilay = 1; ilay <= n_zh_in; ++ilay)
            {
                Float dz = zh({ilay + 1}) - zh({ilay});
                if (switch_liq_cloud_optics)
                {
                    (lwp_norm_reshaped)({ilay, icol}) = (lwp)({icol, ilay})/dz;
                    (rel_reshaped)({ilay, icol}) = (rel)({icol, ilay});
                }
                if (switch_ice_cloud_optics)
                {
                    (iwp_norm_reshaped)({ilay, icol}) = (iwp)({icol, ilay})/dz;
                    (dei_reshaped)({ilay, icol}) = (dei)({icol, ilay});
                }
            }
        }

        if (switch_liq_cloud_optics){
            const std::vector<Float>& var_lwp = lwp_norm_reshaped.v();
            const std::vector<Float>& var_rel = rel_reshaped.v();

            for (int idx_y = 0; idx_y < n_col_y; idx_y++)
            {
                for (int idx_x = 0; idx_x < n_col_x; idx_x++)
                {
                    int i = idx_x + idx_y * n_col_x;
                    const Array<ijk,1>& tilted_path = by_col_paths[i];
                    const std::vector<ijk>& tilted_path_v = tilted_path.v();
                    const Array<Float,1>& zh_tilt = by_col_zh_tilt[i];
                    const int n_z_tilt = by_col_n_z_tilt[i];

                    std::vector<Float> var_lwp_tmp(n_z_tilt);
                    std::vector<Float> var_rel_tmp(n_z_tilt);
                    std::vector<Float> var_lwp_out(n_z_in);
                    std::vector<Float> var_rel_out(n_z_in);

                    // determine lwp in tilted column (at tilted column layers)
                    for (int ilay=0; ilay < n_z_tilt; ++ilay)
                    {
                        Float dz = zh_tilt({ilay + 1}) - zh_tilt({ilay});
                        const ijk offset = tilted_path_v[ilay];
                        const int i_col_new  = ((idx_y+offset.j)%n_col_y) * n_col_x + ((idx_x + offset.i)%n_col_x);
                        const int idx_in = offset.k + i_col_new*n_z_in;
                        var_lwp_tmp[ilay] = var_lwp[idx_in] * dz;
                        var_rel_tmp[ilay] = var_rel[idx_in];
                    }

                    // compress lwp at tilted column layers to original layers
                    for (int ilay=0; ilay < n_z_in; ++ilay)
                    {
                        Float sum = 0.0;
                        Float t_sum = 0.0;
                        Float w_sum = 0.0;
                        int num_inputs = 0;
                        Float avg = 0.0;

                        for (int ilay_tilt=0; ilay_tilt < n_z_tilt; ++ilay_tilt)
                        {
                            const ijk offset = tilted_path_v[ilay_tilt];
                            if (offset.k == ilay)
                            {
                                sum += var_lwp_tmp[ilay_tilt];
                                t_sum += var_rel_tmp[ilay_tilt] * var_lwp_tmp[ilay_tilt];
                                w_sum += var_lwp_tmp[ilay_tilt];
                                num_inputs += 1;
                                avg += var_rel_tmp[ilay_tilt];
                            }
                        }
                        var_lwp_out[ilay] = sum;
                        if (w_sum > 1e-6)
                            var_rel_out[ilay] = t_sum / w_sum;
                        else
                            var_rel_out[ilay] = avg / num_inputs;
                    }

                    col_results[i].lwp   = std::move(var_lwp_out);
                    col_results[i].rel   = std::move(var_rel_out);
                }
            }
        }

        if (switch_ice_cloud_optics){
            const std::vector<Float>& var_iwp = iwp_norm_reshaped.v();
            const std::vector<Float>& var_dei = dei_reshaped.v();

            for (int idx_y = 0; idx_y < n_col_y; idx_y++)
            {
                for (int idx_x = 0; idx_x < n_col_x; idx_x++)
                {
                    int i = idx_x + idx_y * n_col_x;
                    const Array<ijk,1>& tilted_path = by_col_paths[i];
                    const std::vector<ijk>& tilted_path_v = tilted_path.v();
                    const Array<Float,1>& zh_tilt = by_col_zh_tilt[i];
                    const int n_z_tilt = by_col_n_z_tilt[i];

                    std::vector<Float> var_iwp_tmp(n_z_tilt);
                    std::vector<Float> var_dei_tmp(n_z_tilt);
                    std::vector<Float> var_iwp_out(n_z_in);
                    std::vector<Float> var_dei_out(n_z_in);

                    // determine iwp in tilted column (at tilted column layers)
                    for (int ilay=0; ilay < n_z_tilt; ++ilay)
                    {
                        Float dz = zh_tilt({ilay + 1}) - zh_tilt({ilay});
                        const ijk offset = tilted_path_v[ilay];
                        const int i_col_new  = ((idx_y+offset.j)%n_col_y) * n_col_x + ((idx_x + offset.i)%n_col_x);
                        const int idx_in = offset.k + i_col_new*n_z_in;
                        var_iwp_tmp[ilay] = var_iwp[idx_in] * dz;
                        var_dei_tmp[ilay] = var_dei[idx_in];
                    }

                    // compress iwp at tilted column layers to original layers
                    for (int ilay=0; ilay < n_z_in; ++ilay)
                    {
                        Float sum = 0.0;
                        Float t_sum = 0.0;
                        Float w_sum = 0.0;
                        int num_inputs = 0;
                        Float avg = 0.0;

                        for (int ilay_tilt=0; ilay_tilt < n_z_tilt; ++ilay_tilt)
                        {
                            const ijk offset = tilted_path_v[ilay_tilt];
                            if (offset.k == ilay)
                            {
                                sum += var_iwp_tmp[ilay_tilt];
                                t_sum += var_dei_tmp[ilay_tilt] * var_iwp_tmp[ilay_tilt];
                                w_sum += var_iwp_tmp[ilay_tilt];
                                num_inputs += 1;
                                avg += var_dei_tmp[ilay_tilt];
                            }
                        }
                        var_iwp_out[ilay] = sum;
                        if (w_sum > 1e-6)
                            var_dei_out[ilay] = t_sum / w_sum;
                        else
                            var_dei_out[ilay] = avg / num_inputs;
                    }

                    col_results[i].iwp   = std::move(var_iwp_out);
                    col_results[i].dei   = std::move(var_dei_out);
                }
            }
        }

        post_process_output(col_results, n_col_x, n_col_y, n_z_in, n_zh_in,
                            &lwp_out, &iwp_out, &rel_out, &dei_out,
                            switch_liq_cloud_optics, switch_ice_cloud_optics);

    }

    restore_bkg_profile_bundle(n_col_x, n_col_y,
                               n_lay, n_lev, n_lay, n_lev,
                               n_z_in, n_zh_in, n_z_in, n_zh_in,
                               &p_lay_out, &t_lay_out, &p_lev_out, &t_lev_out,
                               &lwp_out, &iwp_out, &rel_out, &dei_out, &rh_out,
                               gas_concs_out, aerosol_concs_out,
                               &p_lay, &t_lay, &p_lev, &t_lev,
                               &lwp, &iwp, &rel, &dei, &rh,
                               gas_concs, aerosol_concs,
                               gas_names, aerosol_names,
                               switch_liq_cloud_optics, switch_ice_cloud_optics, switch_aerosol_optics
    );
}

void translate_fluxes(const int n_x, const int n_y, const int n_lev_in,
                      const Array<Float,1>& zh_tilt, const Array<Float,1>& zh,
                      const std::vector<ijk>& tilted_path, Array<Float,2>& flux)
{
    std::vector<Float> flux_tmp(n_lev_in*n_x*n_y, Float(0.));

    const int idx_top = zh.v().size() -1;   // -1 as zh is one longer than z (at which tilted path is defined)
    for (int ilev=0; ilev<idx_top; ++ilev)
    {
        for (int k=0; k<zh_tilt.v().size(); ++k)
        {
            if (std::abs(zh_tilt.v()[k] - zh.v()[ilev]) < 1e-2)
            {
                const ijk offset = tilted_path[k];
                for (int iy = 0; iy < n_y; ++iy)
                    for (int ix = 0; ix < n_x; ++ix)
                    {
                        const int idx_out = ix + iy * n_x + ilev * n_y * n_x;
                        const int modulo_x = ix - offset.i < 0 ? (ix - offset.i)% n_x + n_x: (ix - offset.i)% n_x;
                        const int modulo_y = iy - offset.j < 0 ? (iy - offset.j)% n_y + n_y: (iy - offset.j)% n_y;
                        const int idx_in = modulo_x + modulo_y * n_x + ilev * n_y * n_x;
                        flux_tmp[idx_out] = flux.ptr()[idx_in];
                    }
            }
        }
    }

    const int idx_top_tilt = zh_tilt.v().size()-2;       // -2: -1 for index starting at zero and -1 as zh is one longer than z
    const ijk offset_tod = tilted_path[idx_top_tilt];
    for (int ilev=idx_top; ilev < n_lev_in; ++ilev)
    {
        for (int iy = 0; iy < n_y; ++iy)
            for (int ix = 0; ix < n_x; ++ix)
            {
                const int idx_out = ix + iy * n_x + ilev * n_y * n_x;
                const int modulo_x = ix - offset_tod.i < 0 ? (ix - offset_tod.i)% n_x + n_x: (ix - offset_tod.i)% n_x;
                const int modulo_y = iy - offset_tod.j < 0 ? (iy - offset_tod.j)% n_y + n_y: (iy - offset_tod.j)% n_y;
                const int idx_in = modulo_x + modulo_y * n_x + ilev * n_y * n_x;
                flux_tmp[idx_out] = flux.ptr()[idx_in];
            }
    }

    flux = std::move(flux_tmp);
}

void translate_heating(const int n_x, const int n_y, const int n_z,
                      const Array<Float,1>& zh_tilt, const Array<Float,1>& zh,
                      const std::vector<ijk>& tilted_path, Array<Float,3>& flux)
{
    std::vector<Float> heat_tmp(n_z*n_x*n_y, Float(0.));

    const int idx_top = zh.v().size() -1;   // -1 as zh is one longer than z (at which tilted path is defined)
    for (int ilay=0; ilay<idx_top; ++ilay)
    {
        for (int k=0; k<zh_tilt.v().size(); ++k)
        {
            if (std::abs(zh_tilt.v()[k] - zh.v()[ilay]) < 1e-2)
            {
                const ijk offset = tilted_path[k];
                for (int iy = 0; iy < n_y; ++iy)
                    for (int ix = 0; ix < n_x; ++ix)
                    {
                        const int idx_out = ix + iy * n_x + ilay * n_y * n_x;
                        const int modulo_x = ix - offset.i < 0 ? (ix - offset.i)% n_x + n_x: (ix - offset.i)% n_x;
                        const int modulo_y = iy - offset.j < 0 ? (iy - offset.j)% n_y + n_y: (iy - offset.j)% n_y;
                        const int idx_in = modulo_x + modulo_y * n_x + ilay * n_y * n_x;
                        heat_tmp[idx_out] = flux.ptr()[idx_in];
                    }
            }
        }
    }

    flux = std::move(heat_tmp);
}

void translate_top(const int n_x, const int n_y,
                       const Array<Float,1>& zh_tilt,
                       const std::vector<ijk>& tilted_path, Array<Float,2>& flux)
{
    std::vector<Float> heat_tmp(n_x*n_y, Float(0.));

    const int idx_top_tilt = zh_tilt.v().size()-2;       // -2: -1 for index starting at zero and -1 as zh is one longer than z
    const ijk offset_tod = tilted_path[idx_top_tilt];

    for (int iy = 0; iy < n_y; ++iy)
        for (int ix = 0; ix < n_x; ++ix)
        {
            const int idx_out = ix + iy * n_x;
            const int modulo_x = ix - offset_tod.i < 0 ? (ix - offset_tod.i)% n_x + n_x: (ix - offset_tod.i)% n_x;
            const int modulo_y = iy - offset_tod.j < 0 ? (iy - offset_tod.j)% n_y + n_y: (iy - offset_tod.j)% n_y;
            const int idx_in = modulo_x + modulo_y * n_x;
            heat_tmp[idx_out] = flux.ptr()[idx_in];
        }

    flux = std::move(heat_tmp);
}

void tica_mean(Array<Float,2>& var, const int n_x, const int n_y, const int n_z_in)
{
    const int n_col = n_x * n_y;
    for (int k = 0; k < n_z_in; ++k) {
        // calc mean
        double tmp = 0.;
        double mean;
        for (int iy = 0; iy < n_y; ++iy)
        {
            for (int ix = 0; ix < n_x; ++ix)
            {
                const int idx = ix + iy * n_x + k * n_y * n_x;
                tmp += var.ptr()[idx];
            }
        }
        mean = tmp / n_col;

        // fill field with mean
        for (int iy = 0; iy < n_y; ++iy)
        {
            for (int ix = 0; ix < n_x; ++ix)
            {
                const int idx = ix + iy * n_x + k * n_y * n_x;
                var.ptr()[idx] = mean;
            }
        }
    }
}

void tica_mean(Array<Float,3>& var, const int n_x, const int n_y, const int n_z_in)
{
    const int n_col = n_x * n_y;
    for (int k = 0; k < n_z_in; ++k) {
        // calc mean
        double tmp = 0.;
        double mean;
        for (int iy = 0; iy < n_y; ++iy)
        {
            for (int ix = 0; ix < n_x; ++ix)
            {
                const int idx = ix + iy * n_x + k * n_y * n_x;
                tmp += var.ptr()[idx];
            }
        }
        mean = tmp / n_col;

        // fill field with mean
        for (int iy = 0; iy < n_y; ++iy)
        {
            for (int ix = 0; ix < n_x; ++ix)
            {
                const int idx = ix + iy * n_x + k * n_y * n_x;
                var.ptr()[idx] = mean;
            }
        }
    }
}
