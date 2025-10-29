#ifndef TILTED_COLUMN_H
#define TILTED_COLUMN_H

#include <cmath>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "Array.h"
#include "Source_functions.h"

struct ColumnResult {
    Array<Float,1> lwp;
    Array<Float,1> iwp;
    Array<Float,1> rel;
    Array<Float,1> dei;
};

struct ijk
{
    int i;
    int j;
    int k;
};

inline int sign(Float value)
{
    return (Float(0.) < value) - (value < Float(0.));
}

void create_tilted_path(const std::vector<Float>& xh, const std::vector<Float>& yh,
        const std::vector<Float>& zh, const std::vector<Float>& z,
        const Float sza, const Float azi,
        const Float x_start, const Float y_start,
        std::vector<ijk>& tilted_path,
        std::vector<Float>& zh_tilted);

void get_tilted_path_bounds(const int n_z_tilt,
                const std::vector<ijk>& tilted_path,
                std::vector<int>& tilted_path_bounds);

void post_process_output(const std::vector<ColumnResult>& col_results,
        const int n_col_x, const int n_col_y,
        const int n_z, const int n_zh,
        Array<Float,2>* lwp_out,
        Array<Float,2>* iwp_out,
        Array<Float,2>* rel_out,
        Array<Float,2>* dei_out,
        const bool switch_liq_cloud_optics,
        const bool switch_ice_cloud_optics);

void restore_bkg_profile(const int n_x, const int n_y,
                      const int n_full,
                      const int n_tilt,
                      const int bkg_start,
                      std::vector<Float>& var,
                      std::vector<Float>& var_w_bkg);

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
    );

void compress_columns_weighted_avg(const int n_x, const int n_y,
                      const int n_out,
                      const int n_tilt,
                      const Array<ijk,1>& path,
                      std::vector<Float>& var, std::vector<Float>& var_weighting);

void compress_columns_p_or_t(const int n_x, const int n_y,
                      const int n_out_lay,  const int n_tilt,
                      const Array<ijk,1>& path,
                      const Array<Float,1>& zh_tilt, const Array<Float,1>& zh,
                      const Array<Float,1>& z,
                      std::vector<Float>& var_lev, std::vector<Float>& var_lay);

void tilt_and_compress_fields(const int n_z_in, const int n_zh_in, const int n_col_x, const int n_col_y,
    const int n_z_tilt, const int n_zh_tilt, const int n_col,
    const Array<Float,1>& zh, const Array<Float,1>& z,
    const Array<Float,1>& zh_tilt, const Array<ijk,1>& path,
    Array<Float,2>* p_lay_copy, Array<Float,2>* t_lay_copy, Array<Float,2>* p_lev_copy, Array<Float,2>* t_lev_copy,
    Array<Float,2>* rh_copy,
    Gas_concs& gas_concs_copy, const std::vector<std::string>& gas_names,
    Aerosol_concs& aerosol_concs_copy, const std::vector<std::string>& aerosol_names, const bool switch_aerosol_optics
    );

void create_tilted_columns(const int n_x, const int n_y, const int n_lay_in, const int n_lev_in,
                           const std::vector<Float>& zh_tilted, const std::vector<ijk>& tilted_path,
                           std::vector<Float>& var);

void interpolate(const int n_x, const int n_y, const int n_lay_in, const int n_lev_in,
                 const std::vector<Float>& zh_in, const std::vector<Float>& zf_in,
                 const std::vector<Float>& play_in, const std::vector<Float>& plev_in,
                 const Float zp, const ijk offset,
                 Float* p_out);

void create_tilted_columns_levlay(const int n_x, const int n_y, const int n_lay_in, const int n_lev_in,
                                 const std::vector<Float>& zh_in, const std::vector<Float>& z_in,
                                 const std::vector<Float>& zh_tilted, const std::vector<ijk>& tilted_path,
                                 std::vector<Float>& var_lay, std::vector<Float>& var_lev);

void tica_tilt(const Float sza, const Float azi,
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
    int rnd_seed);

void translate_fluxes(const int n_x, const int n_y, const int n_lev_in,
                      const Array<Float,1>& zh_tilt, const Array<Float,1>& zh,
                      const std::vector<ijk>& tilted_path, Array<Float,2>& flux);

void translate_heating(const int n_x, const int n_y, const int n_z,
                      const Array<Float,1>& zh_tilt, const Array<Float,1>& zh,
                      const std::vector<ijk>& tilted_path, Array<Float,3>& flux);

void translate_top(const int n_x, const int n_y,
                       const Array<Float,1>& zh_tilt,
                       const std::vector<ijk>& tilted_path, Array<Float,2>& flux);

void tica_mean(Array<Float,2>& var, const int n_x, const int n_y, const int n_z_in);
void tica_mean(Array<Float,3>& var, const int n_x, const int n_y, const int n_z_in);


/// simplified functions
void create_tilted_columns_simple(const int n_x, const int n_y, const std::vector<ijk>& tilted_path,
                                  std::vector<Float>& var);

void compress_columns_weighted_avg_simple(const int n_x, const int n_y,
                                          const int n_out,
                                          const int n_tilt,
                                          const Array<ijk,1>& path,
                                          std::vector<Float>& var, const std::vector<Float>& zh);

void create_tilted_columns_clouds_simple(const int n_x, const int n_y, const std::vector<ijk>& tilted_path,
                                         std::vector<Float>& water_path, std::vector<Float>& effective_size,
                                         const std::vector<Float>& zh);

void compress_columns_weighted_avg_clouds_simple(const int n_x, const int n_y,
                                                 const int n_out,
                                                 const int n_tilt,
                                                 const Array<ijk,1>& path,
                                                 std::vector<Float>& water_path,
                                                 std::vector<Float>& effective_size,
                                                 const std::vector<Float>& dz_tilt);

void translate_levels(const int n_x, const int n_y, const int n_zh,
                      const Array<Float,1>& zh_tilt, const Array<Float,1>& zh,
                      const std::vector<ijk>& tilted_path, Array<Float,2>& flux, const bool forward);

void translate_layers(const int n_x, const int n_y, const int n_z,
                      const Array<Float,1>& zh_tilt, const Array<Float,1>& zh,
                      const std::vector<ijk>& tilted_path, Array<Float,2>& flux, const bool forward);

void tica_tilt_simple(const int n_col_x, const int n_col_y, const int n_col,
                      const int n_lay, const int n_lev, const int n_z_in, const int n_zh_in ,
                      Array<Float,1> zh,
                      Array<Float,2> p_lay, Array<Float,2> t_lay, Array<Float,2> p_lev, Array<Float,2> t_lev,
                      Array<Float,2> lwp, Array<Float,2> iwp, Array<Float,2> rel, Array<Float,2> dei, Array<Float,2> rh,
                      Gas_concs gas_concs, Aerosol_concs aerosol_concs,
                      Array<Float,2>& p_lay_out, Array<Float,2>& t_lay_out, Array<Float,2>& p_lev_out, Array<Float,2>& t_lev_out,
                      Array<Float,2>& lwp_out, Array<Float,2>& iwp_out, Array<Float,2>& rel_out, Array<Float,2>& dei_out, Array<Float,2>& rh_out,
                      Gas_concs& gas_concs_out, Aerosol_concs& aerosol_concs_out,
                      std::vector<std::string>& gas_names, std::vector<std::string>& aerosol_names,
                      bool switch_cloud_optics, bool switch_liq_cloud_optics, bool switch_ice_cloud_optics, bool switch_aerosol_optics,
                      Array<ijk,1> center_path, Array<Float,1> center_zh_tilt, const int n_z_tilt_center);



#ifdef __CUDACC__
/// GPU functions
void tica_tilt_gpu(const Float sza, const Float azi,
    const int n_col_x, const int n_col_y, const int n_col,
    const int n_lay, const int n_lev, const int n_z_in, const int n_zh_in ,
    const Array_gpu<Float,1>& z, const Array_gpu<Float,1>& zh,
    Array_gpu<Float,2>& p_lay, Array_gpu<Float,2>& t_lay, Array_gpu<Float,2>& p_lev, Array_gpu<Float,2>& t_lev,
    Array_gpu<Float,2>& lwp, Array_gpu<Float,2>& iwp, Array_gpu<Float,2>& rel, Array_gpu<Float,2>& dei, Array_gpu<Float,2>& rh,
    const Array_gpu<ijk,1>& tilted_path, const Array_gpu<int,1>& tilted_path_bounds,
    const Array_gpu<Float,1>& zh_tilt, Array_gpu<Float,1>& p_lev_tilt,
    Gas_concs_gpu& gas_concs, Aerosol_concs_gpu& aerosol_concs,
    std::vector<std::string>& gas_names, std::vector<std::string>& aerosol_names,
    bool switch_cloud_optics, bool switch_liq_cloud_optics, bool switch_ice_cloud_optics, bool switch_aerosol_optics,
    int rnd_seed);

void tica_reverse_gpu(
        const int n_col_x, const int n_col_y, const int n_lay, const int n_lev,
        const int n_z, const int n_z_in, const int n_zh_in ,
        const bool do_tilting, const bool switch_twostream, const bool switch_raytracing,
        const Array_gpu<Float,1>& zh, const Array_gpu<Float,1>& zh_tilt,
        const Array_gpu<ijk,1>& tilted_path, const Array_gpu<int,1>& tilted_path_bounds,
        const Array_gpu<Float,1>& p_lev_tilt,
        const Array_gpu<Float,2>& sw_flux_dn_tilt, const Array_gpu<Float,2>& sw_flux_dn_dir_tilt,
        const Array_gpu<Float,2>& sw_flux_up_tilt, const Array_gpu<Float,2>& sw_flux_net_tilt,
        const Array_gpu<Float,3>& rt_flux_abs_dir_tilt, const Array_gpu<Float,3>& rt_flux_abs_dif_tilt, const Array_gpu<Float,2>& rt_flux_tod_up_tilt,
        Array_gpu<Float,2>& sw_flux_dn, Array_gpu<Float,2>& sw_flux_dn_dir,
        Array_gpu<Float,2>& sw_flux_up, Array_gpu<Float,2>& sw_flux_net,
        Array_gpu<Float,3>& rt_flux_abs_dir, Array_gpu<Float,3>& rt_flux_abs_dif, Array_gpu<Float,2>& rt_flux_tod_up);

    // // esat and qsat functions taken from microHH (https://github.com/microhh/microhh)
    __host__ __device__
    inline Float esat_liq(const Float T)
    {
        constexpr Float c00  = Float(+6.1121000000E+02);
        constexpr Float c10  = Float(+4.4393067270E+01);
        constexpr Float c20  = Float(+1.4279398448E+00);
        constexpr Float c30  = Float(+2.6415206946E-02);
        constexpr Float c40  = Float(+3.0291749160E-04);
        constexpr Float c50  = Float(+2.1159987257E-06);
        constexpr Float c60  = Float(+7.5015702516E-09);
        constexpr Float c70  = Float(-1.5604873363E-12);
        constexpr Float c80  = Float(-9.9726710231E-14);
        constexpr Float c90  = Float(-4.8165754883E-17);
        constexpr Float c100 = Float(+1.3839187032E-18);

        const Float x = min(max(Float(-75.), T-Float(273.15)), Float(50.));       // Limit the temperature range to avoid numerical errors
        return c00+x*(c10+x*(c20+x*(c30+x*(c40+x*(c50+x*(c60+x*(c70+x*(c80+x*(c90+x*c100)))))))));
    }

    __host__ __device__
    inline Float qsat_liq(const Float p, const Float T)
    {
        constexpr Float ep = Float(0.6219718);
        return ep*esat_liq(T)/(p-(Float(1.)-ep)*esat_liq(T));
    }

    __host__ __device__
    inline Float esat_ice(const Float T)
    {
        const Float x= min(max(Float(-100.), T-Float(273.15)), Float(50.));     // Limit the temperature range to avoid numerical errors
        return Float(611.15)*exp(Float(22.452)*x / (Float(272.55)+x));
    }

    __host__ __device__
    inline Float qsat_ice(const Float p, const Float T)
    {
        constexpr Float ep = Float(0.6219718);
        return ep*esat_ice(T)/(p-(Float(1.)-ep)*esat_ice(T));
    }
#endif

#endif
