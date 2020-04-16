/*
This source is published under the following 3-clause BSD license.

Copyright (c) 2012, Lukas Hosek and Alexander Wilkie
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * None of the names of the contributors may be used to endorse or promote
      products derived from this software without specific prior written
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/* ============================================================================

This file is part of a sample implementation of the analytical skylight model
presented in the SIGGRAPH 2012 paper


           "An Analytic Model for Full Spectral Sky-Dome Radiance"

                                    by

                       Lukas Hosek and Alexander Wilkie
                Charles University in Prague, Czech Republic


                        Version: 1.1, July 4th, 2012

Version history:

1.1  The coefficients of the spectral model are now scaled so that the output
     is given in physical units: W / (m^-2 * sr * nm). Also, the output of the
     XYZ model is now no longer scaled to the range [0...1]. Instead, it is
     the result of a simple conversion from spectral data via the CIE 2 degree
     standard observer matching functions. Therefore, after multiplication
     with 683 lm / W, the Y channel now corresponds to luminance in lm.

1.0  Initial release (May 11th, 2012).


Please visit http://cgg.mff.cuni.cz/projects/SkylightModelling/ to check if
an updated version of this code has been published!

============================================================================ */

/*

All instructions on how to use this code are to be found in the accompanying
header file.

*/

#include <mitsuba/mitsuba.h>
#include "skymodel.h"
#include "skymodeldata.h"

typedef double *ArHosekSkyModel_Dataset;
typedef double *ArHosekSkyModel_Radiance_Dataset;

// internal functions

void ArHosekSkyModel_CookConfiguration(
        ArHosekSkyModel_Dataset       dataset,
        ArHosekSkyModelConfiguration  config,
        double                        turbidity,
        double                        albedo,
        double                        solar_elevation
        )
{
    double  * elev_matrix;

    int     int_turbidity = (int)turbidity;
    assert(int_turbidity >= 1 && int_turbidity <= 10);
    double  turbidity_rem = turbidity - (double)int_turbidity;

    solar_elevation = pow(solar_elevation / (M_PI / 2.0), (1.0 / 3.0));

    // alb 0 low turb

    elev_matrix = dataset + ( 9 * 6 * (int_turbidity-1) );


    for( unsigned int i = 0; i < 9; ++i )
    {
        //(1-t).^3* A1 + 3*(1-t).^2.*t * A2 + 3*(1-t) .* t .^ 2 * A3 + t.^3 * A4;
        config[i] =
        (1.0-albedo) * (1.0 - turbidity_rem)
        * ( pow(1.0-solar_elevation, 5.0) * elev_matrix[i]  +
           5.0  * pow(1.0-solar_elevation, 4.0) * solar_elevation * elev_matrix[i+9] +
           10.0*pow(1.0-solar_elevation, 3.0)*pow(solar_elevation, 2.0) * elev_matrix[i+18] +
           10.0*pow(1.0-solar_elevation, 2.0)*pow(solar_elevation, 3.0) * elev_matrix[i+27] +
           5.0*(1.0-solar_elevation)*pow(solar_elevation, 4.0) * elev_matrix[i+36] +
           pow(solar_elevation, 5.0)  * elev_matrix[i+45]);
    }

    // alb 1 low turb
    elev_matrix = dataset + (9*6*10 + 9*6*(int_turbidity-1));
    for(unsigned int i = 0; i < 9; ++i)
    {
        //(1-t).^3* A1 + 3*(1-t).^2.*t * A2 + 3*(1-t) .* t .^ 2 * A3 + t.^3 * A4;
        config[i] +=
        (albedo) * (1.0 - turbidity_rem)
        * ( pow(1.0-solar_elevation, 5.0) * elev_matrix[i]  +
           5.0  * pow(1.0-solar_elevation, 4.0) * solar_elevation * elev_matrix[i+9] +
           10.0*pow(1.0-solar_elevation, 3.0)*pow(solar_elevation, 2.0) * elev_matrix[i+18] +
           10.0*pow(1.0-solar_elevation, 2.0)*pow(solar_elevation, 3.0) * elev_matrix[i+27] +
           5.0*(1.0-solar_elevation)*pow(solar_elevation, 4.0) * elev_matrix[i+36] +
           pow(solar_elevation, 5.0)  * elev_matrix[i+45]);
    }

    if(int_turbidity == 10)
        return;

    // alb 0 high turb
    elev_matrix = dataset + (9*6*(int_turbidity));
    for(unsigned int i = 0; i < 9; ++i)
    {
        //(1-t).^3* A1 + 3*(1-t).^2.*t * A2 + 3*(1-t) .* t .^ 2 * A3 + t.^3 * A4;
        config[i] +=
        (1.0-albedo) * (turbidity_rem)
        * ( pow(1.0-solar_elevation, 5.0) * elev_matrix[i]  +
           5.0  * pow(1.0-solar_elevation, 4.0) * solar_elevation * elev_matrix[i+9] +
           10.0*pow(1.0-solar_elevation, 3.0)*pow(solar_elevation, 2.0) * elev_matrix[i+18] +
           10.0*pow(1.0-solar_elevation, 2.0)*pow(solar_elevation, 3.0) * elev_matrix[i+27] +
           5.0*(1.0-solar_elevation)*pow(solar_elevation, 4.0) * elev_matrix[i+36] +
           pow(solar_elevation, 5.0)  * elev_matrix[i+45]);
    }

    // alb 1 high turb
    elev_matrix = dataset + (9*6*10 + 9*6*(int_turbidity));
    for(unsigned int i = 0; i < 9; ++i)
    {
        //(1-t).^3* A1 + 3*(1-t).^2.*t * A2 + 3*(1-t) .* t .^ 2 * A3 + t.^3 * A4;
        config[i] +=
        (albedo) * (turbidity_rem)
        * ( pow(1.0-solar_elevation, 5.0) * elev_matrix[i]  +
           5.0  * pow(1.0-solar_elevation, 4.0) * solar_elevation * elev_matrix[i+9] +
           10.0*pow(1.0-solar_elevation, 3.0)*pow(solar_elevation, 2.0) * elev_matrix[i+18] +
           10.0*pow(1.0-solar_elevation, 2.0)*pow(solar_elevation, 3.0) * elev_matrix[i+27] +
           5.0*(1.0-solar_elevation)*pow(solar_elevation, 4.0) * elev_matrix[i+36] +
           pow(solar_elevation, 5.0)  * elev_matrix[i+45]);
    }
}

double ArHosekSkyModel_CookRadianceConfiguration(
        ArHosekSkyModel_Radiance_Dataset  dataset,
        double                            turbidity,
        double                            albedo,
        double                            solar_elevation
        )
{
    double* elev_matrix;

    int int_turbidity = (int)turbidity;
    assert(int_turbidity >= 1 && int_turbidity <= 10);
    double turbidity_rem = turbidity - (double)int_turbidity;
    double res;
    solar_elevation = pow(solar_elevation / (M_PI / 2.0), (1.0 / 3.0));

    // alb 0 low turb
    elev_matrix = dataset + (6*(int_turbidity-1));
    //(1-t).^3* A1 + 3*(1-t).^2.*t * A2 + 3*(1-t) .* t .^ 2 * A3 + t.^3 * A4;
    res = (1.0-albedo) * (1.0 - turbidity_rem) *
        ( pow(1.0-solar_elevation, 5.0) * elev_matrix[0] +
         5.0*pow(1.0-solar_elevation, 4.0)*solar_elevation * elev_matrix[1] +
         10.0*pow(1.0-solar_elevation, 3.0)*pow(solar_elevation, 2.0) * elev_matrix[2] +
         10.0*pow(1.0-solar_elevation, 2.0)*pow(solar_elevation, 3.0) * elev_matrix[3] +
         5.0*(1.0-solar_elevation)*pow(solar_elevation, 4.0) * elev_matrix[4] +
         pow(solar_elevation, 5.0) * elev_matrix[5]);

    // alb 1 low turb
    elev_matrix = dataset + (6*10 + 6*(int_turbidity-1));
    //(1-t).^3* A1 + 3*(1-t).^2.*t * A2 + 3*(1-t) .* t .^ 2 * A3 + t.^3 * A4;
    res += (albedo) * (1.0 - turbidity_rem) *
        ( pow(1.0-solar_elevation, 5.0) * elev_matrix[0] +
         5.0*pow(1.0-solar_elevation, 4.0)*solar_elevation * elev_matrix[1] +
         10.0*pow(1.0-solar_elevation, 3.0)*pow(solar_elevation, 2.0) * elev_matrix[2] +
         10.0*pow(1.0-solar_elevation, 2.0)*pow(solar_elevation, 3.0) * elev_matrix[3] +
         5.0*(1.0-solar_elevation)*pow(solar_elevation, 4.0) * elev_matrix[4] +
         pow(solar_elevation, 5.0) * elev_matrix[5]);
    if(int_turbidity == 10)
        return res;

    // alb 0 high turb
    elev_matrix = dataset + (6*(int_turbidity));
    //(1-t).^3* A1 + 3*(1-t).^2.*t * A2 + 3*(1-t) .* t .^ 2 * A3 + t.^3 * A4;
    res += (1.0-albedo) * (turbidity_rem) *
        ( pow(1.0-solar_elevation, 5.0) * elev_matrix[0] +
         5.0*pow(1.0-solar_elevation, 4.0)*solar_elevation * elev_matrix[1] +
         10.0*pow(1.0-solar_elevation, 3.0)*pow(solar_elevation, 2.0) * elev_matrix[2] +
         10.0*pow(1.0-solar_elevation, 2.0)*pow(solar_elevation, 3.0) * elev_matrix[3] +
         5.0*(1.0-solar_elevation)*pow(solar_elevation, 4.0) * elev_matrix[4] +
         pow(solar_elevation, 5.0) * elev_matrix[5]);

    // alb 1 high turb
    elev_matrix = dataset + (6*10 + 6*(int_turbidity));
    //(1-t).^3* A1 + 3*(1-t).^2.*t * A2 + 3*(1-t) .* t .^ 2 * A3 + t.^3 * A4;
    res += (albedo) * (turbidity_rem) *
        ( pow(1.0-solar_elevation, 5.0) * elev_matrix[0] +
         5.0*pow(1.0-solar_elevation, 4.0)*solar_elevation * elev_matrix[1] +
         10.0*pow(1.0-solar_elevation, 3.0)*pow(solar_elevation, 2.0) * elev_matrix[2] +
         10.0*pow(1.0-solar_elevation, 2.0)*pow(solar_elevation, 3.0) * elev_matrix[3] +
         5.0*(1.0-solar_elevation)*pow(solar_elevation, 4.0) * elev_matrix[4] +
         pow(solar_elevation, 5.0) * elev_matrix[5]);
    return res;
}

double ArHosekSkyModel_GetRadianceInternal(
        ArHosekSkyModelConfiguration  configuration,
        double                        theta,
        double                        gamma
        )
{
    const double expM = exp(configuration[4] * gamma);
    const double rayM = cos(gamma)*cos(gamma);
    const double mieM = (1.0 + cos(gamma)*cos(gamma)) / pow((1.0 + configuration[8]*configuration[8] - 2.0*configuration[8]*cos(gamma)), 1.5);
    const double zenith = sqrt(cos(theta));

    return (1.0 + configuration[0] * exp(configuration[1] / (cos(theta) + 0.01))) *
            (configuration[2] + configuration[3] * expM + configuration[5] * rayM + configuration[6] * mieM + configuration[7] * zenith);
}

// spectral version

ArHosekSkyModelState  * arhosekskymodelstate_alloc_init(
        const double  turbidity,
        const double  albedo,
        const double  elevation
        )
{
    ArHosekSkyModelState  * state = new ArHosekSkyModelState();

    for( unsigned int wl = 0; wl < 11; ++wl )
    {
        ArHosekSkyModel_CookConfiguration(
            datasets[wl],
            state->configs[wl],
            turbidity,
            albedo,
            elevation
            );

        state->radiances[wl] =
            ArHosekSkyModel_CookRadianceConfiguration(
                datasetsRad[wl],
                turbidity,
                albedo,
                elevation
                );
    }

    return state;
}

void arhosekskymodelstate_free(
        ArHosekSkyModelState  * state
        )
{
    delete state;
}

double arhosekskymodel_radiance(
        ArHosekSkyModelState  * state,
        double                  theta,
        double                  gamma,
        double                  wavelength
        )
{
    int low_wl = static_cast<int>((wavelength - 320.0) / 40.0);
    if (low_wl < 0 || low_wl >= 11)
        return 0.0;

    double interp = fmod((wavelength - 320.0 ) / 40.0, 1.0);

    double val_low =
          ArHosekSkyModel_GetRadianceInternal(
                state->configs[low_wl],
                theta,
                gamma
              )
        * state->radiances[low_wl];

    if(interp < 1e-6)
        return val_low;

    double result = (1.0 - interp) * val_low;

    if (low_wl+1 < 11)
        result += interp * ArHosekSkyModel_GetRadianceInternal(
            state->configs[low_wl+1], theta, gamma) * state->radiances[low_wl+1];

    return result;
}


// xyz and versions

ArHosekTristimSkyModelState  * arhosek_xyz_skymodelstate_alloc_init(
    const double  turbidity,
    const double  albedo,
    const double  elevation
    )
{
    ArHosekTristimSkyModelState  * state = new ArHosekTristimSkyModelState();

    for( unsigned int channel = 0; channel < 3; ++channel )
    {
        ArHosekSkyModel_CookConfiguration(
            datasetsXYZ[channel],
            state->configs[channel],
            turbidity,
            albedo,
            elevation
            );

        state->radiances[channel] =
        ArHosekSkyModel_CookRadianceConfiguration(
            datasetsXYZRad[channel],
            turbidity,
            albedo,
            elevation
            );
    }

    return state;
}

ArHosekTristimSkyModelState  * arhosek_rgb_skymodelstate_alloc_init(
    const double  turbidity,
    const double  albedo,
    const double  elevation
    )
{
    ArHosekTristimSkyModelState  * state = new ArHosekTristimSkyModelState();

    for( unsigned int channel = 0; channel < 3; ++channel )
    {
        ArHosekSkyModel_CookConfiguration(
            datasetsRGB[channel],
            state->configs[channel],
            turbidity,
            albedo,
            elevation
            );

        state->radiances[channel] =
        ArHosekSkyModel_CookRadianceConfiguration(
            datasetsRGBRad[channel],
            turbidity,
            albedo,
            elevation
            );
    }

    return state;
}

void arhosek_tristim_skymodelstate_free(
    ArHosekTristimSkyModelState  * state
    )
{
    delete state;
}

double arhosek_tristim_skymodel_radiance(
    ArHosekTristimSkyModelState  * state,
    double                  theta,
    double                  gamma,
    int                     channel
    )
{
    return
    ArHosekSkyModel_GetRadianceInternal(
        state->configs[channel],
        theta,
        gamma
        )
    * state->radiances[channel];
}
