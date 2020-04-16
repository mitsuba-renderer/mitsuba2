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


                        Version: 1.2, August 23rd, 2012

Version history:

1.2  RGB version added.

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

This code is taken from ART, a rendering research system written in a
mix of C99 / Objective C. Since ART is not a small system and is intended to
be inter-operable with other libraries, and since C does not have namespaces,
the structures and functions in ART all have to have the somewhat wordy
canonical names that begin with Ar.../ar..., like those seen in this example.

Usage information:
==================


Model initialisation
--------------------

A separate ArHosekSkyModelState has to be maintained for each spectral
band you want to use the model for. So in a renderer with num_channels
bands, you would need something like

    ArHosekSkyModelState  * skymodel_state[num_channels];

You then have to allocate these states. In the following code snippet, we
assume that "albedo" is defined as

    double  albedo[num_channels];

with a ground albedo value between [0,1] for each channel. The solar elevation
is given in radians.

    for ( unsigned int i = 0; i < num_channels; i++ )
        skymodel_state[i] =
            arhosekskymodelstate_alloc_init(
                  turbidity,
                  albedo[i],
                  solarElevation
                );


Using the model to generate skydome samples
-------------------------------------------

Generating a skydome radiance spectrum "skydome_result" for a given location
on the skydome determined via the angles theta and gamma works as follows:

    double  skydome_result[num_channels];

    for ( unsigned int i = 0; i < num_channels; i++ )
        skydome_result[i] =
            arhosekskymodel_radiance(
                skymodel_state[i],
                theta,
                gamma,
                channel_center[i]
              );

The variable "channel_center" is assumed to hold the channel center wavelengths
for each of the num_channels samples of the spectrum we are building.


Cleanup after use
-----------------

After rendering is complete, the content of the sky model states should be
disposed of via

        for ( unsigned int i = 0; i < num_channels; i++ )
            arhosekskymodelstate_free( skymodel_state[i] );


CIE XYZ Version of the Model
----------------------------

Usage of the CIE XYZ version of the model is exactly the same, except that
num_channels is of course always 3, and that ArHosekTristimSkyModelState and
arhosek_tristim_skymodel_radiance() have to be used instead of their spectral
counterparts.

RGB Version of the Model
------------------------

RGB version uses sRGB primaries with a linear gamma ramp. The same set of
functions as with the XYZ version is used, except the model is initialized
by calling arhosek_rgb_skymodelstate_alloc_init.

*/

typedef double ArHosekSkyModelConfiguration[9];


// Spectral version of the model


typedef struct ArHosekSkyModelState
{
    ArHosekSkyModelConfiguration  configs[11];
    double                        radiances[11];
}
ArHosekSkyModelState;

ArHosekSkyModelState  * arhosekskymodelstate_alloc_init(
        const double  turbidity,
        const double  albedo,
        const double  elevation
        );

void arhosekskymodelstate_free(
        ArHosekSkyModelState  * state
        );

double arhosekskymodel_radiance(
        ArHosekSkyModelState  * state,
        double                  theta,
        double                  gamma,
        double                  wavelength
        );


// CIE XYZ and RGB versions


typedef struct ArHosekTristimSkyModelState
{
    ArHosekSkyModelConfiguration  configs[3];
    double                        radiances[3];
}
ArHosekTristimSkyModelState;

ArHosekTristimSkyModelState  * arhosek_xyz_skymodelstate_alloc_init(
        const double  turbidity,
        const double  albedo,
        const double  elevation
        );

ArHosekTristimSkyModelState  * arhosek_rgb_skymodelstate_alloc_init(
        const double  turbidity,
        const double  albedo,
        const double  elevation
        );

void arhosek_tristim_skymodelstate_free(
        ArHosekTristimSkyModelState * state
        );

double arhosek_tristim_skymodel_radiance(
        ArHosekTristimSkyModelState * state,
        double                  theta,
        double                  gamma,
        int                     channel
        );
