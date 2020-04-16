#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/texture.h>
#include <mitsuba/render/srgb.h>
#include <mitsuba/core/timer.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/core/qmc.h>

#include "sunsky/sunmodel.h"
#include "sunsky/skymodel.h"

#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>

NAMESPACE_BEGIN(mitsuba)

# define SKY_MIN_SPECTRUM_SAMPLES 8
# define SUN_SPECTRUM_SAMPLES 32
// todo: encode spectra directly, rather than upsampling RGB
# define SKY_PIXELFORMAT Bitmap::PixelFormat::RGB

/* Apparent radius of the sun as seen from the earth (in degrees).
   This is an approximation--the actual value is somewhere between
   0.526 and 0.545 depending on the time of year */
#define SUN_APP_RADIUS 0.5358

/**!

.. _emitter-sky:

Sky emitter (:monosp:`sky`)
---------------------------

.. pluginparameters::

 * - turbidity
   - |Float|
   - This parameter determines the amount of aerosol present in the atmosphere.
     Valid range: 1-10. Default: 3, corresponding to a clear sky in a temperate climate.
 * - scale
   - |Float|
   - This parameter can be used to scale the amount of illumination emitted by the sky emitter.
     Default: 1
 * - sun_scale, sky_scale
   - |Float|
   - Scale of the sun and sky illumination, respectively. Default: scale
 * - sun_direction
   - |vector|
   - Override the sun direction in world space. When this value is provided,
     parameters pertaining to the computation of the sun direction (year, hour,
     latitude, etc. are unnecessary. Default: none.
 * - latitude, longitude, timezone
   - |Float|
   - These three parameters specify the oberver's latitude and longitude
     in degrees, and the local timezone offset in hours, which are required
     to compute the sun's position. Default: 35.6894, 139.6917, 9 --- Tokyo, Japan
 * - albedo
   - Color
   - Specifies the ground albedo. Default: 0.2
 * - stretch
   - |Float|
   - Stretch factor to extend emitter below the horizon, must be in [1,2]
     Default: 1, i.e. not used
 * - to_world
   - |transform|
   - Specifies an optional emitter-to-world transformation.  (Default: none, i.e. emitter space = world space)

This plugin provides the physically-based skylight model by
Hosek and Wilkie :cite:`Hosek2012Analytic`. It can be used to
create predictive daylight renderings of scenes under clear skies,
which is useful for architectural and computer vision applications.
The implementation in Mitsuba is based on code that was
generously provided by the authors.

The model has two main parameters: the turbidity of the atmosphere
and the position of the sun.
The position of the sun in turn depends on a number of secondary
parameters, including the ``latitude``, ``longitude``,
and ``timezone`` at the location of the observer, as well as the
current ``year``, ``month``, ``day``, ``hour``,
``minute``, and ``second``.
Using all of these, the elevation and azimuth of the sun are computed
using the PSA algorithm by Blanco et al. :cite:`Blanco2001Computing`,
which is accurate to about 0.5 arcminutes (1/120 degrees).
Note that this algorithm does not account for daylight
savings time where it is used, hence a manual correction of the
time may be necessary.
For detailed coordinate and timezone information of various cities, see
http://www.esrl.noaa.gov/gmd/grad/solcalc .

If desired, the world-space solar vector may also be specified
using the ``sun_direction`` parameter, in which case all of the
previously mentioned time and location parameters become irrelevant.

**Turbidity**, the other important parameter, specifies the aerosol
content of the atmosphere. Aerosol particles cause additional scattering that
manifests in a halo around the sun, as well as color fringes near the
horizon.
Smaller turbidity values (:math:`\sim 1-2`) produce an
arctic-like clear blue sky, whereas larger values (:math:`\sim 8-10`)
create an atmosphere that is more typical of a warm, humid day.
Note that this model does not aim to reproduce overcast, cloudy, or foggy
atmospheres with high corresponding turbidity values. An photographic
environment map may be more appropriate in such cases
The default coordinate system of the emitter associates the up
direction with the :math:`+Y` axis. The east direction is associated with :math:`+X`
and the north direction is equal to :math:`+Z`. To change this coordinate
system, rotations can be applied using the ``to_world`` parameter.

By default, the emitter will not emit any light below the
horizon, which means that these regions are black when
observed directly. By setting the ``stretch`` parameter to values
between 1 and 2, the sky can be extended to cover these directions
as well. This is of course a complete kludge and only meant as a quick
workaround for scenes that are not properly set up.

Instead of evaluating the full sky model every on every radiance query,
the implementation precomputes a low resolution environment map
(512 :math:`\times` 256) of the entire sky that is then forwarded to the
``envmap`` plugin---this dramatically improves rendering
performance. This resolution is generally plenty since the sky radiance
distribution is so smooth, but it can be adjusted manually if
necessary using the ``resolution`` parameter.

Note that while the model encompasses sunrise and sunset configurations,
it does not extend to the night sky, where illumination from stars, galaxies,
and the moon dominate. When started with a sun configuration that lies
below the horizon, the plugin will fail with an error message.

**Physical units and spectral rendering**
Like the ``blackbody`` emission profile,
the sky model introduces physical units into the rendering process.
The radiance values computed by this plugin have units of power (:math:`W`) per
unit area (:math:`m^{-2}`) per steradian (:math:`sr^{-1}`) per unit
wavelength (:math:`nm^{-1}`). If these units are inconsistent with your scene
description, you may use the optional ``scale`` parameter to adjust them.

When Mitsuba is compiled for spectral rendering, the plugin switches
from RGB to a spectral variant of the skylight model, which relies on
precomputed data between 320 and 720 nm sampled at 40nm-increments.

**Ground albedo**
The albedo of the ground (e.g. due to rock, snow, or vegetation) can have a
noticeable and nonlinear effect on the appearance of the sky.
 
 */

template <typename Float, typename Spectrum>
class SkyEmitter final : public Emitter<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Emitter, m_flags, m_world_transform)
    MTS_IMPORT_TYPES(Scene, Shape, Texture)

    static const int SpectrumSamples = is_spectral_v<Spectrum>
        ? std::max(SKY_MIN_SPECTRUM_SAMPLES, (int) scalar_spectrum_t<UnpolarizedSpectrum>::Size)
        : 3;
    using SkyWavelengths = mitsuba::Spectrum<ScalarFloat, SpectrumSamples>;
    using ColorCoefficients = std::conditional_t< is_spectral_v<Spectrum>, ScalarVector3f, ScalarColor3f >;

    using SphericalCoordinates = mitsuba::SphericalCoordinates<ScalarFloat>;

    SkyEmitter(const Properties &props) : Base(props) {
        ScalarFloat scale = props.float_("scale", 1.0f);
        m_sunScale = props.float_("sun_scale", scale);
        m_skyScale = props.float_("sky_scale", scale);
        m_turbidity = props.float_("turbidity", 3.0f);
        m_stretch = props.float_("stretch", 1.0f);
        m_resolution = props.int_("resolution", 512);
        if constexpr (!is_spectral_v<Spectrum>)
            m_albedo = props.color("albedo", ScalarColor3f(0.2f));
        else
            m_albedo = (ScalarVector3f) srgb_model_fetch( props.color("albedo", ScalarColor3f(0.2f)) );
        
        m_sun = computeSunCoordinates<ScalarFloat>(props);
        m_sunRadiusScale = props.float_("sun_radius_scale", 1.0f);
        m_extend = props.bool_("extend", false);

        // note: should not matter, emitter is expanded to subobjets
        m_flags = EmitterFlags::Infinite | EmitterFlags::SpatiallyVarying;

        if (m_stretch < 1 || m_stretch > 2)
            Log(Error, "The stretch parameter must be in the range [1,2]!");

        m_wavelengths = sample_rgb_spectrum(
                math::sample_shifted<SkyWavelengths>(0.5f / SkyWavelengths::Size)
            ).first;
        skyModelChanged();
    }
    ~SkyEmitter() {
        skyModelChanged(true);
    }

    void skyModelChanged(bool cleanOnly = false) {
        if constexpr (!is_spectral_v<Spectrum>) {
            for (int i=0; i<3; ++i) {
                arhosek_tristim_skymodelstate_free(m_state[i]);
                m_state[i] = nullptr;
            }
        } else {
            for (int i=0; i<SpectrumSamples; ++i) {
                arhosekskymodelstate_free(m_state[i]);
                m_state[i] = nullptr;
            }
        }
        if (cleanOnly)
            return;

        ScalarFloat sunElevation = 0.5f * math::Pi<ScalarFloat> - m_sun.elevation;

        if (m_turbidity < 1 || m_turbidity > 10)
            Log(Error, "The turbidity parameter must be in the range [1,10]!");
        if constexpr (!is_spectral_v<Spectrum>) {
        for (size_t i=0; i<m_albedo.Size; ++i) {
            if (m_albedo[i] < 0 || m_albedo[i] > 1)
                Log(Error, "The albedo parameter must be in the range [0,1]!");
        }
        }
        if (sunElevation < 0)
            Log(Error, "The sun is below the horizon -- this is not supported by the sky model.");

        // Instantiate sky model, if needed (disable by setting m_skyScale to zero)
        if (m_skyScale > 0.0f) {
            if constexpr (!is_spectral_v<Spectrum>) {
                for (int i=0; i<3; ++i)
                    m_state[i] = arhosek_rgb_skymodelstate_alloc_init(
                        m_turbidity, m_albedo[i], sunElevation);
            } else {
                for (int i=0; i<SpectrumSamples; ++i) {
                    // note: arbitrary number of samples not supported for vector eval at the moment
                    ScalarFloat band = srgb_model_eval< mitsuba::Spectrum<ScalarFloat, 1> >(m_albedo, m_wavelengths[i])[0];
                    m_state[i] = arhosekskymodelstate_alloc_init(
                        m_turbidity, band, sunElevation);
                }
            }
        }
    }

    std::vector<ref<Object>> expand() const override {
        Timer timer;
        Log(Debug, "Rasterizing skylight emitter to an %ix%i environment map ..",
                m_resolution, m_resolution/2);
        ref<Bitmap> bitmap = new Bitmap(SKY_PIXELFORMAT, struct_type_v<ScalarFloat>,
            ScalarVector2i(m_resolution, m_resolution/2));

        // Evaluate sky model
        if (!(m_skyScale > 0.0f))
            bitmap->clear();
        else {
            ScalarPoint2f factor(math::TwoPi<ScalarFloat> / bitmap->width(),
                math::Pi<ScalarFloat> / bitmap->height());

            ScalarFloat *data = (ScalarFloat *) bitmap->data();
            int bitmapStride = bitmap->width();
            tbb::parallel_for( tbb::blocked_range2d<int>(0, bitmap->height(), 8, 0, bitmap->width(), 16),
            [data, bitmapStride, factor, this](const tbb::blocked_range2d<int>& r) {
                for (int y = r.rows().begin(); y != r.rows().end(); ++y) {
                    ScalarFloat theta = (y+.5f) * factor.y();
                    ScalarFloat *target = data + (y * bitmapStride + r.cols().begin()) * ColorCoefficients::Size;

                    for (int x = r.cols().begin(); x != r.cols().end(); ++x) {
                        ScalarFloat phi = (x+.5f) * factor.x();

                        store_unaligned( target, this->getSkyRadiance(SphericalCoordinates(theta, phi)) );
                        target += ColorCoefficients::Size;
                    }
                }
            } );

            Log(Debug, "Done (took %i ms)", timer.value());

            #if defined(MTS_DEBUG_SUNSKY)
            // Write a debug image for inspection
            {
                int size = 513 /* odd-sized */, border = 2;
                int fsize = size+2*border, hsize = size/2;
                ref<Bitmap> debugBitmap = new Bitmap(Bitmap::PixelFormat::RGB, Struct::Type::Float32, ScalarVector2i(fsize));
                debugBitmap->clear();

                tbb::parallel_for( tbb::blocked_range2d<int>(0, size.y, 8, 0, size.x, 16),
                [debugBitmap, fsize, hsize, this](const tbb::blocked_range2d<int>& r) {
                    for (int y = r.rows().begin(); y != r.rows().end(); ++y) {
                        float *target = debugBitmap->data() + ((y + border) * fsize + r.cols().begin() + border) * 3;

                        for (int x = r.cols().begin(); x != r.cols().end(); ++x) {
                            ScalarFloat xp = -(x - hsize) / (ScalarFloat) hsize;
                            ScalarFloat yp = -(y - hsize) / (ScalarFloat) hsize;

                            ScalarFloat radius = std::sqrt(xp*xp + yp*yp);

                            ColorCoefficients result(0.0f);
                            if (radius < 1) {
                                ScalarFloat theta = radius * 0.5f * math::Pi<ScalarFloat>;
                                ScalarFloat phi = std::atan2(xp, yp);
                                result = this->getSkyRadiance(SphericalCoordinates(theta, phi));
                            }

                            *target++ = (float) result[0];
                            *target++ = (float) result[1];
                            *target++ = (float) result[2];
                        }
                    }
                } );

                ref<FileStream> fs = new FileStream("sky.exr", FileStream::TruncReadWrite);
                debugBitmap->write(Bitmap::OpenEXR, fs);
            }
            #endif
        }

        ref<Base> sunEmitter;
        if (m_sunScale > 0.0f) {
            /* Rasterizing the sphere to an environment map and checking the
               individual pixels for coverage (which is what Mitsuba 0.3.0 did)
               was slow and not very effective; for instance the power varied
               dramatically with resolution changes. Since the sphere generally
               just covers a few pixels, the code below rasterizes it much more
               efficiently by generating a few thousand QMC samples.

               Step 1: compute a *very* rough estimate of how many
               pixel in the output environment map will be covered
               by the sun */

            SphericalCoordinates sun = m_sun;
            sun.elevation *= m_stretch;
            ScalarFrame3f sunFrame = ScalarFrame3f(toSphere(sun));

            ScalarFloat theta = degToRad(ScalarFloat(SUN_APP_RADIUS) / 2);

            if (m_sunRadiusScale == 0) {
                ScalarFloat solidAngle = math::TwoPi<ScalarFloat> * (1 - std::cos(theta));
                ScalarTransform4f trafo = m_world_transform->template eval<ScalarFloat>(0);
    #if 1
                Properties props("directional");
                props.set_vector3f("direction", -(trafo * sunFrame.n));
                //props.set_float("sampling_weight", m_samplingWeight);
                props.set_object("irradiance", getSunSpectrum(solidAngle).get());
    #else
                Properties props("point");
                Float farAway = 2.e16f;
                props.set_point3f("position", farAway * -(trafo * sunFrame.n));
                //props.set_float("sampling_weight", m_samplingWeight);
                props.set_object("intensity", getSunSpectrum(solidAngle * farAway * farAway).get());
    #endif
                sunEmitter = PluginManager::instance()->create_object<Base>(props);
            } else {
                ColorCoefficients sunRadiance = getSunRadiance();

                size_t pixelCount = m_resolution*m_resolution/2;
                ScalarFloat cosTheta = std::cos(theta * m_sunRadiusScale);

                /* Ratio of the sphere that is covered by the sun */
                ScalarFloat coveredPortion = 0.5f * (1 - cosTheta);

                /* Approx. number of samples that need to be generated,
                be very conservative */
                size_t nSamples = (size_t) std::max((ScalarFloat) 100,
                    (pixelCount * coveredPortion * 1000));

                ScalarPoint2f factor(bitmap->width() * math::InvTwoPi<ScalarFloat>,
                    bitmap->height() * math::InvPi<ScalarFloat>);

                ColorCoefficients value = sunRadiance
                    * (math::TwoPi<ScalarFloat> * (1-std::cos(theta)))
                    * static_cast<ScalarFloat>(bitmap->width() * bitmap->height())
                        / (math::TwoPi<ScalarFloat> * math::Pi<ScalarFloat> * nSamples);

                ScalarFloat *data = (ScalarFloat *) bitmap->data();
                int bitmapStride = bitmap->width();
                for (size_t i=0; i<nSamples; ++i) {
                    ScalarVector3f dir = sunFrame.to_world(
                        warp::square_to_uniform_cone(sample02(i), cosTheta) );

                    ScalarFloat sinTheta = safe_sqrt(1-dir.y()*dir.y());
                    SphericalCoordinates sphCoords = fromSphere(dir);

                    ScalarPoint2i pos(
                        std::min(std::max(0, (int) (sphCoords.azimuth * factor.x())), (int) bitmap->width()-1),
                        std::min(std::max(0, (int) (sphCoords.elevation * factor.y())), (int) bitmap->height()-1));

                    ScalarFloat* ptr = data + ColorCoefficients::Size * (pos.x() + pos.y() * bitmapStride);
                    ColorCoefficients val = load_unaligned<ColorCoefficients>(ptr);
                    val += value / std::max((ScalarFloat) 1e-3f, sinTheta);
                    store_unaligned(ptr, val);
                }

            }

            Log(Debug, "Done (sun, totalling %i ms)", timer.value());
        }

        // Instantiate a nested environment map plugin
        Properties props("envmap");
        props.set_pointer("bitmap", bitmap.get());
        props.set_animated_transform("to_world", const_cast<AnimatedTransform*>( m_world_transform.get() ));
        //props.set_float("sampling_weight", m_samplingWeight);
        ref<Base> envmap = PluginManager::instance()->create_object<Base>(props);

        std::vector< ref<Object> > objects = { (ref<Object>) envmap };
        if (sunEmitter)
            objects.push_back( (ref<Object>) sunEmitter );
        return objects;
    }

    ScalarBoundingBox3f bbox() const override {
        /* This emitter does not occupy any particular region
           of space, return an invalid bounding box */
        return ScalarBoundingBox3f();
    }


    void parameters_changed() override {
        skyModelChanged();
        // todo: re-expand or update m_envmap
        NotImplementedError("parameters_changed");
    }

    void traverse(TraversalCallback *callback) override {
        callback->put_parameter("turbidity", m_turbidity);
        callback->put_parameter("albedo", m_albedo);
        callback->put_parameter("sun_elevation", m_sun.elevation);
        callback->put_parameter("sun_azimuth", m_sun.azimuth);
        callback->put_parameter("sun_radius_scale", m_sunRadiusScale);
        callback->put_parameter("sun_scale", m_sunScale);
        callback->put_parameter("sky_scale", m_skyScale);
        callback->put_parameter("stretch", m_stretch);
        callback->put_parameter("resolution", m_resolution);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "SkyEmitter[" << std::endl
            << "  turbidity = " << m_turbidity << "," << std::endl
            << "  sun_scale = " << m_sunScale << "," << std::endl
            << "  sky_scale = " << m_skyScale << "," << std::endl
            << "  sun_radius_scale = " << m_sunRadiusScale << "," << std::endl
            << "  albedo = " << m_albedo << "," << std::endl
			<< "  sunPos = " << string::indent(m_sun.toString()) << "," << std::endl
            << "  stretch = " << m_stretch << "," << std::endl
            << "  extend = " << m_extend << "," << std::endl
            << "  resolution = " << m_resolution << "," << std::endl
            << "]";
        return oss.str();
    }

protected:
    MTS_DECLARE_CLASS()

    /// Environment map resolution in pixels
    int m_resolution;
    /// Constant scale factor applied to the model
    ScalarFloat m_skyScale, m_sunScale, m_sunRadiusScale;
    /// Sky turbidity
    ScalarFloat m_turbidity;
    /// Position of the sun in spherical coordinates
    SphericalCoordinates m_sun;
    /// Stretch factor to extend to the bottom hemisphere
    ScalarFloat m_stretch;
    /// Extend to the bottom hemisphere (super-unrealistic mode)
    bool m_extend;
    /// Ground albedo
    ColorCoefficients m_albedo;

    /// Wavelengths for which the model is computed
    SkyWavelengths m_wavelengths;

    using ModelState = std::conditional_t< is_spectral_v<Spectrum>, ArHosekSkyModelState, ArHosekTristimSkyModelState >;
    /// State vector for the sky model
    ModelState *m_state[SpectrumSamples] = { nullptr };

    /// Resulting nested envmap
    // ref<Base> m_envmap;

    /// Calculates the spectral radiance of the sky in the specified direction.
    ColorCoefficients getSkyRadiance(const SphericalCoordinates &coords) const {
        ScalarFloat theta = coords.elevation / m_stretch;

        if (std::cos(theta) <= 0) {
            if (!m_extend)
                return ColorCoefficients(0.0f);
            else
                theta = 0.5f * math::Pi<ScalarFloat> - math::Epsilon<ScalarFloat>; // super-unrealistic mode
        }

        // Compute the angle between the sun and (theta, phi) in radians
        ScalarFloat cosGamma = std::cos(theta) * std::cos(m_sun.elevation)
            + std::sin(theta) * std::sin(m_sun.elevation)
            * std::cos(coords.azimuth - m_sun.azimuth);

        ScalarFloat gamma = safe_acos(cosGamma);

        SkyWavelengths bins;
        if constexpr (!is_spectral_v<Spectrum>) {
            for (int i=0; i<SpectrumSamples; i++) {
                bins[i] = (ScalarFloat) (arhosek_tristim_skymodel_radiance(m_state[i],
                    theta, gamma, i) / 106.856980); // (sum of Spectrum::CIE_Y)
            }
        } else {
            for (int i=0; i<SpectrumSamples; i++) {
                ScalarFloat val = ScalarFloat(arhosekskymodel_radiance(m_state[i],
                    theta, gamma, m_wavelengths[i]) / 106.856980);
                bins[i] = val;
            }
        }
        bins = max(bins * m_skyScale, SkyWavelengths(0.0f));

        ColorCoefficients result;
        if constexpr (!is_spectral_v<Spectrum>)
            result = (ColorCoefficients) bins;
        else {
            // todo: properly encode spectrum directly and pass to envmap
            result = (ColorCoefficients) spectrum_to_rgb(
                std::vector<ScalarFloat>(m_wavelengths.begin(), m_wavelengths.end()),
                std::vector<ScalarFloat>(bins.begin(), bins.end()),
                false);
        }


        if (m_extend)
            result *= smoothstep(2 - 2*coords.elevation*math::InvPi<ScalarFloat>);

        return result;
    }

    ref<Texture> getSunSpectrum(ScalarFloat scale) const {
        scale *= m_sunScale;
        
        using SunWavelengths = mitsuba::Spectrum<ScalarFloat, SUN_SPECTRUM_SAMPLES>;
        SunWavelengths wavelengths = sample_rgb_spectrum(
                math::sample_shifted<SunWavelengths>(0.5f / SunWavelengths::Size)
            ).first;
        SunWavelengths bins = computeSunRadiance(m_sun.elevation, m_turbidity, wavelengths);
        bins *= scale;

        if constexpr (!is_spectral_v<Spectrum>) {
            ScalarColor3f color = (ColorCoefficients) spectrum_to_rgb(
                std::vector<ScalarFloat>(wavelengths.begin(), wavelengths.end()),
                std::vector<ScalarFloat>(bins.begin(), bins.end()),
                false);
            Properties props("srgb_d65");
            props.set_color("color", color);
            return PluginManager::instance()->create_object<Texture>(props);
        } else {
            Properties props("regular");
            props.set_float("lambda_min", wavelengths[0]);
            props.set_float("lambda_max", wavelengths[wavelengths.Size - 1]);
            props.set_long("size", bins.Size);
            props.set_pointer("values", bins.data());
            return PluginManager::instance()->create_object<Texture>(props);
        }
    }

    ColorCoefficients getSunRadiance() const {
        using SunWavelengths = mitsuba::Spectrum<ScalarFloat, SUN_SPECTRUM_SAMPLES>;
        SunWavelengths wavelengths = sample_rgb_spectrum(
                math::sample_shifted<SunWavelengths>(0.5f / SunWavelengths::Size)
            ).first;
        SunWavelengths bins = computeSunRadiance(m_sun.elevation, m_turbidity, wavelengths);
        bins *= m_sunScale;

        ColorCoefficients result;
        // todo: properly encode spectrum directly and pass to envmap
        result = (ColorCoefficients) spectrum_to_rgb(
            std::vector<ScalarFloat>(wavelengths.begin(), wavelengths.end()),
            std::vector<ScalarFloat>(bins.begin(), bins.end()),
            false);

        return result;
    }

    static ScalarFloat smoothstep(ScalarFloat x) {
        return x * x * (3.0f - 2.0f * x);
    }

    /// Van der Corput radical inverse in base 2 with single precision
    static float radicalInverse2Single(uint32_t n, uint32_t scramble = 0U) {
        /* Efficiently reverse the bits in 'n' using binary operations */
    #if (defined(__GNUC__) && (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 2))) || defined(__clang__)
        n = __builtin_bswap32(n);
    #else
        n = (n << 16) | (n >> 16);
        n = ((n & 0x00ff00ff) << 8) | ((n & 0xff00ff00) >> 8);
    #endif
        n = ((n & 0x0f0f0f0f) << 4) | ((n & 0xf0f0f0f0) >> 4);
        n = ((n & 0x33333333) << 2) | ((n & 0xcccccccc) >> 2);
        n = ((n & 0x55555555) << 1) | ((n & 0xaaaaaaaa) >> 1);

        // Account for the available precision and scramble
        n = (n >> (32 - 24)) ^ (scramble & ~-(1 << 24));

        return (float) n / (float) (1U << 24);
    }

    /// Van der Corput radical inverse in base 2 with double precision
    static double radicalInverse2Double(uint64_t n, uint64_t scramble = 0ULL) {
        /* Efficiently reverse the bits in 'n' using binary operations */
    #if (defined(__GNUC__) && (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 2))) || defined(__clang__)
        n = __builtin_bswap64(n);
    #else
        n = (n << 32) | (n >> 32);
        n = ((n & 0x0000ffff0000ffffULL) << 16) | ((n & 0xffff0000ffff0000ULL) >> 16);
        n = ((n & 0x00ff00ff00ff00ffULL) << 8)  | ((n & 0xff00ff00ff00ff00ULL) >> 8);
    #endif
        n = ((n & 0x0f0f0f0f0f0f0f0fULL) << 4)  | ((n & 0xf0f0f0f0f0f0f0f0ULL) >> 4);
        n = ((n & 0x3333333333333333ULL) << 2)  | ((n & 0xccccccccccccccccULL) >> 2);
        n = ((n & 0x5555555555555555ULL) << 1)  | ((n & 0xaaaaaaaaaaaaaaaaULL) >> 1);

        // Account for the available precision and scramble
        n = (n >> (64 - 53)) ^ (scramble & ~-(1LL << 53));

        return (double) n / (double) (1ULL << 53);
    }

    /// Sobol' radical inverse in base 2 with single precision.
    static float sobol2Single(uint32_t n, uint32_t scramble = 0U) {
        for (uint32_t v = 1U << 31; n != 0; n >>= 1, v ^= v >> 1)
            if (n & 1)
                scramble ^= v;
        return (float) scramble / (float) (1ULL << 32);
    }

    /// Sobol' radical inverse in base 2 with double precision.
    static double sobol2Double(uint64_t n, uint64_t scramble = 0ULL) {
        scramble &= ~-(1LL << 53);
        for (uint64_t v = 1ULL << 52; n != 0; n >>= 1, v ^= v >> 1)
            if (n & 1)
                scramble ^= v;
        return (double) scramble / (double) (1ULL << 53);
    }

    /// Generate an element from a (0, 2) sequence (without scrambling)
    static ScalarPoint2f sample02(size_t n) {
        if constexpr (std::is_same_v<ScalarFloat, float>)
            return ScalarPoint2f(
                radicalInverse2Single((uint32_t) n),
                sobol2Single((uint32_t) n)
            );
        else if constexpr (std::is_same_v<ScalarFloat, double>)
            return ScalarPoint2f(
                radicalInverse2Double((uint64_t) n),
                sobol2Double((uint64_t) n)
            );
        else
            static_assert(false_v<ScalarFloat>, "ScalarFloat unsupported by sample02");
    }
};

MTS_IMPLEMENT_CLASS_VARIANT(SkyEmitter, Emitter)
MTS_EXPORT_PLUGIN(SkyEmitter, "Sky emitter")
NAMESPACE_END(mitsuba)
