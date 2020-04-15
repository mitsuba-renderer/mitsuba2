#include <mitsuba/core/bsphere.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/texture.h>
#include "sunsky/sunmodel.h"
#include "sunsky/skymodel.h"

NAMESPACE_BEGIN(mitsuba)

/**!

.. _emitter-sky:

Skylight emitter (:monosp:`sky`)
-------------------------------------------------

.. pluginparameters::

 * - radiance
   - |spectrum|
   - Specifies the emitted radiance in units of power per unit area per unit steradian.
     (Default: :ref:`emitter-d65`)

This plugin implements a skylight emitter, which surrounds
the scene and radiates diffuse illumination towards it. This is often
a good default light source when the goal is to visualize some loaded
geometry that uses basic (e.g. diffuse) materials.

 */

template <typename Float, typename Spectrum>
class SkyEmitter final : public Emitter<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Emitter, m_flags)
    MTS_IMPORT_TYPES(Scene, Shape, Texture)

    ~SkyEmitter() {
        for (size_t i = 0; i < Spectrum::Size; ++i)
            arhosekskymodelstate_free(m_state[i]);
    }

    SkyEmitter(const Properties &props) : Base(props) {
        /* Until `set_scene` is called, we have no information
           about the scene and default to the unit bounding sphere. */
        m_bsphere = ScalarBoundingSphere3f(ScalarPoint3f(0.f), 1.f);

        m_flags = +EmitterFlags::Infinite;

        m_scale = props.float_("scale", 1.0f);
        m_turbidity = props.float_("turbidity", 3.0f);
        m_stretch = props.float_("stretch", 1.0f);
        m_resolution = props.int_("resolution", 512);
        m_sun = compute_sun_coordinates<Float>(props);
        m_extend = props.bool_("extend", false);
        m_albedo = props.texture<Texture>("albedo", 0.2f);

        SurfaceInteraction3f si;
        si.wavelengths = 0/0;   // TODO : change when implementing spectral

        UnpolarizedSpectrum albedo = m_albedo->eval(si);

        if (m_turbidity < 1 || m_turbidity > 10)
            Log(Error, "The turbidity parameter must be in the range[1,10]!");
        if (m_stretch < 1 || m_stretch > 2)
            Log(Error, "The stretch parameter must be in the range [1,2]!");
        for (size_t i = 0; i < Spectrum::Size; ++i) {
            if (albedo[i] < 0 || albedo[i] > 1)
                Log(Error, "The albedo parameter must be in the range [0,1]!");
        }

        Float sun_elevation = 0.5f * math::Pi<Float> - m_sun.elevation;

        if (sun_elevation < 0)
            Log(Error, "The sun is below the horizon -- this is not supported by the sky model.");

        for (size_t i = 0; i < Spectrum::Size; i++) {
            if constexpr (Spectrum::Size == 3)
                m_state[i] = arhosek_rgb_skymodelstate_alloc_init(
                    (double)m_turbidity, (double)albedo[i], (double)sun_elevation);
            else
                m_state[i] = arhosekskymodelstate_alloc_init(
                    (double)sun_elevation, (double)m_turbidity, (double)albedo[i]);
        }
    }

    void set_scene(const Scene *scene) override {
        m_bsphere = scene->bbox().bounding_sphere();
        m_bsphere.radius = max(math::RayEpsilon<Float>,
                               m_bsphere.radius * (1.f + math::RayEpsilon<Float>));
    }

    Spectrum eval(const SurfaceInteraction3f &si, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::EndpointEvaluate, active);

        // Compute spherical coords of wi
        SphericalCoordinates coords = from_sphere(si.wi);
        
        Float theta =  (math::Pi<Float> - coords.elevation) / m_stretch;

        if (cos(theta) <= 0) {
            if (!m_extend)
                return Spectrum(0.0f);
            else
                theta = 0.5f * math::Pi<Float> - 1e-4f;
        }

        Float cos_gamma = cos(theta) * cos(math::Pi<Float> - m_sun.elevation)
            + sin(theta) * sin(math::Pi<Float> - m_sun.elevation)
            * cos(coords.azimuth - m_sun.azimuth);

        // Angle between the sun and the spherical coordinates in radians
        Float gamma = safe_acos(cos_gamma);

        Spectrum result;
        for (size_t i = 0; i < Spectrum::Size; i++) {
            if constexpr (Spectrum::Size == 3)
                result[i] = (Float) (arhosek_tristim_skymodel_radiance(
                    m_state[i], (double)theta, (double)gamma, i) / 106.856980); // (sum of Spectrum::CIE_Y)
            else {
                Float step_size = (MTS_WAVELENGTH_MIN - MTS_WAVELENGTH_MIN) / (Float) Spectrum::Size;
                Float wavelength0 = MTS_WAVELENGTH_MIN + step_size * i;
                Float wavelength1 = MTS_WAVELENGTH_MIN + step_size * (i+1);
                result[i] = (Float) arhosekskymodel_radiance(
                    m_state[i], (double)theta, (double)gamma, 0.5f * (wavelength0 + wavelength1));
            }
        }

        result = max(result, 0.f);

        if(m_extend)
           result *= smooth_step<Float>(0.f, 1.f, 2.f - 2.f * coords.elevation * math::InvPi<Float>);

        return result * m_scale;
    }

    std::pair<DirectionSample3f, Spectrum>
    sample_direction(const Interaction3f &it, const Point2f &sample, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::EndpointSampleDirection, active);

        Vector3f d = warp::square_to_uniform_sphere(sample);
        Float dist = 2.f * m_bsphere.radius;

        DirectionSample3f ds;
        ds.p      = it.p + d * dist;
        ds.n      = -d;
        ds.uv     = Point2f(0.f);
        ds.time   = it.time;
        ds.delta  = false;
        ds.object = this;
        ds.d      = d;
        ds.dist   = dist;
        ds.pdf    = pdf_direction(it, ds, active);

        SurfaceInteraction3f si = zero<SurfaceInteraction3f>();
        si.wavelengths = it.wavelengths;

        return std::make_pair(
            ds,
            unpolarized<Spectrum>(eval(si, active)) / ds.pdf
        );
    }

    Float pdf_direction(const Interaction3f &, const DirectionSample3f &ds,
                        Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::EndpointEvaluate, active);

        return warp::square_to_uniform_sphere_pdf(ds.d);
    }

    /// This emitter does not occupy any particular region of space, return an invalid bounding box
    ScalarBoundingBox3f bbox() const override {
        return ScalarBoundingBox3f();
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "SkyEmitter[" << std::endl
            //<< "  radiance = " << string::indent(m_radiance) << "," << std::endl
            << "  bsphere = " << m_bsphere << "," << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
protected:
    ScalarBoundingSphere3f m_bsphere;
    /// Environment map resolution in pixels
    int m_resolution;
    /// Constant scale factor applied to the model
    Float m_scale;
    /// Sky turbidity
    Float m_turbidity;
    /// Position of the sun in spherical coordinates
    SphericalCoordinates<Float> m_sun;
    /// Stretch factor to extend to the bottom hemisphere
    Float m_stretch;
    /// Extend to the bottom hemisphere (super-unrealistic mode)
    bool m_extend;
    /// Ground albedo
    ref<Texture> m_albedo;
    /// State vector for the sky model
    ArHosekSkyModelState *m_state[Spectrum::Size];
};

MTS_IMPLEMENT_CLASS_VARIANT(SkyEmitter, Emitter)
MTS_EXPORT_PLUGIN(SkyEmitter, "Skylight emitter")
NAMESPACE_END(mitsuba)
