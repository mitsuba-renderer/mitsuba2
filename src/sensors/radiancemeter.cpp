#include <mitsuba/core/fwd.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/sensor.h>
#include <mitsuba/render/texture.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _sensor-radiancemeter:

Radiance meter (:monosp:`radiancemeter`)
----------------------------------------

.. pluginparameters::

 * - to_world
   - |transform|
   - Specifies an optional camera-to-world transformation.
     (Default: none (i.e. camera space = world space))
 * - origin
   - |point|
   - Location from which the sensor will be recording in world coordinates.
     Must be used with `origin`.
 * - direction
   - |vector|
   - Alternative (and exclusive) to `to_world`. Direction in which the
     sensor is pointing in world coordinates. Must be used with `origin`.
 * - srf
   - |spectrum|
   - If set, sensor response function used to sample wavelengths from. This 
     parameter is ignored if used with nonspectral variants.

This sensor plugin implements a simple radiance meter, which measures
the incident power per unit area per unit solid angle along a
certain ray. It can be thought of as the limit of a standard
perspective camera as its field of view tends to zero.
This sensor is used with films of 1 by 1 pixels.

Such a sensor is useful for conducting virtual experiments and
testing the renderer for correctness.

By default, the sensor is located at the origin and performs
a measurement in the positive Z direction :monosp:`(0,0,1)`. This can
be changed by providing a custom :monosp:`to_world` transformation, or a pair
of :monosp:`origin` and :monosp:`direction` values. If both types of
transformation are specified, the :monosp:`to_world` transformation has higher
priority.

*/

template <typename Float, typename Spectrum>
class RadianceMeter final : public Sensor<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Sensor, m_film, m_world_transform, m_needs_sample_2,
                    m_needs_sample_3)
    MTS_IMPORT_TYPES(Texture)

    RadianceMeter(const Properties &props) : Base(props), m_srf(nullptr) {
        if (props.has_property("srf")) {
            if constexpr(is_spectral_v<Spectrum>) {
                m_srf = props.texture<Texture>("srf");
            } else {
                Log(Warn, "Ignoring spectral response function "
                          "(not supported for non-spectral variants)");
            }
        }

        if (props.has_property("to_world")) {
            // if direction and origin are present but overridden by
            // to_world, they must still be marked as queried
            props.mark_queried("direction");
            props.mark_queried("origin");
        } else {
            if (props.has_property("direction") !=
                props.has_property("origin")) {
                Throw("If the sensor is specified through origin and direction "
                      "both values must be set!");
            }

            if (props.has_property("direction")) {
                ScalarPoint3f origin     = props.point3f("origin");
                ScalarVector3f direction = props.vector3f("direction");
                ScalarPoint3f target     = origin + direction;
                auto [up, unused]        = coordinate_system(direction);

                m_world_transform = new AnimatedTransform(
                    ScalarTransform4f::look_at(origin, target, up));
            }
        }

        if (m_film->size() != ScalarPoint2i(1, 1))
            Throw("This sensor only supports films of size 1x1 Pixels!");

        if (m_film->reconstruction_filter()->radius() >
            0.5f + math::RayEpsilon<Float>)
            Log(Warn, "This sensor should be used with a reconstruction filter "
                      "with a radius of 0.5 or lower (e.g. default box)");

        m_needs_sample_2 = false;
        m_needs_sample_3 = false;
    }

    std::pair<Ray3f, Spectrum> sample_ray(Float time, Float wavelength_sample,
                                          const Point2f & /*position_sample*/,
                                          const Point2f & /*aperture_sample*/,
                                          Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);
        
        // 1. Sample spectrum
        Wavelength wavelengths;
        Spectrum wav_weight;
        
        if (m_srf == nullptr) {
            std::tie(wavelengths, wav_weight) = 
                sample_wavelength<Float, Spectrum>(wavelength_sample);
        } else {
            std::tie(wavelengths, wav_weight) = 
                m_srf->sample_spectrum(
                    zero<SurfaceInteraction3f>(), 
                    math::sample_shifted<Wavelength>(wavelength_sample)
                );
        }

        // 2. Set ray origin and direction
        Ray3f ray;
        ray.time = time;
        ray.wavelengths = wavelengths;

        auto trafo = m_world_transform->eval(time, active);
        ray.o      = trafo.transform_affine(Point3f{ 0.f, 0.f, 0.f });
        ray.d      = trafo.transform_affine(Vector3f{ 0.f, 0.f, 1.f });

        ray.update();

        return { ray, wav_weight };
    }

    std::pair<RayDifferential3f, Spectrum>
    sample_ray_differential(Float time, Float wavelength_sample,
                            const Point2f & /*position_sample*/,
                            const Point2f & /*aperture_sample*/,
                            Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);
        // 1. Sample spectrum
        Wavelength wavelengths;
        Spectrum wav_weight;
        
        if (m_srf == nullptr) {
            std::tie(wavelengths, wav_weight) = 
                sample_wavelength<Float, Spectrum>(wavelength_sample);
        } else {
            std::tie(wavelengths, wav_weight) = 
                m_srf->sample_spectrum(
                    zero<SurfaceInteraction3f>(), 
                    math::sample_shifted<Wavelength>(wavelength_sample)
                );
        }

        // 2. Set ray origin and direction
        RayDifferential3f ray;
        ray.time = time;
        ray.wavelengths = wavelengths;

        auto trafo = m_world_transform->eval(time, active);
        ray.o      = trafo.transform_affine(Point3f{ 0.f, 0.f, 0.f });
        ray.d      = trafo.transform_affine(Vector3f{ 0.f, 0.f, 1.f });

        // 3. Set differentials; since the film size is always 1x1, we don't
        //    have differentials
        ray.has_differentials = false;

        ray.update();

        return { ray, wav_weight };
    }

    ScalarBoundingBox3f bbox() const override {
        // Return an invalid bounding box
        return ScalarBoundingBox3f();
    }

    std::string to_string() const override {
        using string::indent;
        
        std::ostringstream oss;
        oss << "RadianceMeter[" << std::endl
            << "  world_transform = " << m_world_transform << "," << std::endl
            << "  film = " << m_film << "," << std::endl
            << "  srf = " << indent(m_srf)  << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    ref<Texture> m_srf;
};

MTS_IMPLEMENT_CLASS_VARIANT(RadianceMeter, Sensor)
MTS_EXPORT_PLUGIN(RadianceMeter, "RadianceMeter");
NAMESPACE_END(mitsuba)
