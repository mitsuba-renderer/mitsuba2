#include <mitsuba/core/properties.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/shape.h>
#include <mitsuba/render/texture.h>

NAMESPACE_BEGIN(mitsuba)
/**!

.. _emitter-smootharea:

Smooth area light (:monosp:`smootharea`)
----------------------------------------

.. pluginparameters::

 * - radiance
   - |spectrum|
   - Specifies the emitted radiance in units of power per unit area per unit steradian.
     (Default: :ref:`d65 <emitter-d65>`)
 * - blur_size
   - |float|
   - Specifies the width of the smooth transition region from full emission to zero
     at the borders of the area light, in uv space. (Default: 0.1)

This plugin implements an area light with a smooth transition from full emission
to zero (black) at its borders. This type of light is usefull for differentiable
rendering since it typically avoids discontinuities around area lights. The transition
region is defined in uv space. This plugin should be used with a flat quadrilateral mesh
with texture coordinates that map to the unit square.

 */

template <typename Float, typename Spectrum>
class SmoothAreaLight final : public Emitter<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Emitter, m_flags, m_shape, m_medium)
    MTS_IMPORT_TYPES(Scene, Shape, Texture)

    SmoothAreaLight(const Properties &props) : Base(props) {
        if (props.has_property("to_world"))
            Throw("Found a 'to_world' transformation -- this is not allowed. "
                  "The area light inherits this transformation from its parent "
                  "shape.");

        m_radiance = props.texture<Texture>("radiance", Texture::D65(1.f));
        m_blur_size = props.float_("blur_size", 0.1f);

        // TODO: detect if underlying spectrum really is spatially varying
        m_flags = EmitterFlags::Surface | EmitterFlags::SpatiallyVarying;
    }

    void set_shape(Shape *shape) override {
        if (m_shape)
            Throw("An area emitter can be only be attached to a single shape.");

        Base::set_shape(shape);
        m_area_times_pi = m_shape->surface_area() * math::Pi<ScalarFloat>;
    }

    Float smooth_profile(Float x) const {
        Float res(0);
        res = select(x >= m_blur_size && x <= Float(1) - m_blur_size, Float(1), res);
        res = select(x < m_blur_size && x > Float(0), x / m_blur_size, res);
        res = select(x > Float(1) - m_blur_size && x < Float(1),
                     (1 - x) / m_blur_size, res);
        return res;
    }

    Spectrum eval(const SurfaceInteraction3f &si, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::EndpointEvaluate, active);

        return select(
            Frame3f::cos_theta(si.wi) > 0.f,
            unpolarized<Spectrum>(m_radiance->eval(si, active))
                * smooth_profile(si.uv.x()) * smooth_profile(si.uv.y()),
            0.f
        );
    }

    std::pair<Ray3f, Spectrum> sample_ray(Float time, Float wavelength_sample,
                                          const Point2f &sample2, const Point2f &sample3,
                                          Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);

        // 1. Sample spatial component
        PositionSample3f ps = m_shape->sample_position(time, sample2, active);

        // 2. Sample directional component
        Vector3f local = warp::square_to_cosine_hemisphere(sample3);

        // 3. Sample spectrum
        SurfaceInteraction3f si(ps, zero<Wavelength>(0.f));
        auto [wavelengths, spec_weight] = m_radiance->sample(
            si, math::sample_shifted<Wavelength>(wavelength_sample), active);

        spec_weight *= smooth_profile(ps.uv.x()) * smooth_profile(ps.uv.y());

        return std::make_pair(
            Ray3f(ps.p, Frame3f(ps.n).to_world(local), time, wavelengths),
            unpolarized<Spectrum>(spec_weight) * m_area_times_pi
        );
    }

    std::pair<DirectionSample3f, Spectrum>
    sample_direction(const Interaction3f &it, const Point2f &sample, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::EndpointSampleDirection, active);

        Assert(m_shape, "Can't sample from an area emitter without an associated Shape.");

        DirectionSample3f ds = m_shape->sample_direction(it, sample, active);
        active &= dot(ds.d, ds.n) < 0.f && neq(ds.pdf, 0.f);

        SurfaceInteraction3f si(ds, it.wavelengths);
        Spectrum spec = m_radiance->eval(si, active) / ds.pdf;
        spec *= smooth_profile(ds.uv.x()) * smooth_profile(ds.uv.y());

        ds.object = this;
        return { ds, unpolarized<Spectrum>(spec) & active };
    }

    Float pdf_direction(const Interaction3f &it, const DirectionSample3f &ds,
                        Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::EndpointEvaluate, active);

        return select(dot(ds.d, ds.n) < 0.f,
                      m_shape->pdf_direction(it, ds, active), 0.f);
    }

    ScalarBoundingBox3f bbox() const override { return m_shape->bbox(); }

    void traverse(TraversalCallback *callback) override {
        callback->put_object("radiance", m_radiance.get());
    }

    void parameters_changed(const std::vector<std::string> &keys) override {
        if (string::contains(keys, "parent"))
            m_area_times_pi = m_shape->surface_area() * math::Pi<ScalarFloat>;
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "SmoothAreaLight[" << std::endl
            << "  radiance = " << string::indent(m_radiance) << "," << std::endl
            << "  surface_area = ";
        if (m_shape) oss << m_shape->surface_area();
        else         oss << "  <no shape attached!>";
        oss << "," << std::endl;
        if (m_medium) oss << string::indent(m_medium->to_string());
        else         oss << "  <no medium attached!>";
        oss << std::endl << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    ref<Texture> m_radiance;
    ScalarFloat m_area_times_pi = 0.f;
    ScalarFloat m_blur_size;
};

MTS_IMPLEMENT_CLASS_VARIANT(SmoothAreaLight, Emitter)
MTS_EXPORT_PLUGIN(SmoothAreaLight, "Smooth Area emitter")
NAMESPACE_END(mitsuba)

