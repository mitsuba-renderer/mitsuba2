#include <mitsuba/core/bbox.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/texture.h>
#include <mitsuba/render/sensor.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _emitter-projector:

Projector light source (:monosp:`projector`)
------------------------------------

.. pluginparameters::

 * - intensity
   - |spectrum|
   - Specifies the maximum radiant intensity at the center in units of power per unit steradian. (Default: 1).
     This cannot be spatially varying (e.g. have bitmap as type).
 * - aspect_ratio
   - |float|
   - The aspect ratio (i.e. width/height) of the projection. (Default: 1.778).
 * - focal_length
   - |string|
   - Denotes the projector’s focal length specified using :monosp:`35mm` film equivalent units. See the 
     main description for sensors for further details. (Default: :monosp:`50mm`).
 * - fov
   - |float|
   - An alternative to :monosp:`focal_length`: denotes the projector’s field of view in degrees—must be
     between 0 and 180, excluding the extremes.
 * - fov_axis
   - |string|
   - When the parameter :monosp:`fov` is given (and only then), this parameter further specifies
     the image axis, to which it applies.
     1. :monosp:`x`: :monosp:`fov` maps to the :monosp:`x`-axis in screen space.
     2. :monosp:`y`: :monosp:`fov` maps to the :monosp:`y`-axis in screen space.
     3. :monosp:`diagonal`: :monosp:`fov` maps to the screen diagonal.
     4. :monosp:`smaller`: :monosp:`fov` maps to the smaller dimension
        (e.g. :monosp:`x` when :monosp:`width` < :monosp:`height`)
     5. :monosp:`larger`: :monosp:`fov` maps to the larger dimension
        (e.g. :monosp:`y` when :monosp:`width` < :monosp:`height`)
     The default is :monosp:`x`.
 * - texture
   - |texture|
   - An optional texture to be projected. If none is specified, a white light is projected. This must be
     spatially varying (e.g. have bitmap as type).
 * - to_world
   - |transform|
   - Specifies an optional emitter-to-world transformation.  (Default: none, i.e. emitter space = world space)

This emitter plugin implements a simple point light source, which
uniformly radiates illumination into all directions.

 */

template <typename Float, typename Spectrum>
class ProjectorLight final : public Emitter<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Emitter, m_flags, m_medium, m_needs_sample_3, m_world_transform)
    MTS_IMPORT_TYPES(Scene, Shape, Texture)

    ProjectorLight(const Properties &props) : Base(props) {
        m_flags = +EmitterFlags::DeltaPosition;
        m_intensity = props.texture<Texture>("intensity", Texture::D65(1.f));
        m_texture = props.texture<Texture>("texture", Texture::D65(1.f));

        if (m_intensity->is_spatially_varying())
            Throw("The parameter 'intensity' cannot be spatially varying (e.g. bitmap type)!");

        if (props.has_property("texture")) {
            if (!m_texture->is_spatially_varying())
                Throw("The parameter 'texture' must be spatially varying (e.g. bitmap type)!");
            m_flags |= +EmitterFlags::SpatiallyVarying;
        }

        m_aspect = props.float_("aspect_ratio", 1.778f);
        m_x_fov = parse_fov(props, m_aspect);
        const ScalarFloat m_x_fov_rad = deg_to_rad(m_x_fov);
        const ScalarFloat m_width_cutoff_angle = m_x_fov_rad / 2.0f;
        const ScalarFloat m_height_cutoff_angle = atan(tan(m_width_cutoff_angle) / m_aspect);
        m_cos_height_cutoff_angle = cos(m_height_cutoff_angle);
        m_cos_width_cutoff_angle = cos(m_width_cutoff_angle);
        m_height_uv_factor = tan(m_height_cutoff_angle);
        m_width_uv_factor = tan(m_width_cutoff_angle);
        m_needs_sample_3 = false;
    }

    inline Spectrum get_spectrum(const Vector3f &d, Wavelength wavelengths, Mask active) const {
        SurfaceInteraction3f si = zero<SurfaceInteraction3f>();
        si.wavelengths = wavelengths;
        Spectrum result = m_intensity->eval(si, active);

        Vector3f local_dir = normalize(d);
        Vector3f xz_projection = Vector3f(local_dir.x(), 0.0f, local_dir.z());
        Vector3f yz_projection = Vector3f(0.0f, local_dir.y(), local_dir.z());

        xz_projection /= norm(xz_projection);
        yz_projection /= norm(yz_projection);

        const Float cos_xz_theta = xz_projection.z();
        const Float cos_yz_theta = yz_projection.z();

        if (m_texture->is_spatially_varying()) {
            si.uv = Point2f(0.5f + 0.5f * local_dir.x() / (local_dir.z() * m_width_uv_factor),
                            0.5f + 0.5f * local_dir.y() / (local_dir.z() * m_height_uv_factor));
            si.wavelengths = wavelengths;
            result *= m_texture->eval(si, active);
        }

        result = select(cos_xz_theta <= m_cos_width_cutoff_angle, Spectrum(0.0f), result);
        return select(cos_yz_theta <= m_cos_height_cutoff_angle, Spectrum(0.0f), result);
    }

    std::pair<Ray3f, Spectrum> sample_ray(Float /*time*/, Float /*wavelength_sample*/,
                                          const Point2f & /*spatial_sample*/,
                                          const Point2f & /*dir_sample*/,
                                          Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);

        NotImplementedError("sample_ray");
    }

    std::pair<DirectionSample3f, Spectrum> sample_direction(const Interaction3f &it,
                                                            const Point2f & /*sample*/,
                                                            Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::EndpointSampleDirection, active);

        auto trafo = m_world_transform->eval(it.time, active);

        DirectionSample3f ds;
        ds.p            = trafo.translation();
        ds.n            = 0.f;
        ds.uv           = 0.f;
        ds.time         = it.time;
        ds.pdf          = 1.f;
        ds.delta        = true;
        ds.object       = this;
        ds.d            = ds.p - it.p;
        ds.dist         = norm(ds.d);
        Float inv_dist  = rcp(ds.dist);
        ds.d            *= inv_dist;
        Vector3f local_d = trafo.inverse() * -ds.d;
        Spectrum projector_spec = get_spectrum(local_d, it.wavelengths, active);

        return { ds, unpolarized<Spectrum>(projector_spec * (inv_dist * inv_dist)) };
    }

    Float pdf_direction(const Interaction3f &, const DirectionSample3f &, Mask) const override {
        return 0.f;
    }

    Spectrum eval(const SurfaceInteraction3f &, Mask) const override { return 0.f; }

    ScalarBoundingBox3f bbox() const override {
        return m_world_transform->translation_bounds();
    }

    void traverse(TraversalCallback *callback) override {
        callback->put_object("intensity", m_intensity.get());
        callback->put_object("texture", m_texture.get());
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "ProjectorLight[" << std::endl
            << "  world_transform = " << string::indent(m_world_transform->to_string()) << "," << std::endl
            << "  intensity = " << m_intensity << "," << std::endl
            << "  aspect = " << m_aspect << "," << std::endl
            << "  x_fov = " << m_x_fov << "," << std::endl
            << "  texture = " << (m_texture ? string::indent(m_texture->to_string()) : "")
            << "  medium = " << (m_medium ? string::indent(m_medium->to_string()) : "")
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    ref<Texture> m_intensity;
	ref<Texture> m_texture;
	ScalarFloat m_aspect, m_x_fov;
    ScalarFloat m_height_uv_factor, m_width_uv_factor;
    ScalarFloat m_cos_height_cutoff_angle, m_cos_width_cutoff_angle;
};


MTS_IMPLEMENT_CLASS_VARIANT(ProjectorLight, Emitter)
MTS_EXPORT_PLUGIN(ProjectorLight, "Projector emitter")
NAMESPACE_END(mitsuba)
