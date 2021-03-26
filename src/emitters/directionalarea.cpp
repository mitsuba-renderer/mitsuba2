#include <mitsuba/core/properties.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/mesh.h>
#include <mitsuba/render/shape.h>
#include <mitsuba/render/texture.h>

NAMESPACE_BEGIN(mitsuba)

/**
 * Similar to an area light, but emitting only in the normal direction.
 */
template <typename Float, typename Spectrum>
class DirectionalArea final : public Emitter<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Emitter, set_shape, m_flags, m_shape, m_medium,
                    m_needs_sample_3)
    MTS_IMPORT_TYPES(Scene, Shape, Mesh, Texture)

    DirectionalArea(const Properties &props) : Base(props) {
        if (props.has_property("to_world"))
            Throw("Found a 'to_world' transformation -- this is not allowed. "
                  "The area light inherits this transformation from its parent "
                  "shape.");

        m_radiance       = props.texture("radiance", Texture::D65(1.f));
        m_needs_sample_3 = false;

        m_flags = EmitterFlags::Surface | EmitterFlags::DeltaDirection;
        if (m_radiance->is_spatially_varying())
            m_flags |= +EmitterFlags::SpatiallyVarying;
    }

    void set_shape(Shape *shape) override {
        Base::set_shape(shape);
        m_area = m_shape->surface_area();
    }

    std::pair<Ray3f, Spectrum> sample_ray(Float time, Float wavelength_sample,
                                          const Point2f &sample2,
                                          const Point2f & /*sample3*/,
                                          Mask active) const override {
        // 1. Sample spatial component
        PositionSample3f ps = m_shape->sample_position(time, sample2);

        // 2. Directional component is the normal vector.
        const Vector3f d = ps.n;

        // 3. Sample spectral component
        // TODO: need more fields?
        SurfaceInteraction3f si(ps, ek::zero<Wavelength>());
        auto [wavelength, wav_weight] = m_radiance->sample_spectrum(
            si, math::sample_shifted<Wavelength>(wavelength_sample), active);

        Ray3f ray(ps.p, d, time, wavelength);
        return { ray, m_area * wav_weight };
    }

    std::pair<DirectionSample3f, Spectrum>
    sample_direction(const Interaction3f &it, const Point2f & /*sample*/,
                     Mask active) const override {
        // using Index = uint32_array_t<Float>;

        Assert(m_shape, "Can't sample from an area emitter without an associated Shape.");
        /* Need to find the position, if any, that connects the reference point
         * to the emitter, along our surface's normal vector. */

        // This assumes the mesh is actually a plane with constant surface normal
        // TODO: More general case for arbitrary meshes?
        DirectionSample3f ds;

        const Mesh *mesh = dynamic_cast<Mesh *>(m_shape);
        size_t fc = mesh->face_count();
        if (mesh == nullptr || fc == 0) {
            return { ds, Spectrum(0.f) };
        }
        auto fi = mesh->face_indices(UInt32(0), active);

        Point3f p0 = mesh->vertex_position(fi[0], active),
                p1 = mesh->vertex_position(fi[1], active),
                p2 = mesh->vertex_position(fi[2], active);

        Vector3f dp0 = p1 - p0,
                dp1 = p2 - p0;
        Normal3f n = normalize(cross(dp0, dp1));

        Float dist;
        Ray3f ray(it.p, -n, it.time, it.wavelengths);
        for (size_t i = 0; i < fc; ++i) {
            auto prelim_it = mesh->ray_intersect_triangle(i, ray, active);
            ek::masked(dist, active) = prelim_it.t;
            active &= prelim_it.is_valid();
        }
        Point3f p                = ray(dist);
        ek::masked(ds.p, active) = p;
        ek::masked(ds.n, active) = n;
        ds.pdf                   = ek::select(active, 1.f, 0.f);
        ds.delta                 = true;
        ds.time                  = it.time;
        ds.d                     = -ds.n;
        ds.emitter               = this;

        SurfaceInteraction3f si(ds, it.wavelengths);  // TODO
        Spectrum weight = m_radiance->eval(si, active);
        return { ds, select(active, weight, Spectrum(0.f)) };
    }

    Float pdf_direction(const Interaction3f & /*it*/,
                        const DirectionSample3f & /*ds*/,
                        Mask /*active*/) const override {
        // TODO: this is potentially wrong
        return 1.f;
    }

    std::pair<PositionSample3f, Float>
    sample_position(Float time, const Point2f &sample,
                    Mask active) const override {
        Assert(m_shape, "Can't sample from an area emitter without an associated Shape.");
        PositionSample3f ps = m_shape->sample_position(time, sample, active);
        Float weight        = ek::select(ps.pdf > 0.f, ek::rcp(ps.pdf), 0.f);
        // ps.emitter          = this;
        return { ps, weight };
    }

    // Float pdf_position(const PositionSample3f &ps,
    //                    Mask active) const override {
    //     return m_shape->pdf_position(ps, active);
    // }

    std::pair<Wavelength, Spectrum>
    sample_wavelengths(const SurfaceInteraction3f &si, Float sample,
                       Mask active) const override {
        return m_radiance->sample_spectrum(
            si, math::sample_shifted<Wavelength>(sample), active);
    }

    Spectrum eval(const SurfaceInteraction3f & /*si*/, Mask /*active*/) const override {
        return 0.f;
    }

    // template <typename Ray, typename Value = typename Ray::Value,
    //           typename Spectrum = mitsuba::Spectrum<Value>>
    // MTS_INLINE Spectrum eval_environment(const Ray &, mask_t<Value>) const {
    //     return 0.f;
    // }

    ScalarBoundingBox3f bbox() const override { return m_shape->bbox(); }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "DirectionalArea[" << std::endl
            << "  radiance = " << string::indent(m_radiance) << "," << std::endl
            << "  surface_area = ";
        if (m_shape)
            oss << m_shape->surface_area();
        else
            oss << "  <no shape attached!>";
        oss << "," << std::endl;
        if (m_medium)
            oss << string::indent(m_medium->to_string());
        else
            oss << "  <no medium attached!>";
        oss << std::endl << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    ref<Texture> m_radiance;
    ScalarFloat m_area = 0.f;
};

MTS_IMPLEMENT_CLASS_VARIANT(DirectionalArea, Emitter)
MTS_EXPORT_PLUGIN(DirectionalArea, "Directional area emitter");
NAMESPACE_END(mitsuba)
