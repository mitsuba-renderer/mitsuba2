#include <mitsuba/core/fwd.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/string.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/core/util.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/shape.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _shape-rectangle:

Rectangle (:monosp:`rectangle`)
-------------------------------------------------

.. pluginparameters::

 * - flip_normals
   - |bool|
   - Is the rectangle inverted, i.e. should the normal vectors be flipped? (Default: |false|)
 * - to_world
   - |transform|
   - Specifies a linear object-to-world transformation. It is allowed to use non-uniform scaling,
     but no shear. (Default: none (i.e. object space = world space))

.. subfigstart::
.. subfigure:: ../../resources/data/docs/images/render/shape_rectangle.jpg
   :caption: Basic example
.. subfigure:: ../../resources/data/docs/images/render/shape_rectangle_parameterization.jpg
   :caption: A textured rectangle with the default parameterization
.. subfigend::
   :label: fig-rectangle

This shape plugin describes a simple rectangular shape primitive.
It is mainly provided as a convenience for those cases when creating and
loading an external mesh with two triangles is simply too tedious, e.g.
when an area light source or a simple ground plane are needed.
By default, the rectangle covers the XY-range :math:`[-1,1]\times[-1,1]`
and has a surface normal that points into the positive Z-direction.
To change the rectangle scale, rotation, or translation, use the
:monosp:`to_world` parameter.


The following XML snippet showcases a simple example of a textured rectangle:

.. code-block:: xml

    <shape type="rectangle">
        <bsdf type="diffuse">
            <texture name="reflectance" type="checkerboard">
                <transform name="to_uv">
                    <scale x="5" y="5" />
                </transform>
            </texture>
        </bsdf>
    </shape>

.. warning:: This plugin is currently not supported by the Embree and OptiX raytracing backend.

 */

template <typename Float, typename Spectrum>
class Rectangle final : public Shape<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Shape, bsdf, emitter, is_emitter)
    MTS_IMPORT_TYPES()

    using typename Base::ScalarSize;

    Rectangle(const Properties &props) : Base(props) {
        m_object_to_world = props.transform("to_world", ScalarTransform4f());
        if (props.bool_("flip_normals", false))
            m_object_to_world =
                m_object_to_world * ScalarTransform4f::scale(ScalarVector3f(1.f, 1.f, -1.f));

        m_world_to_object = m_object_to_world.inverse();

        ScalarVector3f dp_du = m_object_to_world * ScalarVector3f(2.f, 0.f, 0.f);
        ScalarVector3f dp_dv = m_object_to_world * ScalarVector3f(0.f, 2.f, 0.f);

        m_du = norm(dp_du);
        m_dv = norm(dp_dv);

        ScalarNormal3f normal = normalize(m_object_to_world * ScalarNormal3f(0.f, 0.f, 1.f));
        m_frame = ScalarFrame3f(dp_du / m_du, dp_dv / m_dv, normal);

        m_inv_surface_area = rcp(surface_area());
        if (abs(dot(m_frame.s, m_frame.t)) > math::RayEpsilon<ScalarFloat>)
            Throw("The `to_world` transformation contains shear, which is not"
                  " supported by the Rectangle shape.");

        if (is_emitter())
            emitter()->set_shape(this);
    }

    ScalarBoundingBox3f bbox() const override {
        ScalarBoundingBox3f bbox;
        bbox.expand(m_object_to_world.transform_affine(ScalarPoint3f(-1.f, -1.f, 0.f)));
        bbox.expand(m_object_to_world.transform_affine(ScalarPoint3f( 1.f, -1.f, 0.f)));
        bbox.expand(m_object_to_world.transform_affine(ScalarPoint3f( 1.f,  1.f, 0.f)));
        bbox.expand(m_object_to_world.transform_affine(ScalarPoint3f(-1.f,  1.f, 0.f)));
        return bbox;
    }

    ScalarFloat surface_area() const override {
        return m_du * m_dv;
    }

    // =============================================================
    //! @{ \name Sampling routines
    // =============================================================

    PositionSample3f sample_position(Float time, const Point2f &sample,
                                     Mask active) const override {
        MTS_MASK_ARGUMENT(active);

        PositionSample3f ps;
        ps.p = m_object_to_world.transform_affine(
            Point3f(sample.x() * 2.f - 1.f, sample.y() * 2.f - 1.f, 0.f));
        ps.n    = m_frame.n;
        ps.pdf  = m_inv_surface_area;
        ps.uv   = sample;
        ps.time = time;

        return ps;
    }

    Float pdf_position(const PositionSample3f & /*ps*/, Mask active) const override {
        MTS_MASK_ARGUMENT(active);
        return m_inv_surface_area;
    }

    DirectionSample3f sample_direction(const Interaction3f &it, const Point2f &sample,
                                       Mask active) const override {
        MTS_MASK_ARGUMENT(active);

        DirectionSample3f ds = sample_position(it.time, sample, active);
        ds.d = ds.p - it.p;

        Float dist_squared = squared_norm(ds.d);
        ds.dist  = sqrt(dist_squared);
        ds.d    /= ds.dist;

        Float dp = abs_dot(ds.d, ds.n);
        ds.pdf *= select(neq(dp, 0.f), dist_squared / dp, Float(0.f));

        return ds;
    }

    Float pdf_direction(const Interaction3f & /*it*/, const DirectionSample3f &ds,
                        Mask active) const override {
        MTS_MASK_ARGUMENT(active);

        Float pdf = pdf_position(ds, active),
              dp  = abs_dot(ds.d, ds.n);

        return pdf * select(neq(dp, 0.f), (ds.dist * ds.dist) / dp, 0.f);
    }

    //! @}
    // =============================================================

    // =============================================================
    //! @{ \name Ray tracing routines
    // =============================================================

    std::pair<Mask, Float> ray_intersect(const Ray3f &ray_, Float *cache,
                                         Mask active) const override {
        MTS_MASK_ARGUMENT(active);

        Ray3f ray     = m_world_to_object.transform_affine(ray_);
        Float t       = -ray.o.z() * ray.d_rcp.z();
        Point3f local = ray(t);

        // Is intersection within ray segment and rectangle?
        active = active && t >= ray.mint
                        && t <= ray.maxt
                        && abs(local.x()) <= 1.f
                        && abs(local.y()) <= 1.f;

        t = select(active, t, Float(math::Infinity<Float>));

        if (cache) {
            masked(cache[0], active) = local.x();
            masked(cache[1], active) = local.y();
        }

        return { active, t };
    }

    Mask ray_test(const Ray3f &ray_, Mask active) const override {
        MTS_MASK_ARGUMENT(active);

        Ray3f ray     = m_world_to_object.transform_affine(ray_);
        Float t       = -ray.o.z() * ray.d_rcp.z();
        Point3f local = ray(t);

        // Is intersection within ray segment and rectangle?
        return active && t >= ray.mint
                      && t <= ray.maxt
                      && abs(local.x()) <= 1.f
                      && abs(local.y()) <= 1.f;
    }

    void fill_surface_interaction(const Ray3f &ray_, const Float *cache,
                                  SurfaceInteraction3f &si_out, Mask active) const override {
        MTS_MASK_ARGUMENT(active);

#if !defined(MTS_ENABLE_EMBREE)
        Float local_x = cache[0];
        Float local_y = cache[1];
#else
        Ray3f ray     = m_world_to_object.transform_affine(ray_);
        Float t       = -ray.o.z() * ray.d_rcp.z();
        Point3f local = ray(t);
        Float local_x = local.x();
        Float local_y = local.y();
#endif

        SurfaceInteraction3f si(si_out);

        si.n          = m_frame.n;
        si.sh_frame.n = m_frame.n;
        si.dp_du      = m_du * m_frame.s;
        si.dp_dv      = m_dv * m_frame.t;
        si.p          = ray_(si.t);
        si.time       = ray_.time;
        si.uv         = Point2f(fmadd(local_x, .5f, .5f),
                                fmadd(local_y, .5f, .5f));

        si_out[active] = si;
    }

    std::pair<Vector3f, Vector3f> normal_derivative(const SurfaceInteraction3f & /*si*/,
                                                    bool /*shading_frame*/,
                                                    Mask /*active*/) const override {
        return { Vector3f(0.f), Vector3f(0.f) };
    }

    ScalarSize primitive_count() const override { return 1; }

    ScalarSize effective_primitive_count() const override { return 1; }

    void traverse(TraversalCallback *callback) override {
        callback->put_parameter("frame", m_frame);
        callback->put_parameter("du", m_du);
        callback->put_parameter("dv", m_dv);
        Base::traverse(callback);
    }

    void parameters_changed() override {
        Base::parameters_changed();
        m_object_to_world = ScalarTransform4f::to_frame(m_frame) *
                            ScalarTransform4f::scale(ScalarVector3f(0.5f * m_du, 0.5f * m_dv, 1.f));
        m_world_to_object = m_object_to_world.inverse();
        m_inv_surface_area = 1.f / surface_area();
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "Rectangle[" << std::endl
            << "  object_to_world = " << string::indent(m_object_to_world) << "," << std::endl
            << "  frame = " << string::indent(m_frame) << "," << std::endl
            << "  inv_surface_area = " << m_inv_surface_area << "," << std::endl
            << "  bsdf = " << string::indent(bsdf()->to_string()) << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    ScalarTransform4f m_object_to_world;
    ScalarTransform4f m_world_to_object;
    ScalarFrame3f m_frame;
    ScalarFloat m_du, m_dv;
    ScalarFloat m_inv_surface_area;
};

MTS_IMPLEMENT_CLASS_VARIANT(Rectangle, Shape)
MTS_EXPORT_PLUGIN(Rectangle, "Rectangle intersection primitive");
NAMESPACE_END(mitsuba)
