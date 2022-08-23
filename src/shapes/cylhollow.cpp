
#include <mitsuba/core/fwd.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/core/util.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/shape.h>

#if defined(MTS_ENABLE_OPTIX)
    #include "optix/cylhollow.cuh"
#endif

#include <unistd.h>

NAMESPACE_BEGIN(mitsuba)

    static int dbg = 0;
    static int dbg2 = 0;

/**!

.. _shape-sphere:

CylHollow (:monosp:`sphere`)
-------------------------------------------------

.. pluginparameters::

 * - center
   - |point|
   - Center of the sphere (Default: (0, 0, 0))
 * - radius
   - |float|
   - Radius of the sphere (Default: 1)
 * - flip_normals
   - |bool|
   - Is the sphere inverted, i.e. should the normal vectors be flipped? (Default:|false|, i.e.
     the normals point outside)
 * - to_world
   - |transform|
   -  Specifies an optional linear object-to-world transformation.
      Note that non-uniform scales and shears are not permitted!
      (Default: none, i.e. object space = world space)

.. subfigstart::
.. subfigure:: ../../resources/data/docs/images/render/shape_sphere_basic.jpg
   :caption: Basic example
.. subfigure:: ../../resources/data/docs/images/render/shape_sphere_parameterization.jpg
   :caption: A textured sphere with the default parameterization
.. subfigend::
   :label: fig-sphere

This shape plugin describes a simple sphere intersection primitive. It should
always be preferred over sphere approximations modeled using triangles.

A sphere can either be configured using a linear :monosp:`to_world` transformation or the :monosp:`center` and :monosp:`radius` parameters (or both).
The two declarations below are equivalent.

.. code-block:: xml

    <shape type="sphere">
        <transform name="to_world">
            <scale value="2"/>
            <translate x="1" y="0" z="0"/>
        </transform>
        <bsdf type="diffuse"/>
    </shape>

    <shape type="sphere">
        <point name="center" x="1" y="0" z="0"/>
        <float name="radius" value="2"/>
        <bsdf type="diffuse"/>
    </shape>

When a :ref:`sphere <shape-sphere>` shape is turned into an :ref:`area <emitter-area>`
light source, Mitsuba 2 switches to an efficient
`sampling strategy <https://www.akalin.com/sampling-visible-sphere>`_ by Fred Akalin that
has particularly low variance.
This makes it a good default choice for lighting new scenes.

.. subfigstart::
.. subfigure:: ../../resources/data/docs/images/render/shape_sphere_light_mesh.jpg
   :caption: Spherical area light modeled using triangles
.. subfigure:: ../../resources/data/docs/images/render/shape_sphere_light_analytic.jpg
   :caption: Spherical area light modeled using the :ref:`sphere <shape-sphere>` plugin
.. subfigend::
   :label: fig-sphere-light
 */

template <typename Float, typename Spectrum>
class CylHollow final : public Shape<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Shape, m_to_world, m_to_object, set_children,
                    get_children_string, parameters_grad_enabled)
    MTS_IMPORT_TYPES()

    using Double = std::conditional_t<is_cuda_array_v<Float>, Float, Float64>;
    using Double3 = Vector<Double, 3>;

    using typename Base::ScalarSize;

    CylHollow(const Properties &props) : Base(props) {

        (void) dbg; (void) dbg2;

        /// Are the sphere normals pointing inwards? default: no
        m_flip_normals = props.bool_("flip_normals", false);

        /// Thickness of shell
        m_thickness = props.float_("thickness", 1.0f);

        /// Length of the tube
        m_h = props.float_("length", 1.0f);

        // Update the to_world transform if radius and center are also provided
        m_to_world = m_to_world * ScalarTransform4f::translate(props.point3f("center", 0.f));
        m_to_world = m_to_world * ScalarTransform4f::scale(props.float_("radius", 1.f));

        update();
        set_children();
    }

    void update() {
        // Extract center and radius from to_world matrix (25 iterations for numerical accuracy)
        auto [S, Q, T] = transform_decompose(m_to_world.matrix, 25);

        if (abs(S[0][1]) > 1e-6f || abs(S[0][2]) > 1e-6f || abs(S[1][0]) > 1e-6f ||
            abs(S[1][2]) > 1e-6f || abs(S[2][0]) > 1e-6f || abs(S[2][1]) > 1e-6f)
            Log(Warn, "'to_world' transform shouldn't contain any shearing!");

        if (!(abs(S[0][0] - S[1][1]) < 1e-6f && abs(S[0][0] - S[2][2]) < 1e-6f))
            Log(Warn, "'to_world' transform shouldn't contain non-uniform scaling!");

        m_center = T;
        m_radius0 = S[0][0];

        // Just try to make the outer radius depend
        // on world transform for now
        m_radius1 = m_radius0 + m_thickness;

        if (m_radius0 <= 0.f) {
            Log(Warn, "Radius < 0 is not supported. Rendering will fail!");
            m_radius0 = std::abs(m_radius0);
            m_flip_normals = !m_flip_normals;
        }

        m_h_low = m_center[2];
        m_h_hi  = m_center[2] + m_h;

        // Reconstruct the to_world transform with uniform scaling and no shear
        m_to_world = transform_compose(ScalarMatrix3f(m_radius0), Q, T);
        m_to_object = m_to_world.inverse();

        m_inv_surface_area = rcp(surface_area());
    }

    ScalarBoundingBox3f bbox() const override {

        ScalarBoundingBox3f bbox;

        bbox.min = m_center;
        bbox.max = m_center + ScalarPoint3f(m_radius1, m_radius1, m_h);


                bbox.min = m_center - 1000;
                bbox.max = m_center + 1000;


        return bbox;
    }

    ScalarFloat surface_area() const override {

        ScalarFloat ret;

        /*
         * Inner, outer surfaces + top/bottom
         * */
        ret = 2.0f * math::Pi<ScalarFloat> * m_radius0 * m_h;
        ret += 2.0f * math::Pi<ScalarFloat> * m_radius1 * m_h;
        ret += 2.0f * enoki::pow( math::Pi<ScalarFloat>, 2.0f );

 //       return ret;
                return 1000 * 4.f * math::Pi<ScalarFloat> * m_radius1 * m_radius1;
    }

    // =============================================================
    //! @{ \name Sampling routines
    // =============================================================

    PositionSample3f sample_position(Float time, const Point2f &sample,
                                     Mask active) const override {
        MTS_MASK_ARGUMENT(active);

        Log(Warn, "sample_position is not implemented correctly");

        Point3f local = warp::square_to_uniform_sphere(sample);

        PositionSample3f ps;
        ps.p = fmadd(local, m_radius0, m_center);
        ps.n = local;

        if (m_flip_normals)
            ps.n = -ps.n;

        ps.time = time;
        ps.delta = m_radius0 == 0.f;
        ps.pdf = m_inv_surface_area;

        return ps;
    }

    Float pdf_position(const PositionSample3f & /*ps*/, Mask active) const override {
        MTS_MASK_ARGUMENT(active);
        Log(Warn, "pdf_position is not implemented correctly");
        return m_inv_surface_area;
    }

    DirectionSample3f sample_direction(const Interaction3f &it, const Point2f &sample,
                                       Mask active) const override {
        MTS_MASK_ARGUMENT(active);

        Log(Warn, "sample_direction is not implemented correctly");

        DirectionSample3f result = zero<DirectionSample3f>();

        Vector3f dc_v = m_center - it.p;
        Float dc_2 = squared_norm(dc_v);

        Float radius_adj = m_radius0 * (m_flip_normals ? (1.f + math::RayEpsilon<Float>) :
                                                        (1.f - math::RayEpsilon<Float>));
        Mask outside_mask = active && dc_2 > sqr(radius_adj);
        if (likely(any(outside_mask))) {
            Float inv_dc            = rsqrt(dc_2),
                  sin_theta_max     = m_radius0 * inv_dc,
                  sin_theta_max_2   = sqr(sin_theta_max),
                  inv_sin_theta_max = rcp(sin_theta_max),
                  cos_theta_max     = safe_sqrt(1.f - sin_theta_max_2);

            /* Fall back to a Taylor series expansion for small angles, where
               the standard approach suffers from severe cancellation errors */
            Float sin_theta_2 = select(sin_theta_max_2 > 0.00068523f, /* sin^2(1.5 deg) */
                                       1.f - sqr(fmadd(cos_theta_max - 1.f, sample.x(), 1.f)),
                                       sin_theta_max_2 * sample.x()),
                  cos_theta = safe_sqrt(1.f - sin_theta_2);

            // Based on https://www.akalin.com/sampling-visible-sphere
            Float cos_alpha = sin_theta_2 * inv_sin_theta_max +
                              cos_theta * safe_sqrt(fnmadd(sin_theta_2, sqr(inv_sin_theta_max), 1.f)),
                  sin_alpha = safe_sqrt(fnmadd(cos_alpha, cos_alpha, 1.f));

            auto [sin_phi, cos_phi] = sincos(sample.y() * (2.f * math::Pi<Float>));

            Vector3f d = Frame3f(dc_v * -inv_dc).to_world(Vector3f(
                cos_phi * sin_alpha,
                sin_phi * sin_alpha,
                cos_alpha));

            DirectionSample3f ds = zero<DirectionSample3f>();
            ds.p        = fmadd(d, m_radius0, m_center);
            ds.n        = d;
            ds.d        = ds.p - it.p;

            Float dist2 = squared_norm(ds.d);
            ds.dist     = sqrt(dist2);
            ds.d        = ds.d / ds.dist;
            ds.pdf      = warp::square_to_uniform_cone_pdf(zero<Vector3f>(), cos_theta_max);
            masked(ds.pdf, ds.dist == 0.f) = 0.f;

            result[outside_mask] = ds;
        }

        Mask inside_mask = andnot(active, outside_mask);
        if (unlikely(any(inside_mask))) {
            Vector3f d = warp::square_to_uniform_sphere(sample);
            DirectionSample3f ds = zero<DirectionSample3f>();
            ds.p        = fmadd(d, m_radius0, m_center);
            ds.n        = d;
            ds.d        = ds.p - it.p;

            Float dist2 = squared_norm(ds.d);
            ds.dist     = sqrt(dist2);
            ds.d        = ds.d / ds.dist;
            ds.pdf      = m_inv_surface_area * dist2 / abs_dot(ds.d, ds.n);

            result[inside_mask] = ds;
        }

        result.time = it.time;
        result.delta = m_radius0 == 0.f;

        if (m_flip_normals)
            result.n = -result.n;

        return result;
    }

    Float pdf_direction(const Interaction3f &it, const DirectionSample3f &ds,
                        Mask active) const override {
        MTS_MASK_ARGUMENT(active);
        Log(Warn, "pdf_direction is not implemented correctly");

        // Sine of the angle of the cone containing the sphere as seen from 'it.p'.
        Float sin_alpha = m_radius0 * rcp(norm(m_center - it.p)),
              cos_alpha = enoki::safe_sqrt(1.f - sin_alpha * sin_alpha);

        return select(sin_alpha < math::OneMinusEpsilon<Float>,
            // Reference point lies outside the sphere
            warp::square_to_uniform_cone_pdf(zero<Vector3f>(), cos_alpha),
            m_inv_surface_area * sqr(ds.dist) / abs_dot(ds.d, ds.n)
        );
    }

    //! @}
    // =============================================================

    // =============================================================
    //! @{ \name Ray tracing routines
    // =============================================================

    Double intersect( Double mint, Double maxt,
                    ScalarFloat radius,
                    const Ray3f &ray) const{

        Double res;
        Mask valid0, valid1;

        Double x0 = (scalar_t<Double>) m_center[0];
        Double y0 = (scalar_t<Double>) m_center[1];

        Double ox = Double( ray.o[0] ),
               oy = Double( ray.o[1] );

        Double dx = Double( ray.d[0] ),
               dy = Double( ray.d[1] );

        // How on earth do we avoid this..
        Double two = Double( scalar_t<Double>(2.0f) );

        Double A = enoki::pow( dx, two ) + enoki::pow( dy, two );
        Double B = scalar_t<Double>(2.0f) * ox * dx - scalar_t<Double>(2.0f) * x0 * dx +
                   scalar_t<Double>(2.0f) * oy * dy - scalar_t<Double>(2.0f) * y0 * dy;
        Double C = enoki::pow( ox, two ) - two * ox * x0 + enoki::pow( x0, two ) +
                   enoki::pow( oy, two ) - two * oy * y0 + enoki::pow( y0, two ) -
                   enoki::pow( scalar_t<Double>(radius), 2.0f );

        auto [solution_found, near_t, far_t] = math::solve_quadratic(A, B, C);

        if( any( near_t > far_t ) ){
            Log(Warn, "Assertion failure! near_t <= far_t");
        }

        Float z_lim_low = Float(m_h_low);
        Float z_lim_hi  = Float(m_h_hi);

        valid0 = solution_found && ((near_t > mint) && (near_t < maxt));
        valid0 = valid0 && (ray(near_t)[2] >= z_lim_low) && (ray(near_t)[2] < z_lim_hi);

        valid1 = solution_found && ((far_t > mint) && (far_t < maxt));
        valid1 = valid1 && (ray(far_t)[2] >= z_lim_low) && (ray(far_t)[2] < z_lim_hi);

        res = select( valid0, near_t,
                     select( valid1, far_t, math::Infinity<Double> ) );

        return res;
    }

    Double zintersect( Double mint, Double maxt,
                       ScalarPoint3f center,
                       const Ray3f &ray) const{

        Double t;
        Mask valid;
        Mask bounds;

        Double oz = Double( ray.o[2] );
        Double dz = Double( ray.d[2] );
        Double a = (scalar_t<Double>) center[2] ;

        valid = (dz != 0);
        t = select(valid, (a - oz) / dz, math::Infinity<Double> );

#if 0
        Double3 pt;
        pt = ray(t);
        std::cerr << "point0," << pt[0] << "," << pt[1] << "," << pt[2] << "\n";
        //pt = ray(far_t1);
        //std::cerr << "point3," << pt[0] << "," << pt[1] << "," << pt[2] << "\n";
        //std::cerr << "vec3," << ray.o[0] << "," << ray.o[1] << "," << ray.o[2] << "," << ray.d[0] << "," << ray.d[1] << "," << ray.d[2]  << "\n";
        usleep(1000);
#endif

        valid = (t >= mint) && (t <= maxt);
        t = select(valid, t, math::Infinity<Double>);

        Double3 diff = Double3(center) - Double3(ray( t ));

        Double h = sqrt( dot( diff, diff ) );
        //std::cout << h << " -- " << valid << "\n";

        bounds = (h >= (scalar_t<Double>)m_radius0) && (h <= (scalar_t<Double>)m_radius1);

        Double res = select( bounds, t, math::Infinity<Double> );

#if 0
        if( any( bounds ) ) {
            pt = ray(t);
            std::cerr << "point1," << pt[0] << "," << pt[1] << "," << pt[2] << "\n";
            //pt = ray(far_t1);
            //std::cerr << "point3," << pt[0] << "," << pt[1] << "," << pt[2] << "\n";
            //std::cerr << "vec3," << ray.o[0] << "," << ray.o[1] << "," << ray.o[2] << "," << ray.d[0] << "," << ray.d[1] << "," << ray.d[2]  << "\n";
            usleep(1000);
        }
#endif


        return res;
    }


    PreliminaryIntersection3f ray_intersect_preliminary(const Ray3f &ray,
                                                        Mask active) const override {
        MTS_MASK_ARGUMENT(active);

        Double mint = Double(ray.mint);
        Double maxt = Double(ray.maxt);

        Double t0, t1;
        Double z0, z1;

        t0 = intersect( mint, maxt, m_radius0, ray );
        t1 = intersect( mint, maxt, m_radius1, ray );
        z0 = zintersect( mint, maxt, m_center + ScalarPoint3f(0,0,0), ray);
        z1 = zintersect( mint, maxt, m_center + ScalarPoint3f(0,0,m_h), ray);

        Double t;

        t = select( t0 < t1, t0, t1 );
#if 1
        t = select( t < z0, t, z0 );
        t = select( t < z1, t, z1 );
#endif

        Double3 pt;
#if 0
        if( ! any( t == math::Infinity<Double>) ){
            pt = ray(t);
            std::cerr << "point0," << pt[0] << "," << pt[1] << "," << pt[2] << "\n";
            //pt = ray(far_t1);
            //std::cerr << "point3," << pt[0] << "," << pt[1] << "," << pt[2] << "\n";
            //std::cerr << "vec3," << ray.o[0] << "," << ray.o[1] << "," << ray.o[2] << "," << ray.d[0] << "," << ray.d[1] << "," << ray.d[2]  << "\n";
            usleep(1000);
        }
#endif
#if 0
        if( any(sol0) ){
            pt = ray(near_t0);
            std::cerr << "point0," << pt[0] << "," << pt[1] << "," << pt[2] << "\n";
            pt = ray(far_t0);
            std::cerr << "point1," << pt[0] << "," << pt[1] << "," << pt[2] << "\n";

            //pt = ray(far_t1);
            //std::cerr << "point3," << pt[0] << "," << pt[1] << "," << pt[2] << "\n";
            //std::cerr << "vec3," << ray.o[0] << "," << ray.o[1] << "," << ray.o[2] << "," << ray.d[0] << "," << ray.d[1] << "," << ray.d[2]  << "\n";
            usleep(1000);
        }
#endif
#if 0
        if( any( sol0 ) ){
            pt = ray(near_t0);
            std::cerr << "point0," << pt[0] << "," << pt[1] << "," << pt[2] << "\n";
            pt = ray(far_t0);
            std::cerr << "point1," << pt[0] << "," << pt[1] << "," << pt[2] << "\n";
            //std::cerr << "vec3," << ray.o[0] << "," << ray.o[1] << "," << ray.o[2] << "," << ray.d[0] << "," << ray.d[1] << "," << ray.d[2]  << "\n";
            usleep(1000);
        }

        if( any( sol1 ) ){
            pt = ray(near_t1);
            std::cerr << "point2," << pt[0] << "," << pt[1] << "," << pt[2] << "\n";
            pt = ray(far_t1);
            std::cerr << "point3," << pt[0] << "," << pt[1] << "," << pt[2] << "\n";
            //std::cerr << "vec3," << ray.o[0] << "," << ray.o[1] << "," << ray.o[2] << "," << ray.d[0] << "," << ray.d[1] << "," << ray.d[2]  << "\n";
            usleep(1000);
        }
#endif




#if 0
        // CylHollow doesn't intersect with the segment on the ray
        Mask out_bounds = !(near_t <= maxt && far_t >= mint); // NaN-aware conditionals

        // CylHollow fully contains the segment of the ray
        Mask in_bounds = near_t < mint && far_t > maxt;

        active &= solution_found && !out_bounds && !in_bounds;
#endif

        PreliminaryIntersection3f pi = zero<PreliminaryIntersection3f>();
        pi.t = t;
#if 0
        pi.t = select(active,
                      select(near_t < mint, Float(far_t), Float(near_t)),
                      math::Infinity<Float>);
#else
        //pi.t = math::Infinity<Float>;
#endif

        pi.shape = this;

        return pi;
    }

    Mask ray_test(const Ray3f &ray, Mask active) const override {
        MTS_MASK_ARGUMENT(active);
#if 1
        (void) ray;
        return Mask(0);
#else
        using Double = std::conditional_t<is_cuda_array_v<Float>, Float, Float64>;
        using Double3 = Vector<Double, 3>;

        Double mint = Double(ray.mint);
        Double maxt = Double(ray.maxt);

        Double3 o = Double3(ray.o) - Double3(m_center);
        Double3 d(ray.d);

        Double A = squared_norm(d);
        Double B = scalar_t<Double>(2.f) * dot(o, d);
        Double C = squared_norm(o) - sqr((scalar_t<Double>) m_radius0);

        auto [solution_found, near_t, far_t] = math::solve_quadratic(A, B, C);

        // CylHollow doesn't intersect with the segment on the ray
        Mask out_bounds = !(near_t <= maxt && far_t >= mint); // NaN-aware conditionals

        // CylHollow fully contains the segment of the ray
        Mask in_bounds  = near_t < mint && far_t > maxt;

        return solution_found && !out_bounds && !in_bounds && active;
#endif
    }

    SurfaceInteraction3f compute_surface_interaction(const Ray3f &ray,
                                                     PreliminaryIntersection3f pi,
                                                     HitComputeFlags flags,
                                                     Mask active) const override {
        MTS_MASK_ARGUMENT(active);

        bool differentiable = false;
        if constexpr (is_diff_array_v<Float>)
            differentiable = requires_gradient(ray.o) ||
                             requires_gradient(ray.d) ||
                             parameters_grad_enabled();

        // Recompute ray intersection to get differentiable prim_uv and t
        if (differentiable && !has_flag(flags, HitComputeFlags::NonDifferentiable))
            pi = ray_intersect_preliminary(ray, active);

        active &= pi.is_valid();

        SurfaceInteraction3f si = zero<SurfaceInteraction3f>();
        si.t = select(active, pi.t, math::Infinity<Float>);

        Mask is_zplane;

        Double zcomp = Double(ray( pi.t )[2]);

        is_zplane = (zcomp == (scalar_t<Double>) m_h_low) || (zcomp == (scalar_t<Double>) m_h_hi);

        Double3 d = Double3(1,1,0) * (ray(pi.t) - m_center);

        Double h = sqrt( dot( d, d ) );

        si.sh_frame.n = select( is_zplane,
                                select( zcomp == (scalar_t<Double>) m_h_low,
                                        normalize( Double3( 0,0,-1 ) ),
                                        normalize( Double3( 0,0, 1 ) )),
                                select( h > (scalar_t<Double>) (m_radius0 + m_thickness/2.0f),
                                        normalize(d * Double3( 1, 1,0)),
                                        normalize(d * Double3(-1,-1,0)))
                              );

        si.p = ray(pi.t);

#if 0
        if( 0 || ( ++dbg > 100000 ) ){
            std::cerr << "point2," << si.p[0] << "," << si.p[1] << "," << si.p[2] << "\n";
            std::cerr << "vec2," << ray.o[0] << "," << ray.o[1] << "," << ray.o[2] << "," << ray.d[0] << "," << ray.d[1] << "," << ray.d[2]  << "\n";
            //std::cerr << "vec3," << si.p[0] << "," << si.p[1] << "," << si.p[2] << "," << si.sh_frame.n[0] << "," << si.sh_frame.n[1] << "," << si.sh_frame.n[2] << "\n";
            dbg = 0;
            if( dbg == 1 ) usleep(1000);
        }
#endif

#if 0
        if (likely(has_flag(flags, HitComputeFlags::UV))) {
            Vector3f local = m_to_object.transform_affine(si.p);

            Float rd_2  = sqr(local.x()) + sqr(local.y()),
                  theta = unit_angle_z(local),
                  phi   = atan2(local.y(), local.x());

            masked(phi, phi < 0.f) += 2.f * math::Pi<Float>;

            si.uv = Point2f(phi * math::InvTwoPi<Float>, theta * math::InvPi<Float>);
            if (likely(has_flag(flags, HitComputeFlags::dPdUV))) {
                si.dp_du = Vector3f(-local.y(), local.x(), 0.f);

                Float rd      = sqrt(rd_2),
                      inv_rd  = rcp(rd),
                      cos_phi = local.x() * inv_rd,
                      sin_phi = local.y() * inv_rd;

                si.dp_dv = Vector3f(local.z() * cos_phi,
                                    local.z() * sin_phi,
                                    -rd);

                Mask singularity_mask = active && eq(rd, 0.f);
                if (unlikely(any(singularity_mask)))
                    si.dp_dv[singularity_mask] = Vector3f(1.f, 0.f, 0.f);

                si.dp_du = m_to_world * si.dp_du * (2.f * math::Pi<Float>);
                si.dp_dv = m_to_world * si.dp_dv * math::Pi<Float>;
            }
        }
#endif

        //if (m_flip_normals)
        //    si.sh_frame.n = -si.sh_frame.n;

        si.n = si.sh_frame.n;

#if 0
        if (has_flag(flags, HitComputeFlags::dNSdUV)) {
            ScalarFloat inv_radius = (m_flip_normals ? -1.f : 1.f) / m_radius0;
            si.dn_du = si.dp_du * inv_radius;
            si.dn_dv = si.dp_dv * inv_radius;
        }
#endif

        return si;
    }

    //! @}
    // =============================================================

    void traverse(TraversalCallback *callback) override {
        Base::traverse(callback);
    }

    void parameters_changed(const std::vector<std::string> &/*keys*/) override {
        update();
        Base::parameters_changed();
#if defined(MTS_ENABLE_OPTIX)
        optix_prepare_geometry();
#endif
    }

#if defined(MTS_ENABLE_OPTIX)
    using Base::m_optix_data_ptr;

    void optix_prepare_geometry() override {
        if constexpr (is_cuda_array_v<Float>) {
            if (!m_optix_data_ptr)
                m_optix_data_ptr = cuda_malloc(sizeof(OptixCylHollowData));

            OptixCylHollowData data = { bbox(), m_to_world, m_to_object,
                                     m_center, m_radius0, m_radius1, m_thickness,
                                     m_h, m_h_low, m_h_hi, m_flip_normals };

            cuda_memcpy_to_device(m_optix_data_ptr, &data, sizeof(OptixCylHollowData));
        }
    }
#endif

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "CylHollow[" << std::endl
            << "  to_world = " << string::indent(m_to_world, 13) << "," << std::endl
            << "  center = "  << m_center << "," << std::endl
            << "  radius0 = "  << m_radius0 << "," << std::endl
            << "  radius1 = "  << m_radius1 << "," << std::endl
            << "  thickness = "  << m_thickness << "," << std::endl
            << "  length = "  << m_h << "," << std::endl
            << "  h_low = "  << m_h_low << "," << std::endl
            << "  h_hi = "  << m_h_hi << "," << std::endl
            << "  surface_area = " << surface_area() << "," << std::endl
            << "  " << string::indent(get_children_string()) << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    /// Center in world-space
    ScalarPoint3f m_center;
    /// Radius in world-space
    ScalarFloat m_radius0;
    ScalarFloat m_radius1;
    /// Thickness of shell
    ScalarFloat m_thickness;
    /// Length of the tube
    ScalarFloat m_h;
    /// And the limits
    ScalarFloat m_h_low;
    ScalarFloat m_h_hi;

    ScalarFloat m_inv_surface_area;

    bool m_flip_normals;
};

MTS_IMPLEMENT_CLASS_VARIANT(CylHollow, Shape)
MTS_EXPORT_PLUGIN(CylHollow, "CylHollow intersection primitive");
NAMESPACE_END(mitsuba)
