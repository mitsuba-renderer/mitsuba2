
#include <mitsuba/core/fwd.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/core/util.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/shape.h>

#include <stdio.h>
#include <unistd.h>

#if defined(MTS_ENABLE_OPTIX)
#include "optix/aspheric_surf.cuh"
#endif

NAMESPACE_BEGIN(mitsuba)

    static int dbg = 0;
    static int dbg2 = 0;

    template <typename Float, typename Spectrum>
    class AsphSurf final : public Shape<Float, Spectrum> {
        public:
            MTS_IMPORT_BASE(Shape, m_to_world, m_to_object, set_children,
                            get_children_string, parameters_grad_enabled)
                MTS_IMPORT_TYPES()

                using typename Base::ScalarSize;

                using Double = std::conditional_t<is_cuda_array_v<Float>, Float, Float64>;
                using Double3 = Vector<Double, 3>;

            AsphSurf(const Properties &props) : Base(props) {
                /// Are the sphere normals pointing inwards? default: no
                m_flip_normals = props.bool_("flip_normals", false);

                // Flip curvature?
                m_flip = props.bool_("flip", false);

                // Update the to_world transform if radius and center are also provided
                m_to_world = m_to_world * ScalarTransform4f::translate(props.point3f("center", 0.f));
                m_to_world = m_to_world * ScalarTransform4f::scale(props.float_("radius", 1.f));

                // h limit is common to both
                m_h_lim = props.float_("limit", 0.0f);

                // First lens object - initial parameters
                m_k = props.float_("kappa0", 2.f);
                m_r = props.float_("radius0", 2.f);
                m_p =  1.0f / m_r;

                update();

                // AsphSurfes' z limit
                m_z_lim = ((pow(m_h_lim, 2.0f) * m_p) / (1 + sqrt(1 - (1 + m_k) * pow(m_h_lim*m_p,2.0f))));

                // How far into z plane?
                fprintf(stdout, "AsphSurf using flip=%s inv_norm=%s kappa=%.2f radius=%.2f (rho=%f) hlim=%.2f zlim=%.2f\n",
                        m_flip ? "true" : "false",
                        m_flip_normals ? "true" : "false",
                        (double) m_k, (double) m_r, (double) m_p, (double) m_h_lim, (double) m_z_lim);

                if( isnan( m_z_lim ) ){
                    fprintf(stdout, "nan error\n");
                    fflush(stdout);
                    while(1){};
                }


                set_children();
            }

            void update() {

                std::cerr << "Update\n";
                // Extract center and radius from to_world matrix (25 iterations for numerical accuracy)
                auto [S, Q, T] = transform_decompose(m_to_world.matrix, 25);

                if (abs(S[0][1]) > 1e-6f || abs(S[0][2]) > 1e-6f || abs(S[1][0]) > 1e-6f ||
                    abs(S[1][2]) > 1e-6f || abs(S[2][0]) > 1e-6f || abs(S[2][1]) > 1e-6f)
                    Log(Warn, "'to_world' transform shouldn't contain any shearing!");

                if (!(abs(S[0][0] - S[1][1]) < 1e-6f && abs(S[0][0] - S[2][2]) < 1e-6f))
                    Log(Warn, "'to_world' transform shouldn't contain non-uniform scaling!");

                m_center = T;
                m_radius = S[0][0];

                if (m_radius <= 0.f) {
                    Log(Error, "Radius must be > 0");
                }

                // Reconstruct the to_world transform with uniform scaling and no shear
                m_to_world = transform_compose(ScalarMatrix3f(m_radius), Q, T);
                m_to_object = m_to_world.inverse();

                m_inv_surface_area = rcp(surface_area());
            }


            ScalarBoundingBox3f bbox() const override {
                ScalarBoundingBox3f bbox;

                bbox.min = m_center - 1000;
                bbox.max = m_center + 1000;

                return bbox;
            }

#if 1
            ScalarFloat surface_area() const override {
                std::cerr << "surface_area\n";
                return 1000 * 4.f * math::Pi<ScalarFloat> * m_radius * m_radius;
            }
#endif

            // =============================================================
            //! @{ \name Sampling routines
            // =============================================================

            PositionSample3f sample_position(Float time, const Point2f &sample,
                                             Mask active) const override {
#if 1
                MTS_MASK_ARGUMENT(active);

                std::cout << "sample_position\n";
                std::cerr << "sample_position\n";
                std::cerr << sample << "\n";

                Point3f local = warp::square_to_uniform_sphere(sample);

                PositionSample3f ps;
                ps.p = fmadd(local, m_radius, m_center);
                ps.n = local;

                if (m_flip_normals)
                    ps.n = -ps.n;

                ps.time = time;
                ps.delta = m_radius == 0.f;
                ps.pdf = m_inv_surface_area;

                return ps;
#else
                PositionSample3f ps;
                return ps;
#endif
            }

            Float pdf_position(const PositionSample3f & /*ps*/, Mask active) const override {
                MTS_MASK_ARGUMENT(active);
                std::cerr << "pdf_position\n";
                std::cout << "pdf_position\n";
                return m_inv_surface_area;
            }

            DirectionSample3f sample_direction(const Interaction3f &it, const Point2f &sample,
                                               Mask active) const override {
                MTS_MASK_ARGUMENT(active);
                DirectionSample3f result = zero<DirectionSample3f>();

                std::cerr << "sample_direction\n";
                std::cout << "sample_direction\n";

                Vector3f dc_v = m_center - it.p;
                Float dc_2 = squared_norm(dc_v);

                Float radius_adj = m_radius * (m_flip_normals ? (1.f + math::RayEpsilon<Float>) :
                                               (1.f - math::RayEpsilon<Float>));
                Mask outside_mask = active && dc_2 > sqr(radius_adj);
                if (likely(any(outside_mask))) {
                    Float inv_dc            = rsqrt(dc_2),
                          sin_theta_max     = m_radius * inv_dc,
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
                    ds.p        = fmadd(d, m_radius, m_center);
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
                    ds.p        = fmadd(d, m_radius, m_center);
                    ds.n        = d;
                    ds.d        = ds.p - it.p;

                    Float dist2 = squared_norm(ds.d);
                    ds.dist     = sqrt(dist2);
                    ds.d        = ds.d / ds.dist;
                    ds.pdf      = m_inv_surface_area * dist2 / abs_dot(ds.d, ds.n);

                    result[inside_mask] = ds;
                }

                result.time = it.time;
                result.delta = m_radius == 0.f;

                if (m_flip_normals)
                    result.n = -result.n;

                return result;
            }

            Float pdf_direction(const Interaction3f &it, const DirectionSample3f &ds,
                                Mask active) const override {
                MTS_MASK_ARGUMENT(active);
                std::cerr << "pdf_direction\n";
                std::cout << "pdf_direction\n";

                // Sine of the angle of the cone containing the sphere as seen from 'it.p'.
                Float sin_alpha = m_radius * rcp(norm(m_center - it.p)),
                      cos_alpha = enoki::safe_sqrt(1.f - sin_alpha * sin_alpha);

                return select(sin_alpha < math::OneMinusEpsilon<Float>,
                              // Reference point lies outside the sphere
                              warp::square_to_uniform_cone_pdf(zero<Vector3f>(), cos_alpha),
                              m_inv_surface_area * sqr(ds.dist) / abs_dot(ds.d, ds.n)
                             );
            }

            Mask find_intersections( Double &near_t_, Double &far_t_,
                                     Double3 center,
                                     scalar_t<Double> m_p, scalar_t<Double> m_k,
                                     const Ray3f &ray) const{

                // Unit vector
                Double3 d(ray.d);

                // Origin
                Double3 o(ray.o);

                // Center of sphere
                Double3 c = Double3(center);

                Double dx = d[0], dy = d[1], dz = d[2];
                Double ox = o[0], oy = o[1], oz = o[2];

                Double x0 = c[0], y0 = c[1], z0 = c[2];

                Double g = -1 * ( 1 + m_k );

                Double A = -1 * g * pow(dz, 2.0) + pow(dx,2.0) + pow(dy,2.0);
                Double B = -1 * g * 2 * oz * dz + 2 * g * z0 * dz + 2 * ox * dx - 2 * x0 * dx + 2 * oy * dy - 2 * y0 * dy - 2 * dz / m_p;
                Double C = -1 * g * pow(oz, 2.0) + g * 2 * z0 * oz - g * pow(-1*z0,2.0) + pow(ox,2.0) - 2 * x0 * ox + pow(-1*x0,2.0) + pow(oy,2.0) - 2 * y0 * oy + pow(-1*y0,2.0) - 2 * oz / m_p - 2 * -1*z0 / m_p;

                auto [solution_found, near_t, far_t] = math::solve_quadratic(A, B, C);

                near_t_ = near_t;
                far_t_  = far_t;

                return solution_found;
            }

            Mask point_valid( Double3 t0, Double3 center,
                              scalar_t<Double> z_lim) const {

                Double3 delta0;
                Double hyp0;

                delta0 = t0 - center;

                hyp0 = sqrt( pow( delta0[0], 2.0) + pow(delta0[1], 2.0) + pow(delta0[2], 2.0) );

                Double limit;

                Double w = (Double) z_lim;

                limit = sqrt( (pow( (Double) m_h_lim, 2.0)) + pow(w, 2.0) );

                return (hyp0 <= limit);
            }

            //! @}
            // =============================================================

            // =============================================================
            //! @{ \name Ray tracing routines
            // =============================================================

            PreliminaryIntersection3f ray_intersect_preliminary(const Ray3f &ray,
                                                                Mask active) const override {
                MTS_MASK_ARGUMENT(active);

                Double mint = Double(ray.mint);
                Double maxt = Double(ray.maxt);

                //std::cout << " mint " << mint << " maxt " << maxt << "\n";

                // Point-solutions for each sphere
                Double near_t0, far_t0;

                near_t0 = 0.0;
                far_t0  = 0.0;

                Mask solution;

                if( m_flip ){

                    solution = find_intersections( near_t0, far_t0,
                                           m_center - Double3(0,0,m_r*2.f),
                                           (scalar_t<Double>) m_p, (scalar_t<Double>) m_k,
                                           ray);

                    // Works as long as origin ray "comes from the right
                    // direction" - needs fix.
                    near_t0 = far_t0;
                }
                else{
                    solution = find_intersections( near_t0, far_t0,
                                           m_center,
                                           (scalar_t<Double>) m_p, (scalar_t<Double>) m_k,
                                           ray);
                }

                // Where on the sphere plane is that?
                Mask valid0 = point_valid( ray(near_t0),
                                           m_center,
                                           (scalar_t<Double>) m_z_lim );

                valid0 = valid0 && solution && (near_t0 >= mint && near_t0 < maxt);

                /*
                 * Build the resulting ray.
                 * */
                PreliminaryIntersection3f pi = zero<PreliminaryIntersection3f>();

                pi.t = select( valid0, near_t0, math::Infinity<Float> );

                // Remember to set active mask
                active &= valid0;

                Ray3f out_ray;
                out_ray.o = ray( pi.t );

#if 1
                if( m_flip ){

                    if( 0 || ( ++dbg > 100000 ) ){

                        if( any(  valid0 ) ) {

                            std::cerr << "point1," << out_ray.o[0] << "," << out_ray.o[1] << "," << out_ray.o[2] << "\n";
                            std::cerr << "vec1," << ray.o[0] << "," << ray.o[1] << "," << ray.o[2] << "," << ray.d[0] << "," << ray.d[1] << "," << ray.d[2]  << "\n";
                        }
                        else{
                            //std::cerr << "point2," << out_ray.o[0] << "," << out_ray.o[1] << "," << out_ray.o[2] << "\n";
                            //std::cerr << "vec2," << ray.o[0] << "," << ray.o[1] << "," << ray.o[2] << "," << ray.d[0] << "," << ray.d[1] << "," << ray.d[2]  << "\n";
                        }
                        dbg = 0;
                    }

                    //usleep(1000);
                }
                else{ // !m_flip

                    if( 0 || ( ++dbg2 > 100000 ) ){
                        if( any( valid0 ) ) {

                            std::cerr << "point1," << out_ray.o[0] << "," << out_ray.o[1] << "," << out_ray.o[2] << "\n";
                            std::cerr << "vec1," << ray.o[0] << "," << ray.o[1] << "," << ray.o[2] << "," << ray.d[0] << "," << ray.d[1] << "," << ray.d[2]  << "\n";
                        }
                        else{
                            //std::cerr << "point1," << out_ray.o[0] << "," << out_ray.o[1] << "," << out_ray.o[2] << "\n";
                            //std::cerr << "vec1," << ray.o[0] << "," << ray.o[1] << "," << ray.o[2] << "," << ray.d[0] << "," << ray.d[1] << "," << ray.d[2]  << "\n";
                        }
                        dbg2 = 0;
                    }

                    //usleep(1000);
                }
#endif

                pi.shape = this;

                return pi;
            }

            Mask ray_test(const Ray3f &ray, Mask active) const override {
                MTS_MASK_ARGUMENT(active);

                Mask solution_found;

                PreliminaryIntersection3f pi;

                //Double mint = Double(ray.mint);
                //Double maxt = Double(ray.maxt);

                // Potentially gets cleared in ray_intersect_preliminary.
                Mask active_ = active;

                pi = ray_intersect_preliminary(ray, active_);

                solution_found = (pi.t != math::Infinity<Float>);

                /*
                 * This is not a volume so the concept of near_t and
                 * far_t is not working here I believe.
                 * */
#if 0
                // Sphere doesn't intersect with the segment on the ray
                Mask out_bounds = !(near_t <= maxt && far_t >= mint); // NaN-aware conditionals

                // Sphere fully contains the segment of the ray
                Mask in_bounds  = near_t < mint && far_t > maxt;

#endif
                return solution_found && /* !out_bounds && !in_bounds && */ active;
            }

            SurfaceInteraction3f compute_surface_interaction(const Ray3f &ray,
                                                             PreliminaryIntersection3f pi,
                                                             HitComputeFlags flags,
                                                             Mask active) const override {
                MTS_MASK_ARGUMENT(active);

                // 0xe based on quick look; check interaction.h
                //fprintf(stdout, "HitComputeFlags 0x%x\n", flags);

                bool differentiable = false;
                if constexpr (is_diff_array_v<Float>)
                    differentiable = requires_gradient(ray.o) ||
                        requires_gradient(ray.d) ||
                        parameters_grad_enabled();

                // Recompute ray intersection to get differentiable prim_uv and t
                if (differentiable && !has_flag(flags, HitComputeFlags::NonDifferentiable)){
                    pi = ray_intersect_preliminary(ray, active);
                }

                active &= pi.is_valid();

                SurfaceInteraction3f si = zero<SurfaceInteraction3f>();

                si.t = select(active, pi.t, math::Infinity<Float>);

                Double3 point = ray(pi.t) - m_center;

                /*
                 * Now compute the unit vector
                 * */
                Double fx, fy, fz;
                Double p(m_p);
                Double k(m_k);

                if( m_flip ){

                    fx = ( point[0] * p ) / sqrt( 1 - (1+k) * (pow(point[0], 2) + pow(point[1], 2)) * pow(p, 2));
                    fy = ( point[1] * p ) / sqrt( 1 - (1+k) * (pow(point[0], 2) + pow(point[1], 2)) * pow(p, 2));
                    fz = 1.0;
                }
                else{

                    fx = ( point[0] * p ) / sqrt( 1 - (1+k) * (pow(point[0], 2) + pow(point[1], 2)) * pow(p, 2));
                    fy = ( point[1] * p ) / sqrt( 1 - (1+k) * (pow(point[0], 2) + pow(point[1], 2)) * pow(p, 2));
                    fz = -1.0;
                }

                if( ! m_flip_normals )
                    si.sh_frame.n = normalize( Double3( fx, fy, fz ) );
                else
                    si.sh_frame.n = normalize( Double(-1) * Double3( fx, fy, fz ) );

                // Frame.n is a unit vector. between the center of the
                // ellipsis and the crossing point apparently.
                si.p = ray(pi.t);

#if 0
                if( 0 || ( ++dbg > 100000 ) ){
                    if(m_flip){
                        std::cerr << "point3," << si.p[0] << "," << si.p[1] << "," << si.p[2] << "\n";
                        std::cerr << "vec3," << ray.o[0] << "," << ray.o[1] << "," << ray.o[2] << "," << ray.d[0] << "," << ray.d[1] << "," << ray.d[2]  << "\n";
                        std::cerr << "vec4," << si.p[0] << "," << si.p[1] << "," << si.p[2] << "," << si.sh_frame.n[0] << "," << si.sh_frame.n[1] << "," << si.sh_frame.n[2] << "\n";
                    }
                    else{
                        std::cerr << "point1," << si.p[0] << "," << si.p[1] << "," << si.p[2] << "\n";
                        std::cerr << "vec1," << ray.o[0] << "," << ray.o[1] << "," << ray.o[2] << "," << ray.d[0] << "," << ray.d[1] << "," << ray.d[2]  << "\n";
                        std::cerr << "vec4," << si.p[0] << "," << si.p[1] << "," << si.p[2] << "," << si.sh_frame.n[0] << "," << si.sh_frame.n[1] << "," << si.sh_frame.n[2] << "\n";
                    }
                    if( dbg == 1 ) usleep(1000);
                    dbg = 0;
                }
#endif

                if (likely(has_flag(flags, HitComputeFlags::UV))) {

                    Vector3f local = m_to_object.transform_affine(si.p);

                    si.uv = Point2f( local.x() / m_r,
                                     local.y() / m_r );

                    if (likely(has_flag(flags, HitComputeFlags::dPdUV))) {

                        si.dp_du = Vector3f( fx, 1.0, 0.0 );
                        si.dp_dv = Vector3f( fy, 0.0, 1.0 );
                    }
                }

                si.n = si.sh_frame.n;

                if (has_flag(flags, HitComputeFlags::dNSdUV)) {
                    // Should not happen ATM.
                    std::cout << "dNSdUV\n";
                    Log(Warn, "dNSdUV");
                    Log(Error, "dNSdUV");
#if 0
                    ScalarFloat inv_radius = (m_flip_normals ? -1.f : 1.f) / m_radius;
                    si.dn_du = si.dp_du * inv_radius;
                    si.dn_dv = si.dp_dv * inv_radius;
#endif
                }

                return si;
            }

            //! @}
            // =============================================================

            void traverse(TraversalCallback *callback) override {
                std::cerr << "traverse\n";
                std::cout << "traverse\n";
                Base::traverse(callback);
            }

            void parameters_changed(const std::vector<std::string> &/*keys*/) override {
                std::cerr << "parameters_changed\n";
                std::cout << "parameters_changed\n";
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
                        m_optix_data_ptr = cuda_malloc(sizeof(OptixAsphSurfData));

                    OptixAsphSurfData data = { bbox(), m_to_world, m_to_object,
                        m_center, m_radius, m_k, m_p, m_r, m_h_lim, m_flip, m_z_lim,
                        m_flip_normals };

                    cuda_memcpy_to_device(m_optix_data_ptr, &data, sizeof(OptixAsphSurfData));
                }
            }
#endif

            std::string to_string() const override {
                std::ostringstream oss;
                oss << "AsphSurf[" << std::endl
                    << "  to_world = " << string::indent(m_to_world, 13) << "," << std::endl
                    << "  center = "  << m_center << "," << std::endl
                    << "  radius = "  << m_radius << "," << std::endl
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
                ScalarFloat m_radius;
                /// kappa
                ScalarFloat m_k;
                /// curvature
                ScalarFloat m_p;
                /// radius
                ScalarFloat m_r;

                /// limit of h
                ScalarFloat m_h_lim;

                /// flip curvature?
                bool m_flip;

                /// how far into the "z plane" the surface reaches
                /// -- it is a function of m_h_lim
                ScalarFloat m_z_lim;

                ScalarFloat m_inv_surface_area;

                bool m_flip_normals;
    };

MTS_IMPLEMENT_CLASS_VARIANT(AsphSurf, Shape)
    MTS_EXPORT_PLUGIN(AsphSurf, "AsphSurf intersection primitive");
NAMESPACE_END(mitsuba)
