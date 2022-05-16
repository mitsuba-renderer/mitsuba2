#include <mitsuba/render/sensor.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/core/bbox.h>
#include <mitsuba/core/warp.h>

#include <unistd.h>
#include <stdio.h>

NAMESPACE_BEGIN(mitsuba)

    /**!

      .. _sensor-thinlens:

      Perspective camera with a thin lens (:monosp:`thinlens`)
      --------------------------------------------------------

      .. pluginparameters::

     * - to_world
     - |transform|
     - Specifies an optional camera-to-world transformation.
     (Default: none (i.e. camera space = world space))
     * - aperture_radius
     - |float|
     - Denotes the radius of the camera's aperture in scene units.
     * - focus_distance
     - |float|
     - Denotes the world-space distance from the camera's aperture to the focal plane.
     (Default: :monosp:`0`)
     * - focal_length
     - |string|
     - Denotes the camera's focal length specified using *35mm* film equivalent units.
     See the main description for further details. (Default: :monosp:`50mm`)
     * - fov
     - |float|
     - An alternative to :monosp:`focal_length`: denotes the camera's field of view in degrees---must be
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
     * - near_clip, far_clip
     - |float|
     - Distance to the near/far clip planes. (Default: :monosp:`near_clip=1e-2` (i.e. :monosp:`0.01`)
     and :monosp:`far_clip=1e4` (i.e. :monosp:`10000`))

     .. subfigstart::
     .. subfigure:: ../../resources/data/docs/images/render/sensor_thinlens_small_aperture.jpg
     :caption: The material test ball viewed through a perspective thin lens camera. (:monosp:`aperture_radius=0.1`)
     .. subfigure:: ../../resources/data/docs/images/render/sensor_thinlens.jpg
     :caption: The material test ball viewed through a perspective thin lens camera. (:monosp:`aperture_radius=0.2`)
     .. subfigend::
     :label: fig-thinlens

     This plugin implements a simple perspective camera model with a thin lens
     at its circular aperture. It is very similar to the
     :ref:`perspective <sensor-perspective>` plugin except that the extra lens element
     permits rendering with a specifiable (i.e. non-infinite) depth of field.
     To configure this, it has two extra parameters named :monosp:`aperture_radius`
     and :monosp:`focus_distance`.

     By default, the camera's field of view is specified using a 35mm film
     equivalent focal length, which is first converted into a diagonal field
     of view and subsequently applied to the camera. This assumes that
     the film's aspect ratio matches that of 35mm film (1.5:1), though the
     parameter still behaves intuitively when this is not the case.
     Alternatively, it is also possible to specify a field of view in degrees
     along a given axis (see the :monosp:`fov` and :monosp:`fov_axis` parameters).

     The exact camera position and orientation is most easily expressed using the
    :monosp:`lookat` tag, i.e.:

    .. code-block:: xml

    <sensor type="thinlens">
    <transform name="to_world">
<!-- Move and rotate the camera so that looks from (1, 1, 1) to (1, 2, 1)
    and the direction (0, 0, 1) points "up" in the output image -->
    <lookat origin="1, 1, 1" target="1, 2, 1" up="0, 0, 1"/>
    </transform>

    <!-- Focus on the target -->
    <float name="focus_distance" value="1"/>
    <float name="aperture_radius" value="0.1"/>
    </sensor>

    */


    static int dbg = 0;
    static int dbg2 = 0;

    template <typename Float, typename Spectrum>
    class StartLensCamera final : public ProjectiveCamera<Float, Spectrum> {
        public:
            MTS_IMPORT_BASE(ProjectiveCamera, m_world_transform, m_needs_sample_3, m_film, m_sampler,
                            m_resolution, m_shutter_open, m_shutter_open_time, m_near_clip,
                            m_far_clip, m_focus_distance)
                MTS_IMPORT_TYPES()

                using Double = std::conditional_t<is_cuda_array_v<Float>, Float, Float64>;
            using Double3 = Vector<Double, 3>;

            // =============================================================
            //! @{ \name Constructors
            // =============================================================

            StartLensCamera(const Properties &props) : Base(props) {
                ScalarVector2i size = m_film->size();
                m_x_fov = parse_fov(props, size.x() / (float) size.y());

                m_aperture_radius = props.float_("aperture_radius");

                if (m_aperture_radius == 0.f) {
                    Log(Warn, "Can't have a zero aperture radius -- setting to %f", math::Epsilon<Float>);
                    m_aperture_radius = math::Epsilon<Float>;
                }

                if (m_world_transform->has_scale())
                    Throw("Scale factors in the camera-to-world transformation are not allowed!");

                m_camera_to_sample = perspective_projection(
                                                            m_film->size(), m_film->crop_size(), m_film->crop_offset(),
                                                            m_x_fov, m_near_clip, m_far_clip);

                m_sample_to_camera = m_camera_to_sample.inverse();

                // Position differentials on the near plane
                m_dx = m_sample_to_camera * ScalarPoint3f(1.f / m_resolution.x(), 0.f, 0.f)
                    - m_sample_to_camera * ScalarPoint3f(0.f);
                m_dy = m_sample_to_camera * ScalarPoint3f(0.f, 1.f / m_resolution.y(), 0.f)
                    - m_sample_to_camera * ScalarPoint3f(0.f);

                /* Precompute some data for importance(). Please
                   look at that function for further details. */
                ScalarPoint3f pmin(m_sample_to_camera * ScalarPoint3f(0.f, 0.f, 0.f)),
                              pmax(m_sample_to_camera * ScalarPoint3f(1.f, 1.f, 0.f));

                m_image_rect.reset();
                m_image_rect.expand(ScalarPoint2f(pmin.x(), pmin.y()) / pmin.z());
                m_image_rect.expand(ScalarPoint2f(pmax.x(), pmax.y()) / pmax.z());
                m_normalization = 1.f / m_image_rect.volume();
                m_needs_sample_3 = true;

                /*
                 * Lens specific
                 * */

                // h limit is common to both
                //m_h_lim = (double) props.float_("limit", 0.0f);
                m_h_lim = m_aperture_radius;

                fprintf(stdout, "Aperture radius: %.2f\n", m_h_lim);

                // First lens object - initial parameters
                m_k0 = props.float_("kappa0", 2.f);
                m_r0 = props.float_("radius0", 2.f);
                m_p0 = 1.0f / m_r0;

                // Lens' z-limit
                m_z_lim0 = ((enoki::pow(m_h_lim, 2.0f) * m_p0) / (1 + sqrt(1 - (1 + m_k0) * (enoki::pow(m_h_lim,2.0f) * enoki::pow(m_p0, 2.0f)))));

                m_center = ScalarPoint3f(0.0,0.0, 0.0 - m_z_lim0);

                lens1_offset = m_z_lim0;

                //std::cerr << m_world_transform << "\n";

                fprintf(stdout, "Lens0 center %.2f %.2f %.2f\n", m_center[0], m_center[1], m_center[2]);
                fprintf(stdout, "Lens0 using kappa=%.2f radius=%.2f (rho=%f) hlim=%.2f zlim=%.2f\n",
                         m_k0,  m_r0,  m_p0,  m_h_lim,  m_z_lim0 );
                fprintf(stdout, "Lens0 center1 %.2f %.2f %.2f\n", m_center1[0], m_center1[1], m_center1[2]);

                if( isnan( m_z_lim0 ) || isnan(m_z_lim1)){
                    fprintf(stdout, "nan error\n");
                    fflush(stdout);
                    while(1){};
                }
            }

            Mask find_intersections0( Double &near_t_, Double &far_t_,
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

                Double A = -1 * g * enoki::pow(dz, 2.0f) + enoki::pow(dx,2.0f) + enoki::pow(dy,2.0f);
                Double B = -1 * g * 2 * oz * dz + 2 * g * z0 * dz + 2 * ox * dx - 2 * x0 * dx + 2 * oy * dy - 2 * y0 * dy - 2 * dz / m_p;
                Double C = -1 * g * enoki::pow(oz, 2.0f) + g * 2 * z0 * oz - g * enoki::pow(-1*z0,2.0f) + enoki::pow(ox,2.0f) - 2 * x0 * ox + enoki::pow(-1*x0,2.0f) + enoki::pow(oy,2.0f) - 2 * y0 * oy + enoki::pow(-1*y0,2.0f) - 2 * oz / m_p - 2 * -1*z0 / m_p;

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

                hyp0 = sqrt( enoki::pow( delta0[0], 2.0f) + enoki::pow(delta0[1], 2.0f) + enoki::pow(delta0[2], 2.0f) );

                Double limit;

                Double w = (Double) z_lim;

                limit = sqrt( (enoki::pow( (Double) m_h_lim, 2.0f)) + enoki::pow(w, 2.0f) );

#if 0
                std::cout << "Considering:\n";
                std::cout << m_h_lim << " ----- " << z_lim << "\n";
                std::cout << center << "\n";
                std::cout << "t0 " << t0 << "\n";
                std::cout << "hyp0 " << hyp0 << "\n";
                std::cout << "limit " << limit << "\n";
                std::cout << "result: " << (hyp0 <= limit) << "\n";
#endif


                return (hyp0 <= limit);
            }


            void compute_output_ray( Ray3f &out_ray, Ray3f ray ) const {

                // Point-solutions for each sphere
                Double near_t0;
                Double far_t0;

                near_t0 = 0.0;
                far_t0  = 0.0;

                /*
                 * The closest point on the initial curvature.
                 * */
                Mask solution0 = find_intersections0( near_t0, far_t0,
                                           m_center,
                                           (scalar_t<Double>) m_p0, (scalar_t<Double>) m_k0,
                                           ray);
                (void) near_t0; // unused

                // Is it within h lim?
                //  -- dont need this check so long as
                //  solution0 is true
                /*
                auto valid0 = point_valid( ray(far_t0),
                                           m_center,
                                           (scalar_t<Double>) m_z_lim0 );
                */

                // Is the point valid?
                if(1){
                    //std::cout << valid0 << " --- " << solution0 << " --- " << "\n";

                    Mask valid0 = /*valid0 && */ solution0;

                    if( any( ! ( valid0 ) ) ){

                        // Resulting
                        std::cout << "valid0 " << valid0 << " solution0 " << solution0 << "\n";
                        std::cerr << "point2," << ray.o[0] << "," << ray.o[1] << "," << ray.o[2] << "\n";
                        std::cerr << "vec2," << ray.o[0] << "," << ray.o[1] << "," << ray.o[2] << "," << ray.d[0] << "," << ray.d[1] << "," << ray.d[2]  << "\n";

                        Double3 p = ray( far_t0 );

                        std::cerr << "point3," << p[0] << "," << p[1] << "," << p[2] << "\n";


                        std::cout << "far_t0 is " << far_t0 << "\n";
                        std::cout << "near_t0 is " << near_t0 << "\n";
                        fprintf(stdout, "There should always be a solution, check parameters");
                        fflush(stdout);
                        fflush(stderr);
                        while(1){};
                    }
                }

#if 1 // Origin on sphere surface --- this is numerically difficult to get exact with the next shape...
                out_ray.o = ray( far_t0 );
                out_ray.d = normalize( out_ray.o - ray.o );
#else
                out_ray.o = ray( far_t0 * (5.0 / 10.0) );
                out_ray.d = normalize( out_ray.o - ray.o );

#endif
            }

            //! @}
            // =============================================================

            // =============================================================
            //! @{ \name Sampling methods (Sensor interface)
            // =============================================================

            std::pair<Ray3f, Spectrum> sample_ray(Float time, Float wavelength_sample,
                                                  const Point2f &position_sample,
                                                  const Point2f &aperture_sample,
                                                  Mask active) const override {
                MTS_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);

                auto [wavelengths, wav_weight] = sample_wavelength<Float, Spectrum>(wavelength_sample);
                Ray3f ray;
                ray.time = time;
                ray.wavelengths = wavelengths;

                // Compute the sample position on the near plane (local camera space).
                Point3f near_p = m_sample_to_camera *
                    Point3f(position_sample.x(), position_sample.y(), 0.f);

                // Aperture position
                Point2f tmp = m_aperture_radius * warp::square_to_uniform_disk_concentric(aperture_sample);
                Point3f aperture_p(tmp.x(), tmp.y(), 0.f);

                // Sampled position on the focal plane
                Point3f focus_p = near_p * (m_focus_distance / near_p.z());

                // Convert into a normalized ray direction; adjust the ray interval accordingly.
                Vector3f d = normalize(Vector3f(focus_p - aperture_p));
                Float inv_z = rcp(d.z());

                // ORIGINAL RAY CALC
                //ray.mint = m_near_clip * inv_z;
                //ray.maxt = m_far_clip * inv_z;

                //auto trafo = m_world_transform->eval(ray.time, active);
                //ray.o = trafo.transform_affine(aperture_p);
                //ray.d = trafo * d;
                //ray.update();
                // ORIGINAL RAY CALC END

                // Image plane ray.
                // Origin at image plane (position_sample).
                // Unit vector points to crosspoint with aperture (aperture_sample).
                Ray3f ip_ray;
                ip_ray.o = focus_p;
                ip_ray.d = -1 * d; // Why did I flip this...?


                // TEST START HERE
                Ray3f test_ray;

                compute_output_ray( test_ray, ip_ray );

#if 0
                if( 0 || ( ++dbg > 1000000 ) ){
                    // Origin on sensor and direction
                    //std::cerr << "point0," << ip_ray.o[0] << "," << ip_ray.o[1] << "," << ip_ray.o[2] << "\n";
                    //std::cerr << "vec0," << ip_ray.o[0] << "," << ip_ray.o[1] << "," << ip_ray.o[2] << "," << ip_ray.d[0] << "," << ip_ray.d[1] << "," << ip_ray.d[2]  << "\n";

                    // Resulting
                    std::cerr << "point1," << test_ray.o[0] << "," << test_ray.o[1] << "," << test_ray.o[2] << "\n";
                    std::cerr << "vec1," << test_ray.o[0] << "," << test_ray.o[1] << "," << test_ray.o[2] << "," << test_ray.d[0] << "," << test_ray.d[1] << "," << test_ray.d[2]  << "\n";

                    //std::cerr << "point2," << aperture_p[0] << "," << aperture_p[1] << "," << aperture_p[2] << "\n";


                    dbg = 0;

                    usleep(1000);
                }
#endif

                ray.mint = m_near_clip * inv_z;
                ray.maxt = m_far_clip * inv_z;

                auto trafo = m_world_transform->eval(ray.time, active);

#if 0
                ray.o = trafo.transform_affine( test_ray.o );
                ray.d = (trafo * test_ray.d) * Double3(1.0,1.0,-1.0) ;
#else
                ray.o = trafo.transform_affine( near_p * (-1 * (m_focus_distance / near_p.z() )) );
                ray.d = normalize( trafo.transform_affine( test_ray.o ) - ray.o );
#endif

#if 0
                if( 0 || ( ++dbg > 10000 ) ){

                    std::cerr << "point0," << ray.o[0] << "," << ray.o[1] << "," << ray.o[2] << "\n";
                    std::cerr << "vec0," << ray.o[0] << "," << ray.o[1] << "," << ray.o[2]<< "," << ray.d[0] << "," << ray.d[1] << "," << ray.d[2] << "\n";

                    //if( dbg == 1 ) usleep(1000);

                    /*
                    Ray3f aperture_ray;
                    aperture_ray.o = trafo.transform_affine( aperture_p );

                    std::cerr << "point0," << aperture_ray.o[0] << "," << aperture_ray.o[1] << "," << aperture_ray.o[2] << "\n";
                    usleep(1000);
                    */

                    dbg = 0;

                }
#endif

                ray.update();

                return std::make_pair(ray, wav_weight);
            }

            std::pair<RayDifferential3f, Spectrum>
                sample_ray_differential_impl(Float time, Float wavelength_sample,
                                             const Point2f &position_sample, const Point2f &aperture_sample,
                                             Mask active) const {
                    MTS_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);

                    fprintf(stdout, "differential\n");
                    fprintf(stdout, "differential\n");
                    fprintf(stdout, "differential\n");
                    fprintf(stdout, "differential\n");
                    fprintf(stdout, "differential\n");

                    while(1) {};

                    auto [wavelengths, wav_weight] = sample_wavelength<Float, Spectrum>(wavelength_sample);
                    RayDifferential3f ray;
                    ray.time = time;
                    ray.wavelengths = wavelengths;

                    // Compute the sample position on the near plane (local camera space).
                    Point3f near_p = m_sample_to_camera *
                        Point3f(position_sample.x(), position_sample.y(), 0.f);

                    // Aperture position
                    Point2f tmp = m_aperture_radius * warp::square_to_uniform_disk_concentric(aperture_sample);
                    Point3f aperture_p(tmp.x(), tmp.y(), 0.f);

                    // Sampled position on the focal plane
                    Float f_dist = m_focus_distance / near_p.z();
                    Point3f focus_p   = near_p          * f_dist,
                            focus_p_x = (near_p + m_dx) * f_dist,
                            focus_p_y = (near_p + m_dy) * f_dist;

                    // Convert into a normalized ray direction; adjust the ray interval accordingly.
                    Vector3f d = normalize(Vector3f(focus_p - aperture_p));
                    Float inv_z = rcp(d.z());
                    ray.mint = m_near_clip * inv_z;
                    ray.maxt = m_far_clip * inv_z;

                    auto trafo = m_world_transform->eval(ray.time, active);
                    ray.o = trafo.transform_affine(aperture_p);
                    ray.d = trafo * d;
                    ray.update();

                    ray.o_x = ray.o_y = ray.o;

                    ray.d_x = trafo * normalize(Vector3f(focus_p_x - aperture_p));
                    ray.d_y = trafo * normalize(Vector3f(focus_p_y - aperture_p));
                    ray.has_differentials = true;

#if 0
                    if( 0 || (++dbg == 1) ){
                        fprintf(stderr, "vec,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",
                                ray.o[0], ray.o[1], ray.o[2],
                                ray.d[0], ray.d[1], ray.d[2]);
                        dbg = 0;
                    }
#endif

                    return std::make_pair(ray, wav_weight);
                }

            ScalarBoundingBox3f bbox() const override {
                return m_world_transform->translation_bounds();
            }

            //! @}
            // =============================================================

            void traverse(TraversalCallback *callback) override {
                Base::traverse(callback);
                // TODO aperture_radius, x_fov
            }

            void parameters_changed(const std::vector<std::string> &keys) override {
                Base::parameters_changed(keys);
                // TODO
            }

            std::string to_string() const override {
                using string::indent;

                std::ostringstream oss;
                oss << "StartLensCamera[" << std::endl
                    << "  x_fov = " << m_x_fov << "," << std::endl
                    << "  near_clip = " << m_near_clip << "," << std::endl
                    << "  far_clip = " << m_far_clip << "," << std::endl
                    << "  focus_distance = " << m_focus_distance << "," << std::endl
                    << "  film = " << indent(m_film) << "," << std::endl
                    << "  sampler = " << indent(m_sampler) << "," << std::endl
                    << "  resolution = " << m_resolution << "," << std::endl
                    << "  shutter_open = " << m_shutter_open << "," << std::endl
                    << "  shutter_open_time = " << m_shutter_open_time << "," << std::endl
                    << "  world_transform = " << indent(m_world_transform)  << std::endl
                    << "]";
                return oss.str();
            }

            MTS_DECLARE_CLASS()
        private:
                ScalarTransform4f m_camera_to_sample;
                ScalarTransform4f m_sample_to_camera;
                ScalarBoundingBox2f m_image_rect;
                ScalarFloat m_aperture_radius;
                ScalarFloat m_normalization;
                ScalarFloat m_x_fov;
                ScalarVector3f m_dx, m_dy;

                /// Center in world-space
                ScalarPoint3f m_center;
                ScalarPoint3f m_center1;

                /// kappa
                float m_k0;
                float m_k1;
                /// curvature
                float m_p0;
                float m_p1;
                /// radius
                float m_r0;
                float m_r1;

                /// limit of h
                float m_h_lim;

                /// offset to second lens element
                float lens1_offset;

                /// how far into the "z plane" the surface reaches
                /// -- it is a function of m_h_lim
                float m_z_lim0;
                float m_z_lim1;

    };

MTS_IMPLEMENT_CLASS_VARIANT(StartLensCamera, ProjectiveCamera)
    MTS_EXPORT_PLUGIN(StartLensCamera, "Start Lens Camera");
NAMESPACE_END(mitsuba)
