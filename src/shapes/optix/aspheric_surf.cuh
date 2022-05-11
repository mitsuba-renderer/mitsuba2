
#pragma once

#include <math.h>
#include <mitsuba/render/optix/common.h>
#include <mitsuba/render/optix/math.cuh>

struct OptixAsphSurfData {
    optix::BoundingBox3f bbox;
    optix::Transform4f to_world;
    optix::Transform4f to_object;
    optix::Vector3f center;
    float radius;

    float k;
    float p;
    float r;
    float h_lim;
    bool flip;
    float z_lim;

    bool flip_normals;
};

#ifdef __CUDACC__

#if 0

Mask point_valid( Vector3f t0, float3 center,
                  scalar_t<float> z_lim) const {

    Vector3f delta0;
    float hyp0;

    delta0 = t0 - center;

    hyp0 = sqrt( pow( delta0[0], 2.0) + pow(delta0[1], 2.0) + pow(delta0[2], 2.0) );

    float limit;

    float w = (float) z_lim;

    limit = sqrt( (pow( (float) m_h_lim, 2.0)) + pow(w, 2.0) );

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

    float mint = float(ray.mint);
    float maxt = float(ray.maxt);

    //std::cout << " mint " << mint << " maxt " << maxt << "\n";

    // Point-solutions for each sphere
    float near_t0, far_t0;
    float zplane_t;

    near_t0 = 0.0;
    far_t0  = 0.0;

    /*
     * The closest point on the initial curvature.
     * */
    Mask solution0;
    Mask solution1;

    // z-plane validity
    Mask valid1 = (ray.d[2] != 0.0);
    /*
     * 'd' is the distance from center of out zplane.
     * Use x/y distance vectors to check aperture circle boundaries.
     * */
    Vector3f d;

    if( m_flip ){
        solution0 = find_intersections1( near_t0, far_t0,
                                         //m_center,
                                         m_center + Vector3f(0,0, /*m_z_lim1*/ m_z_lim), (scalar_t<float>) /*m_z_lim*/ 0,
                                         (scalar_t<float>) m_p, (scalar_t<float>) m_k,
                                         ray);

        near_t0 = far_t0; // hack hack

        zplane_t = select( valid1,
                           float((m_center[2] - ray.o[2]) / ray.d[2]),
                           math::Infinity<Float> );

        d = m_center - ray(zplane_t);
    }
    else{
        solution0 = find_intersections0( near_t0, far_t0,
                                         m_center,
                                         (scalar_t<float>) m_p, (scalar_t<float>) m_k,
                                         ray);

        zplane_t = select( valid1,
                           ((m_center[2] + float(m_z_lim)) - ray.o[2]) / ray.d[2],
                           math::Infinity<Float> );

        d = (m_center + Vector3f(0,0,m_z_lim)) - ray(zplane_t);
    }

    // I wish there was a prettier way of doing
    // this...
    valid1 = valid1 && (sqrt( pow( d[0], 2.0 ) + pow( d[1], 2.0 ) ) <= m_h_lim);

    valid1 = valid1 && (zplane_t >= mint && zplane_t < maxt) ;

    float dist1 = select( valid1,
                           zplane_t, math::Infinity<Float> );

    // Where on the sphere plane is that?

    Mask valid0 = point_valid( ray(near_t0),
                               m_center + (m_flip ? Vector3f(0,0,m_z_lim) : float3(0)),
                               (scalar_t<float>) m_z_lim );

    valid0 = valid0 && solution0 && (near_t0 >= mint && near_t0 < maxt);

    float dist0 = select( valid0,
                           near_t0, math::Infinity<Float> );

    /*
     * Build the resulting ray.
     * */
    PreliminaryIntersection3f pi = zero<PreliminaryIntersection3f>();

    // Note: All rays from origin should have a hit.
    //       I think it should be an error if it does not.
#if 0
    pi.t = select( dist0 < dist1,
                   select( valid0, near_t0, math::Infinity<Float> ),
                   select( valid1, zplane_t, math::Infinity<Float> ) );
#else
    pi.t = select( valid0, near_t0, math::Infinity<Float> );
#endif

    // Remember to set active mask
    active &= valid0;

    Ray3f out_ray;
    out_ray.o = ray( pi.t );

#if 1
    if( m_flip ){

        if( 0 || ( ++dbg2 > 100000 ) ){

            if( any(  valid0 ) ) {

                std::cerr << "point3," << out_ray.o[0] << "," << out_ray.o[1] << "," << out_ray.o[2] << "\n";
                std::cerr << "vec3," << ray.o[0] << "," << ray.o[1] << "," << ray.o[2] << "," << ray.d[0] << "," << ray.d[1] << "," << ray.d[2]  << "\n";
            }
            else{
                //std::cerr << "point2," << out_ray.o[0] << "," << out_ray.o[1] << "," << out_ray.o[2] << "\n";
                //std::cerr << "vec2," << ray.o[0] << "," << ray.o[1] << "," << ray.o[2] << "," << ray.d[0] << "," << ray.d[1] << "," << ray.d[2]  << "\n";
            }
            dbg2 = 0;
        }

        //usleep(1000);
    }
    else{ // !m_flip

        if( 0 || ( ++dbg > 100000 ) ){
            if( any( valid0 ) ) {

                std::cerr << "point2," << out_ray.o[0] << "," << out_ray.o[1] << "," << out_ray.o[2] << "\n";
                std::cerr << "point1," << ray.o[0] << "," << ray.o[1] << "," << ray.o[2] << "\n";
                std::cerr << "vec1," << ray.o[0] << "," << ray.o[1] << "," << ray.o[2] << "," << ray.d[0] << "," << ray.d[1] << "," << ray.d[2]  << "\n";
            }
            else{
                //std::cerr << "point1," << out_ray.o[0] << "," << out_ray.o[1] << "," << out_ray.o[2] << "\n";
                //std::cerr << "vec1," << ray.o[0] << "," << ray.o[1] << "," << ray.o[2] << "," << ray.d[0] << "," << ray.d[1] << "," << ray.d[2]  << "\n";
            }
            dbg = 0;
        }

        //usleep(1000);
    }
#endif

    pi.shape = this;

    return pi;
}
#endif

bool find_intersections1( float &near_t, float &far_t,
                          Vector3f center, scalar_t<float> z_lim,
                          scalar_t<float> m_p, scalar_t<float> m_k,
                          const Ray3f &ray) const{

    // Unit vector
    Vector3f d = ray.d;

    // Origin
    Vector3f o = ray.o;

    // Center of sphere
    Vector3f c = center * -1;

    float w = (float) z_lim;

    float dx = d[0], dy = d[1], dz = d[2];
    float ox = o[0], oy = o[1], oz = o[2];


    float cx = c[0], cy = c[1], cz = c[2];

    float A = -1 * pow(dz, 2.0) - m_k * pow(dz, 2.0) - pow(dx, 2.0) - pow(dy, 2.0);
    float B = 2 *       w * dz - 2       * cz * dz - 2       * oz * dz +
        2 * m_k * w * dz - 2 * m_k * cz * dz - 2 * m_k * oz * dz -
        2 * cx * dx - 2 * ox * dx - 2 * cy * dy - 2 * oy * dy -
        2 * dz / m_p;
    float C = -1 * pow(w, 2.0) + 2     * w * cz + 2 * w     * oz     - pow(cz, 2.0) -     2 * cz * oz     - pow(oz, 2.0) -
        m_k * pow(w, 2.0) + 2 * m_k * w * cz + 2 * m_k * w * oz - m_k * pow(cz, 2.0) - m_k * 2 * cz * oz - m_k * pow(oz, 2.0) -
        pow(cx, 2.0) - 2 * cx * ox - pow(ox, 2.0) - pow(cy, 2.0) - 2 * cy * oy - pow(oy, 2.0) + 2 * w / m_p - 2 * cz / m_p - 2 * oz / m_p;

    bool solution_found = solve_quadratic(A, B, C, near_t, far_t);

    return solution_found;
}

bool find_intersections0( float &near_t, float &far_t,
                          Vector3f center,
                          scalar_t<float> m_p, scalar_t<float> m_k,
                          const Ray3f &ray) const{

    // Unit vector
    Vector3f d = ray.d;

    // Origin
    Vector3f o = ray.o;

    // Center of sphere
    Vector3f c = center;

    float dx = d[0], dy = d[1], dz = d[2];
    float ox = o[0], oy = o[1], oz = o[2];

    float x0 = c[0], y0 = c[1], z0 = c[2];

    float g = -1 * ( 1 + m_k );

    float A = -1 * g * pow(dz, 2.0) + pow(dx,2.0) + pow(dy,2.0);
    float B = -1 * g * 2 * oz * dz + 2 * g * z0 * dz + 2 * ox * dx - 2 * x0 * dx + 2 * oy * dy - 2 * y0 * dy - 2 * dz / m_p;
    float C = -1 * g * pow(oz, 2.0) + g * 2 * z0 * oz - g * pow(-1*z0,2.0) + pow(ox,2.0) - 2 * x0 * ox + pow(-1*x0,2.0) + pow(oy,2.0) - 2 * y0 * oy + pow(-1*y0,2.0) - 2 * oz / m_p - 2 * -1*z0 / m_p;

    bool solution_found = solve_quadratic(A, B, C, near_t, far_t);

    return solution_found;
}

extern "C" __global__ void __intersection__sphere() {
    const OptixHitGroupData *sbt_data = (OptixHitGroupData*) optixGetSbtDataPointer();
    OptixAsphSurfData *sphere = (OptixSphereData *)sbt_data->data;

    // Ray in instance-space
    Ray3f ray = get_ray();

#if 1

    Vector3f o = ray.o - sphere->center;
    Vector3f d = ray.d;

    float A = squared_norm(d);
    float B = 2.f * dot(o, d);
    float C = squared_norm(o) - sqr(sphere->radius);

    float near_t, far_t;
    bool solution_found = solve_quadratic(A, B, C, near_t, far_t);

    // AsphSurf doesn't intersect with the segment on the ray
    bool out_bounds = !(near_t <= ray.maxt && far_t >= ray.mint); // NaN-aware conditionals

    // AsphSurf fully contains the segment of the ray
    bool in_bounds = near_t < ray.mint && far_t > ray.maxt;

    float t = (near_t < ray.mint ? far_t: near_t);
#endif

    if (solution_found && !out_bounds && !in_bounds)
        optixReportIntersection(t, OPTIX_HIT_KIND_TRIANGLE_FRONT_FACE);
}


extern "C" __global__ void __closesthit__sphere() {
    unsigned int launch_index = calculate_launch_index();

    if (params.is_ray_test()) {
        params.out_hit[launch_index] = true;
    } else {
        const OptixHitGroupData *sbt_data = (OptixHitGroupData *) optixGetSbtDataPointer();
        OptixAsphSurfData *sphere = (OptixSphereData *)sbt_data->data;

        // Ray in instance-space
        Ray3f ray = get_ray();

        // Early return for ray_intersect_preliminary call
        if (params.is_ray_intersect_preliminary()) {
            write_output_pi_params(params, launch_index, sbt_data->shape_ptr, 0, Vector2f(), ray.maxt);
            return;
        }

        /* Compute and store information describing the intersection. This is
           very similar to AsphSurf::compute_surface_interaction() */

        Vector3f ns = normalize(ray(ray.maxt) - sphere->center);

        if (sphere->flip_normals)
            ns = -ns;

        Vector3f ng = ns;

        // Re-project onto the sphere to improve accuracy
        Vector3f p = fmaf(sphere->radius, ns, sphere->center);

        Vector2f uv;
        Vector3f dp_du, dp_dv;
        if (params.has_uv()) {
            Vector3f local = sphere->to_object.transform_point(p);

            float rd_2  = sqr(local.x()) + sqr(local.y()),
                  theta = acos(local.z()),
                  phi   = atan2(local.y(), local.x());

            if (phi < 0.f)
                phi += TwoPi;

            uv = Vector2f(phi * InvTwoPi, theta * InvPi);

            if (params.has_dp_duv()) {
                dp_du = Vector3f(-local.y(), local.x(), 0.f);

                float rd      = sqrt(rd_2),
                      inv_rd  = 1.f / rd,
                      cos_phi = local.x() * inv_rd,
                      sin_phi = local.y() * inv_rd;

                dp_dv = Vector3f(local.z() * cos_phi,
                                local.z() * sin_phi,
                                -rd);

                // Check for singularity
                if (rd == 0.f)
                    dp_dv = Vector3f(1.f, 0.f, 0.f);

                dp_du = sphere->to_world.transform_vector(dp_du) * TwoPi;
                dp_dv = sphere->to_world.transform_vector(dp_dv) * Pi;
            }
        }

        float inv_radius = (sphere->flip_normals ? -1.f : 1.f) / sphere->radius;
        Vector3f dn_du = dp_du * inv_radius;
        Vector3f dn_dv = dp_dv * inv_radius;

        write_output_si_params(params, launch_index, sbt_data->shape_ptr,
                               0, p, uv, ns, ng, dp_du, dp_dv, dn_du, dn_dv, ray.maxt);
    }
}
#endif
