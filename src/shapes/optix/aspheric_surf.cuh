
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

bool __device__ point_valid( Vector3f t0, Vector3f center, float z_lim, float h_lim) {

    Vector3f delta0;
    float hyp0;

    delta0 = t0 - center;

    hyp0 = sqrt( pow( delta0[0], 2.0) + pow(delta0[1], 2.0) + pow(delta0[2], 2.0) );

    float limit;

    float w = (float) z_lim;

    limit = sqrt( (pow( (float) h_lim, 2.0)) + pow(w, 2.0) );

    return (hyp0 <= limit);
}

bool __device__ find_intersections1( float &near_t, float &far_t,
                          Vector3f center, float z_lim,
                          float m_p, float m_k,
                          const Ray3f &ray){

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

bool __device__ find_intersections0( float &near_t, float &far_t,
                          Vector3f center,
                          float m_p, float m_k,
                          const Ray3f &ray){

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

extern "C" __global__ void __intersection__asphsurf() {
    const OptixHitGroupData *sbt_data = (OptixHitGroupData*) optixGetSbtDataPointer();
    OptixAsphSurfData *asurf = (OptixAsphSurfData *)sbt_data->data;

    // Ray in instance-space
    Ray3f ray = get_ray();

    float zplane_t;
    float near_t0, far_t0;

    Vector3f d;

    bool solution0;
    bool valid0, valid1;

    valid1 = (ray.d[2] != 0.0);

    if( asurf->flip ){
        solution0 = find_intersections1( near_t0, far_t0,
                                         //m_center,
                                         asurf->center + Vector3f(0,0, asurf->z_lim), 0.0f,
                                         asurf->p, asurf->k,
                                         ray);

        near_t0 = far_t0; // hack hack

        if( valid1 )
            zplane_t = (asurf->center[2] - ray.o[2]) / ray.d[2];
        else
            zplane_t = INFINITY;

        d = asurf->center - ray(zplane_t);
    }
    else{
        solution0 = find_intersections0( near_t0, far_t0,
                                         asurf->center,
                                         asurf->p, asurf->k,
                                         ray);

        if( valid1 )
            zplane_t = ((asurf->center[2] + asurf->z_lim) - ray.o[2]) / ray.d[2];
        else
            zplane_t = INFINITY;

        d = (asurf->center + Vector3f(0,0,asurf->z_lim)) - ray(zplane_t);
    }

    // I wish there was a prettier way of doing
    // this...
    valid1 = valid1 && (sqrt( pow( d[0], 2.0 ) + pow( d[1], 2.0 ) ) <= asurf->h_lim);

    valid1 = valid1 && (zplane_t >= ray.mint && zplane_t < ray.maxt) ;

    // Where on the sphere plane is that?

    valid0 = point_valid( ray(near_t0),
                    asurf->center + (asurf->flip ? Vector3f(0,0,asurf->z_lim) : Vector3f(0)),
                    asurf->z_lim, asurf->h_lim );

    valid0 = valid0 && solution0 && (near_t0 >= ray.mint && near_t0 < ray.maxt);

    if( valid0 ){
        optixReportIntersection( near_t0, OPTIX_HIT_KIND_TRIANGLE_FRONT_FACE );
    }
}


extern "C" __global__ void __closesthit__asphsurf() {
    unsigned int launch_index = calculate_launch_index();

    if (params.is_ray_test()) {
        params.out_hit[launch_index] = true;
    } else {
        const OptixHitGroupData *sbt_data = (OptixHitGroupData *) optixGetSbtDataPointer();
        OptixAsphSurfData *asurf = (OptixAsphSurfData *)sbt_data->data;

        // Ray in instance-space
        Ray3f ray = get_ray();

        // Early return for ray_intersect_preliminary call
        if (params.is_ray_intersect_preliminary()) {
            write_output_pi_params(params, launch_index, sbt_data->shape_ptr, 0, Vector2f(), ray.maxt);
            return;
        }

        /* Compute and store information describing the intersection. This is
           very similar to AsphSurf::compute_surface_interaction() */

        Vector3f p;

        // From cylinder.h
        p = ray( ray.maxt );

        Vector3f point = p - asurf->center;

        /*
         * Now compute the unit vector
         * */
        float fx, fy, fz;

        Vector3f ns;

        if( asurf->flip ){

            fx = 1 * ( ( point[0] * asurf->p ) / sqrt( 1 - (1+asurf->k) * (pow(point[0], 2) + pow(point[1], 2)) * pow(asurf->p, 2) ) );
            fy = 1 * ( ( point[1] * asurf->p ) / sqrt( 1 - (1+asurf->k) * (pow(point[0], 2) + pow(point[1], 2)) * pow(asurf->p, 2) ) );
            fz = 1.0;

            Vector3f surf1 = 1 * normalize( Vector3f( fx, fy, fz ) );

            ns = surf1;
        }
        else{

            fx = ( point[0] * asurf->p ) / sqrt( 1 - (1+asurf->k) * (pow(point[0], 2) + pow(point[1], 2)) * pow(asurf->p, 2) );
            fy = ( point[1] * asurf->p ) / sqrt( 1 - (1+asurf->k) * (pow(point[0], 2) + pow(point[1], 2)) * pow(asurf->p, 2) );
            fz = -1.0;

            Vector3f surf0 = normalize( Vector3f( fx, fy, fz ) );

            ns = surf0;
        }

        Vector3f ng = ns;

        Vector2f uv;
        Vector3f dp_du, dp_dv;
        if (params.has_uv()) {

            Vector3f local = asurf->to_object.transform_point(p);

            uv = Vector2f( local.x() / asurf->r,
                             local.y() / asurf->r );

            if (params.has_dp_duv()) {

                dp_du = Vector3f( fx, 1.0, 0.0 );
                dp_dv = Vector3f( fy, 0.0, 1.0 );

            }
        }

        Vector3f dn_du, dn_dv;

        dn_du = dp_du; // <<
        dn_dv = dp_dv; // Was flipped for negative radius

        // Produce all of this
        write_output_si_params(params, launch_index, sbt_data->shape_ptr,
                               0, p, uv, ns, ng, dp_du, dp_dv, dn_du, dn_dv, ray.maxt);
    }
}
#endif
