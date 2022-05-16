
#pragma once

#include <math.h>
#include <mitsuba/render/optix/common.h>
#include <mitsuba/render/optix/math.cuh>

struct OptixCylHollowData {
    optix::BoundingBox3f bbox;
    optix::Transform4f to_world;
    optix::Transform4f to_object;
    optix::Vector3f center;
    float radius0;
    float radius1;
    float thickness;
    float h;
    float h_low;
    float h_hi;
    bool flip_normals;
};

#ifdef __CUDACC__

float __device__ intersect( OptixCylHollowData * cylh,
                            float mint, float maxt,
                            float radius,
                            const Ray3f &ray){

    float res;
    float near_t, far_t;
    bool valid0, valid1;
    bool solution_found;

    float x0 = cylh->center[0];
    float y0 = cylh->center[1];

    float ox = ray.o[0],
          oy = ray.o[1];

    float dx = ray.d[0],
          dy = ray.d[1];

    float A = pow( dx, 2.0f ) + pow( dy, 2.0f );
    float B = 2.0f * ox * dx - 2.0f * x0 * dx +
               2.0f * oy * dy - 2.0f * y0 * dy;
    float C = pow( ox, 2.0f ) - 2.0f * ox * x0 + pow( x0, 2.0f ) +
               pow( oy, 2.0f ) - 2.0f * oy * y0 + pow( y0, 2.0f ) -
               pow( radius, 2.0f );

    solution_found = solve_quadratic( A, B, C, near_t, far_t );

    float z_lim_low = cylh->h_low;
    float z_lim_hi  = cylh->h_hi;

    valid0 = solution_found && ((near_t > mint) && (near_t < maxt));
    valid0 = valid0 && (ray(near_t)[2] >= z_lim_low) && (ray(near_t)[2] < z_lim_hi);

    valid1 = solution_found && ((far_t > mint) && (far_t < maxt));
    valid1 = valid1 && (ray(far_t)[2] >= z_lim_low) && (ray(far_t)[2] < z_lim_hi);

    //res = select( valid0, near_t,
    //             select( valid1, far_t, math::Infinity<float> ) );
    if( valid0 ){
        res = near_t;
    }
    else{
        if( valid1 ){
            res = far_t;
        }
        else{
            res = INFINITY;
        }
    }

    return res;
}

float __device__ zintersect( OptixCylHollowData * cylh,
                             float mint, float maxt,
                             Vector3f center,
                             const Ray3f &ray){

    float t;
    bool valid;
    bool bounds;

    float oz = float( ray.o[2] );
    float dz = float( ray.d[2] );
    float a = center[2] ;

    valid = (dz != 0);
    //t = select(valid, (a - oz) / dz, math::Infinity<float> );
    if( valid ){
        t = (a - oz) / dz;
    }
    else{
        t = INFINITY;
    }

    valid = (t >= mint) && (t <= maxt);
    //t = select(valid, t, math::Infinity<float>);
    if( ! valid ){
        t = INFINITY;
    }

    Vector3f diff = center - ray( t );

    float h = sqrt( dot( diff, diff ) );

    bounds = (h >= cylh->radius0) && (h <= cylh->radius1);

    float res; // = select( bounds, t, math::Infinity<float> );
    if( bounds ){
        res = t;
    }
    else{
        res = INFINITY;
    }

    return res;
}

extern "C" __global__ void __intersection__cylhollow() {
    const OptixHitGroupData *sbt_data = (OptixHitGroupData*) optixGetSbtDataPointer();
    OptixCylHollowData *cylh = (OptixCylHollowData *)sbt_data->data;

    Ray3f ray = get_ray();
    float mint = ray.mint;
    float maxt = ray.maxt;

    float t0, t1;
    float z0, z1;

    t0 = intersect( cylh, mint, maxt, cylh->radius0, ray );
    t1 = intersect( cylh, mint, maxt, cylh->radius1, ray );
    z0 = zintersect( cylh, mint, maxt, cylh->center + Vector3f(0,0,0), ray);
    z1 = zintersect( cylh, mint, maxt, cylh->center + Vector3f(0,0,cylh->h), ray);

    float t;

    //t = select( t0 < t1, t0, t1 );
    if( t0 < t1 ){
        t = t0;
    }
    else{
        t = t1;
    }
#if 1
    //t = select( t < z0, t, z0 );
    if( ! (t < z0) ){
        t = z0;
    }

    //t = select( t < z1, t, z1 );
    if( ! (t < z1) ){
        t = z1;
    }
#endif

    if( t != INFINITY ){ // Hmmm... Should check mint maxt??
        optixReportIntersection(t, OPTIX_HIT_KIND_TRIANGLE_FRONT_FACE);
    }

}


extern "C" __global__ void __closesthit__cylhollow() {
    unsigned int launch_index = calculate_launch_index();

    if (params.is_ray_test()) {
        params.out_hit[launch_index] = true;
    } else {
        const OptixHitGroupData *sbt_data = (OptixHitGroupData *) optixGetSbtDataPointer();
        OptixCylHollowData *cylh = (OptixCylHollowData *)sbt_data->data;

        // Ray in instance-space
        Ray3f ray = get_ray();

        bool is_zplane;
        float t = ray.maxt; // from cylinder.h
                            // This appears to be the point

        float zcomp = ray( ray.maxt )[2];

        is_zplane = (zcomp == cylh->h_low) || (zcomp == cylh->h_hi);

        /*
        si.sh_frame.n = select( is_zplane,
                                select( zcomp == (scalar_t<Double>) m_h_low,
                                        normalize( Double3( 0,0,-1 ) ),
                                        normalize( Double3( 0,0, 1 ) )),
                                select( h > (scalar_t<Double>) (m_radius0 + m_thickness/2.0f),
                                        normalize(d * Double3( 1, 1,0)),
                                        normalize(d * Double3(-1,-1,0)))
                              );
        */
        Vector3f ns;

        if( is_zplane ){
            if( (zcomp == cylh->h_low) ){
                ns = normalize( Vector3f( 0,0,-1 ) );
            }
            else{
                ns = normalize( Vector3f( 0,0,1 ) );
            }
        }
        else{
            
            Vector3f d = Vector3f(1,1,0) * (ray( t ) - cylh->center);
            
            float h = sqrt( dot( d, d ) );

            if( (h > (cylh->radius0 + cylh->thickness/2.0f) ) ){
                ns = normalize( d * Vector3f( 1,1,0 ) );
            }
            else{
                ns = normalize( d * Vector3f( -1,-1,0) );
            }
        }

        Vector3f ng = ns;

        Vector3f p = ray( t );

        //si.p = ray(pi.t);


#if 0


        // Early return for ray_intersect_preliminary call
        if (params.is_ray_intersect_preliminary()) {
            write_output_pi_params(params, launch_index, sbt_data->shape_ptr, 0, Vector2f(), ray.maxt);
            return;
        }

        /* Compute and store information describing the intersection. This is
           very similar to CylHollow::compute_surface_interaction() */

        //Vector3f ns = normalize(ray(ray.maxt) - sphere->center);

        //if (sphere->flip_normals)
        //    ns = -ns;

        //Vector3f ng = ns;

        // Re-project onto the sphere to improve accuracy
        //Vector3f p = fmaf(sphere->radius, ns, sphere->center);

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
#else
        Vector2f uv;
        Vector3f dp_du, dp_dv;
        Vector3f dn_du, dn_dv;

        uv = Vector2f(0);
        dp_du = Vector3f(0);
        dp_dv = Vector3f(0);
        dn_du = Vector3f(0);
        dn_dv = Vector3f(0);
#endif

        write_output_si_params(params, launch_index, sbt_data->shape_ptr,
                               0, p, uv, ns, ng, dp_du, dp_dv, dn_du, dn_dv, ray.maxt);
    }
}
#endif
