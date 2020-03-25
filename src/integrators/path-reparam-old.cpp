#include <random>
#include <enoki/stl.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>

#include <enoki/transform.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/core/frame.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/mmap.h>
#include <list>
#include "path-reparam-utils.h"

#define REUSE_CAMERA_RAYS 1

NAMESPACE_BEGIN(mitsuba)

/**!

.. _integrator-pathreparam:

Differentiable path tracer (:monosp:`pathreparam`)
-------------------------------------------

.. pluginparameters::

 * - max_depth
   - |int|
   - Specifies the longest path depth in the generated output image (where -1 corresponds to
     :math:`\infty`). A value of 1 will only render directly visible light sources. 2 will lead
     to single-bounce (direct-only) illumination, and so on. (Default: -1)
 * - rr_depth
   - |int|
   - Specifies the minimum path depth, after which the implementation will start to use the
     *russian roulette* path termination criterion. (Default: 5)
 * - dc_light_samples
   - |int|
   - Specifies the number of samples for reparameterizing direct lighting integrals. (Default: 4)
 * - dc_bsdf_samples
   - |int|
   - Specifies the number of samples for reparameterizing BSDFs integrals. (Default: 4)
 * - dc_cam_samples
   - |int|
   - Specifies the number of samples for reparameterizing pixel integrals. (Default: 4)
 * - conv_threshold
   - |float|
   - Specifies the BSDFs roughness threshold that activates convolutions. (Default: 0.15f)
 * - kappa_conv
   - |float|
   - Specifies the kappa parameter of von Mises-Fisher distributions for convolutions.
     (Default: 1000.f)
 * - use_convolution
   - |bool|
   - Enable convolution for rough BSDFs. (Default: yes, i.e. |true|)
 * - use_variance_reduction
   - |bool|
   - Enable variation reduction. (Default: yes, i.e. |true|)
 * - disable_gradient_diffuse
   - |bool|
   - Disable reparameterization for diffuse scattering. (Default: no, i.e. |false|)
 * - disable_gradient_bounce
   - |int|
   - Disable reparameterization after several scattering events. (Default: 10)

This integrator implements the reparameterization technique described in the
article "Reparameterizing discontinuous integrands for differentiable rendering".
It is based on the integrator :ref:`path <integrator-path>` and it applies 
reparameterizations for each rendering integral in order to account for discontinuities 
when pixel values are differentiated using GPU modes and the Python API.

This plugin should be used with the plugin :ref:`smootharea <emitter-smootharea>`,
which is similar to the plugin :ref:`area <emitter-area>` with smoothly 
decreasing radiant exitance at the borders of the area light geometry to
avoid discontinuities. Other light sources will lead to incorrect partial derivatives.
Large area lights also result in significant bias since the convolution technique
described in the paper is only applied to rough and diffuse BSDF integrals.

Another limitation of this implementation is memory usage on the GPU: automatic 
differentiation for an entire path tracer typically requires several GB of GPU 
memory. The rendering must sometimes be split into various rendering passes with 
small sample counts in order to fit into GPU memory. 

.. note:: This integrator does not handle participating media

 */

template <typename Float, typename Spectrum>
class DiffPathIntegrator : public MonteCarloIntegrator<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(MonteCarloIntegrator, m_max_depth, m_rr_depth)
    MTS_IMPORT_TYPES(Scene, Sampler, Emitter, EmitterPtr, BSDF, BSDFPtr)

    DiffPathIntegrator(const Properties &props) : Base(props) { 
        m_dc_light_samples = props.size_("dc_light_samples", 4);
        m_dc_bsdf_samples  = props.size_("dc_bsdf_samples",  4);
        m_dc_cam_samples   = props.size_("dc_cam_samples",   4);
        m_conv_threshold   = props.float_("conv_threshold",  0.15f);
        m_kappa_conv       = props.float_("kappa_conv",      1000.f);
        m_use_convolution        = props.bool_("use_convolution",        true);
        m_use_variance_reduction = props.bool_("use_variance_reduction", true);
        m_disable_gradient_diffuse = props.bool_("disable_gradient_diffuse", false);
        m_disable_gradient_bounce = props.size_("disable_gradient_bounce", 1000);

        Log(Debug, "Changes of variables in light integrals using %i samples", 
            m_dc_light_samples);
        Log(Debug, "Changes of variables in BSDFs integrals using %i samples", 
            m_dc_bsdf_samples);
        Log(Debug, "Changes of variables in pixel integrals using %i samples", 
            m_dc_cam_samples);
        Log(Debug, "Changes of variables using convolution if roughness > %f", 
            m_conv_threshold);
        Log(Debug, "Convolutions using kernel with kappa = %f",
            m_kappa_conv);
        Log(Debug, "Variance reduction %s", 
            m_use_variance_reduction ? "enabled" : "disabled");
        Log(Debug, "Convolutions %s", 
            m_use_convolution ? "enabled" : "disabled");
        Log(Debug, "Gradient of diffuse reflections %s", 
            m_disable_gradient_diffuse ? "disabled" : "enabled");
        Log(Debug, "Disable gradients after bounce %i", m_disable_gradient_bounce);
        Log(Debug, "Reusing camera samples is %s", 
            REUSE_CAMERA_RAYS ? "enabled" : "disabled");

    }

    std::pair<Spectrum, Mask> sample(const Scene *scene,
                                     Sampler *sampler,
                                     const RayDifferential3f &primary_ray_,
                                     Float * /* aovs */,
                                     Mask active_primary) const override {

        RayDifferential3f primary_ray = primary_ray_;

        /* Estimate kappa for the convolution of pixel integrals, based on ray 
           differentials. */
        
        Float angle = acos(min(dot(primary_ray.d_x, primary_ray.d), 
                               dot(primary_ray.d_y, primary_ray.d)));
        Float targetMeanCos = min(cos(angle*0.4f /*arbitrary*/), Float(1.f-1e-7f)); 

        /* The vMF distribution has an analytic expression for the mean cosine:
                            mean = 1 + 2/(exp(2*k)-1) - 1/k.
           For large values of kappa, 1-1/k is a precise approximation of this 
           function. It can be inverted to find k from the mean cosine. */

        Float kappa_camera = Float(1.f)/(Float(1.f) - targetMeanCos);

        size_t nb_pimary_rays = slices(primary_ray.d);

        Spectrum result(0.f);

        if constexpr (is_cuda_array_v<Float>) {
            /* ---------------- Convolution of pixel integrals ------------- */
            
            /* Detect discontinuities in a small vMF kernel around each ray. */

            std::vector<RayDifferential3f> rays;
            std::vector<SurfaceInteraction3f> sis;
            std::vector<Point3f> attached_pos;
            
            Frame<Float> frame_input = Frame<Float>(primary_ray.d);
    
            Vector3f vMFsample_0;
            Vector3f vMFsample_1;

            /* Sample the integrals and gather intersections */
            for (size_t cs = 0; cs < m_dc_cam_samples; cs ++) {
                Vector3f vMFsample_cs = warp::square_to_von_mises_fisher<Float>(
                    sampler->next_2d(active_primary), kappa_camera);
                Vector3f dirConv_cs = frame_input.to_world(vMFsample_cs);

                primary_ray.d = dirConv_cs;
                SurfaceInteraction3f si_cs = scene->ray_intersect(primary_ray, active_primary);
                si_cs.compute_differentiable_intersection(primary_ray);

                rays.push_back(RayDifferential(primary_ray));
                sis.push_back(si_cs);
                attached_pos.push_back(si_cs.p_attached());

                /* Keep two samples for creating pairs of paths. 
                   We choose the last samples since they have less 
                   chances of being used in the estimation of the
                   discontinuity. */
                if (cs == m_dc_cam_samples-2)
                    vMFsample_0 = vMFsample_cs;
                if (cs == m_dc_cam_samples-1)
                    vMFsample_1 = vMFsample_cs;
            }
            
            Point3f discontinuity = estimateDiscontinuity(rays, sis, attached_pos, active_primary);
            Vector3f discontinuity_dir = normalize(discontinuity - primary_ray.o);

            /* Create the differentiable rotation */
      
            Vector3f axis = cross<Vector3f>(detach(discontinuity_dir), discontinuity_dir);
            Float cosangle = dot(discontinuity_dir, detach(discontinuity_dir)); 
            Transform4f rotation = rotationFromAxisCosAngle(axis, cosangle);

            /* Tracks radiance scaling due to index of refraction changes */
            Float eta(1.f);
        
            /* MIS weight for intersected emitters (set by prev. iteration) */
            Float emission_weight(1.f);

            /* Make pairs of rays (reuse 2 samples) and apply rotation */
            Spectrum throughput(1.f);

#if !REUSE_CAMERA_RAYS
            /* Resample two rays. This tends to add bias on silhouettes. */
            vMFsample_0 = warp::square_to_von_mises_fisher<Float, Float>(
                                sampler->next_2d(active_primary), kappa_camera);
            vMFsample_1 = warp::square_to_von_mises_fisher<Float, Float>(
                                sampler->next_2d(active_primary), kappa_camera);
#endif

            Vector3f dirConv_0 = frame_input.to_world(vMFsample_0);      
            Vector3f dirConv_1 = frame_input.to_world(vMFsample_1);     

            Vector ray_dir_0 = rotation.transform_affine(detach(dirConv_0));
            Vector ray_dir_1 = rotation.transform_affine(detach(dirConv_1));

            Vector3f ray_d = concat3D<Vector3f>(ray_dir_0, ray_dir_1);
            Point3f ray_o = makePair3D<Point3f>(primary_ray.o);
            Wavelength ray_w = makePairWavelength<Float, Wavelength>(primary_ray.wavelengths);
            
            Mask active(true);
            set_slices(active, nb_pimary_rays*2);

            /* Recompute differentiable pdf */
            Ray3f ray = Ray3f(ray_o, ray_d, 0.0, ray_w);
            Float vMFpdf_diff_0 = warp::square_to_von_mises_fisher_pdf<Float, Float>(
                frame_input.to_local(ray_dir_0), kappa_camera);
            Float vMFpdf_diff_1 = warp::square_to_von_mises_fisher_pdf<Float, Float>(
                frame_input.to_local(ray_dir_1), kappa_camera);
            Float vMFpdf_diff = concatD<Float>(vMFpdf_diff_0, vMFpdf_diff_1);

            /* Apply differentiable weight and keep for variance reduction */
            throughput *= vMFpdf_diff/detach(vMFpdf_diff);
            Float current_weight = vMFpdf_diff/detach(vMFpdf_diff);

            /* ---------------------- First intersection ---------------------- */

            SurfaceInteraction3f si = scene->ray_intersect(ray, active);
            si.compute_differentiable_intersection(ray);

            Mask valid_ray_pair = si.is_valid();
            UInt32 indices = arange<UInt32>(nb_pimary_rays);
            Mask valid_ray = gather<Mask>(valid_ray_pair, indices)
                || gather<Mask>(valid_ray_pair, indices+nb_pimary_rays);

            EmitterPtr emitter = si.emitter(scene);


            for (size_t depth = 1;; ++depth) {
        
                /* ---------------- Intersection with emitters ---------------- */

                if (any_or<true>(neq(emitter, nullptr))) {
                    Spectrum emission(0.f);
                    emission[active] = emission_weight * throughput * emitter->eval(si, active);

                    Spectrum emission0 = gather<Spectrum>(emission, 
                        arange<UInt32>(nb_pimary_rays)); 
                    Spectrum emission1 = gather<Spectrum>(emission, 
                        arange<UInt32>(nb_pimary_rays)+nb_pimary_rays); 

                    Float weights0 = gather<Float>(current_weight, 
                        arange<UInt32>(nb_pimary_rays), Mask(true));
                    Float weights1 = gather<Float>(current_weight, 
                        arange<UInt32>(nb_pimary_rays)+nb_pimary_rays, Mask(true));

                    if (depth >= m_disable_gradient_bounce) {
                        result += detach(emission0)*0.5f;
                        result += detach(emission1)*0.5f;  
                    } else if (m_use_variance_reduction) {
                        /* Avoid numerical errors due to tiny weights */
                        weights0 = select(abs(weights0) < 0.00001f, Float(1.f), weights0);
                        weights1 = select(abs(weights1) < 0.00001f, Float(1.f), weights1);

                        /* Variance reduction, assumption that contribution = weight * constant */
                        result += (emission0 - emission1/weights1 * (weights0-detach(weights0)))*0.5f;
                        result += (emission1 - emission0/weights0 * (weights1-detach(weights1)))*0.5f;
                    } else {
                        result += emission0*0.5f;
                        result += emission1*0.5f;                    
                    }

                }

                active &= si.is_valid();
        
                /*  Russian roulette: try to keep path weights equal to one,
                    while accounting for the solid angle compression at refractive
                    index boundaries. Stop with at least some probability to avoid
                    getting stuck (e.g. due to total internal reflection) */
                if (int(depth) > m_rr_depth) {
                    Float q = min(hmax(throughput) * sqr(eta), .95f);

                    active &= sample1D(active, sampler) < q;

                    throughput *= rcp(q);
                }
        
                if (none(active) || (uint32_t) depth >= (uint32_t) m_max_depth)
                    break;
    

                /* --------------------- Emitter sampling --------------------- */

                BSDFContext ctx;
                BSDFPtr bsdf = si.bsdf(ray);
                Mask active_e = active && has_flag(bsdf->flags(), BSDFFlags::Smooth);
        
                if (likely(any_or<true>(active_e))) {

                    /* Sample the light integral at each active shading point. 
                       Several samples are used for estimating discontinuities
                       in light visibility. */

                    // auto [emitter, emitter_pdf] = scene->sample_emitter(
                    //     si, samplePair2D(active_e, sampler), active_e);

                    auto emitter = scene->emitters()[0];
                    Float emitter_pdf(1.f);
                    if (scene->emitters().size() > 1)
                        Log(Error, "Only one emitter is supported currently");

                    Point3f positionDiscontinuity(0.f);
                    UInt32 hits(0);

                    std::vector<DirectionSample3f> ds_ls;
                    std::vector<Spectrum> emitter_val_ls;
                    std::vector<Mask> is_hit_ls;

                    for (size_t ls = 0; ls < m_dc_light_samples; ls++) {

                        auto [ds, emitter_val] = emitter->sample_direction(si, samplePair2D(active_e, sampler), active_e);

                        ds_ls.push_back(ds);
                        emitter_val_ls.push_back(emitter_val);

                        Mask active_ls = active_e && neq(ds.pdf, 0.f);
                        if (any_or<true>(active_ls)) {
                            /* Look for masking for active rays with valid emitter samples */
                            Ray3f ray_ls(si.p, ds.d, math::RayEpsilon<Float> * (1.f + hmax(abs(si.p))),
                                       ds.dist * (1.f - math::ShadowEpsilon<Float>),
                                       si.time, si.wavelengths);

                            SurfaceInteraction3f si_ls = scene->ray_intersect(ray_ls, active_ls);
                            si_ls.compute_differentiable_intersection(ray_ls);

                            Mask ls_hit = neq(si_ls.shape, nullptr);
                            is_hit_ls.push_back(ls_hit);
                            positionDiscontinuity[ls_hit] += si_ls.p_attached();
                            hits = select(ls_hit, hits+1, hits);
                            emitter_val_ls[ls] = select(ls_hit, Spectrum(0.f), emitter_val_ls[ls]);
                        }
                    }

                    /* Compute differentiable rotations from emitter samples */

                    Mask hasHit = hits > 0.f;
                    if (likely(any_or<true>(hasHit))) {
                        positionDiscontinuity[hasHit] = positionDiscontinuity/hits;                    
                    }

                    Vector3f directionDiscontinuity(0.f);
                    Transform4f rotation_ls; 

                    if (likely(any_or<true>(hasHit))) {
                        directionDiscontinuity[hasHit] = normalize(positionDiscontinuity - si.p);    

                        Float cosangle_ls = dot(directionDiscontinuity, detach(directionDiscontinuity));
                        Vector3f axis_ls = cross<Vector3f>(detach(directionDiscontinuity), directionDiscontinuity);
                        rotation_ls = rotationFromAxisCosAngle(axis_ls, cosangle_ls);

                    }

                    std::vector<Spectrum> contribs_ls;
                    std::vector<Float> weights_ls;

                    /* Reuse all the emitter samples and compute differentiable contributions */

                    for (size_t ls = 0; ls < m_dc_light_samples; ls++) {

                        /* Replace direction sample by differentiable data */
                        if (likely(any_or<true>(hasHit))) {

                            /* Recompute direction */
                            ds_ls[ls].d[hasHit] = rotation_ls.transform_affine(detach(ds_ls[ls].d));          
                            
                            /* Recompute pdf */
                            Float pdf_ls_diff = emitter->pdf_direction(si, ds_ls[ls], hasHit);
                            
                            /* Recompute emitter_val, only if not occluded */
                            Mask visible_and_hit = hasHit && (!is_hit_ls[ls]);
                            if (likely(any_or<true>(visible_and_hit))) {

                                Ray3f ray_ls(si.p, ds_ls[ls].d, math::RayEpsilon<Float> * (1.f + hmax(abs(si.p))),
                                    ds_ls[ls].dist + 1,
                                    si.time, si.wavelengths);

                                SurfaceInteraction3f si_ls = scene->ray_intersect(ray_ls, visible_and_hit);
                                si_ls.compute_differentiable_intersection(ray_ls);

                                // The emitter_val is E * k / pdf = E.
                                // The eval gives E * k.
                                // We need to divide by the detached pdf.
                                Spectrum emitter_val_diff = emitter->eval(si_ls,
                                    visible_and_hit) / detach(ds_ls[ls].pdf);

                                emitter_val_ls[ls] = select(visible_and_hit, 
                                    emitter_val_diff,  emitter_val_ls[ls]);

                            }

                            // Used for MIS. pdf_ls_diff is the pdf of the attached direction.
                            // The pdf of the sample should be detached. The sample is following 
                            // the discontinuity, it does not depend on how the light pdf changes.
                            ds_ls[ls].pdf = select(hasHit, detach(ds_ls[ls].pdf), ds_ls[ls].pdf);

                            Float w(1.0);
                            w = select(hasHit, pdf_ls_diff/ds_ls[ls].pdf, 1.f);

                            weights_ls.push_back(w);

                        } else {
                            weights_ls.push_back(Float(1.f));
                        }


                        /* Compute contribution */

                        Mask active_c = active_e && neq(ds_ls[ls].pdf, 0.f);

                        /* Query the BSDF for that emitter-sampled direction */
                        Vector3f wo = si.to_local(ds_ls[ls].d);
                        Spectrum bsdf_val = bsdf->eval(ctx, si, wo, active_c);

                        /* Determine probability of having sampled that same
                           direction using BSDF sampling. */
                        Float bsdf_pdf = bsdf->pdf(ctx, si, wo, active_c);

                        Float mis = select(ds_ls[ls].delta, 1.f, mis_weight(ds_ls[ls].pdf*emitter_pdf, bsdf_pdf));

                        Spectrum contrib_ls = throughput * emitter_val_ls[ls] / emitter_pdf * bsdf_val * mis;
                        contribs_ls.push_back(contrib_ls);
                    }


                    /* Accumulate contributions and variance reduction (in pairs of paths) */

                    if (m_dc_light_samples > 1) {
                        Spectrum contrib(0.f);
                        Float sum_weights_ls(0.f);
                        for (size_t ls = 0; ls < m_dc_light_samples; ls++) {
                            contrib += contribs_ls[ls];
                            sum_weights_ls += weights_ls[ls];
                        }

                        if (m_use_variance_reduction) {
                            // This is baised, could use cross control variates as well.
                            contrib /= sum_weights_ls; 
                        } else {
                            contrib /= m_dc_light_samples;  
                        }                        

                        /*  Add the contribution of this light sample.
                            The weight is the current weight of the throughput. */
                        Spectrum emitterSampling(0.f);
                        emitterSampling[active_e] += contrib;

                        Spectrum emitterSampling0 = gather<Spectrum>(emitterSampling, arange<UInt32>(nb_pimary_rays)); 
                        Spectrum emitterSampling1 = gather<Spectrum>(emitterSampling, arange<UInt32>(nb_pimary_rays)+nb_pimary_rays); 

                        Float weights0 = gather<Float>(current_weight, arange<UInt32>(nb_pimary_rays), Mask(true));
                        Float weights1 = gather<Float>(current_weight, arange<UInt32>(nb_pimary_rays)+nb_pimary_rays, Mask(true));

                        /* Here the weights weights0 and weights1 come from previous
                           bsdf sampling, their gradients are uncorrelated to the sampled emissions
                           **of the other path** emitterSampling0 and emitterSampling1. */
                        if (depth >= m_disable_gradient_bounce) {
                            result += detach(emitterSampling0)*0.5f;
                            result += detach(emitterSampling1)*0.5f;    
                        } else if (m_use_variance_reduction) {
                            weights0 = select(abs(weights0) < 0.00001f, Float(1.f), weights0);
                            weights1 = select(abs(weights1) < 0.00001f, Float(1.f), weights1);
                            result += (emitterSampling0 - emitterSampling1/weights1  *(weights0-detach(weights0)))*0.5f;
                            result += (emitterSampling1 - emitterSampling0/weights0  *(weights1-detach(weights1)))*0.5f;
                        } else {
                            result += emitterSampling0*0.5f;
                            result += emitterSampling1*0.5f;                            
                        }

                    } else {
                        Throw("DiffPathIntegrator: m_dc_light_samples < 2 not implemented!");
                    }
                }

                /* ----------------------- BSDF sampling ---------------------- */

                Float componentSample = samplePair1D(active, sampler);

                auto [sample_main_bs, bsdf_val_main_bs] = bsdf->sample(ctx, si, componentSample,
                                                                       samplePair2D(active, sampler), active);

                Mask convolution = Mask(m_use_convolution) && active 
                    && sample_main_bs.sampled_roughness > m_conv_threshold;

                if (any(has_flag(sample_main_bs.sampled_type, BSDFFlags::Delta)))
                    Log(Error, "This pluggin does not support perfectly specular reflections"
                        " and transmissions. Rough materials should be used instead.");

                active &= sample_main_bs.pdf > 0.f;

                Frame<Float> frame_main(sample_main_bs.wo);
                std::vector<Vector3f> ds_bs;

                /* Compute directions to samples either from the bsdf or the 
                   convolution of the bsdf. Only the first one is 
                   used for the light paths. */
                for (size_t bs = 0; bs < m_dc_bsdf_samples; bs++) {

                    /* Convolution: sample a vmf lobe */
                    Vector3f sample_bs = warp::square_to_von_mises_fisher<Float>(sample2D(active, sampler), m_kappa_conv);
                    sample_bs = frame_main.to_world(sample_bs);

                    /* Otherwise: must be uncorrelated, but can sample the same component */
                    auto [sample_bs_noconv, bsdf_val_bs] = bsdf->sample(ctx, si, componentSample,
                                                                         sample2D(active, sampler), active);

                    sample_bs = select(convolution, sample_bs, sample_bs_noconv.wo);

                    ds_bs.push_back(sample_bs);

                }

                /* Sample all these rays for discontinuity estimation */
                std::vector<RayDifferential3f> rays_bs;
                std::vector<SurfaceInteraction3f> sis_bs;
                std::vector<Point3f> p_attached_bs;

                Mask use_sliding_bs(false);

                for (size_t bs = 0; bs < m_dc_bsdf_samples; bs++) {
                    Ray3f ray_bs = si.spawn_ray(si.to_world(ds_bs[bs]));
                    SurfaceInteraction3f si_bsdf_bs = scene->ray_intersect(ray_bs, active);
                    si_bsdf_bs.compute_differentiable_intersection(ray_bs);
                    rays_bs.push_back(ray_bs);
                    sis_bs.push_back(si_bsdf_bs);     
                    p_attached_bs.push_back(si_bsdf_bs.p_attached());     
                    // Set hasIt to true if find hit
                    use_sliding_bs = use_sliding_bs || (active && neq(si_bsdf_bs.shape, nullptr));          
                }

                if (m_disable_gradient_diffuse) {
                    use_sliding_bs &= !convolution;
                    current_weight = select(use_sliding_bs, current_weight, detach(current_weight));
                }

                Point3f discontinuity_bs = estimateDiscontinuity(rays_bs, sis_bs, p_attached_bs, active);

                Transform4f rotation_bs;
                if (likely(any_or<true>(use_sliding_bs))) {

                    Vector3f direction_diff = normalize(discontinuity_bs - si.p);
                    Point3f discontinuity_bs_detach = detach(discontinuity_bs);
                    Vector3f direction_detach = normalize(discontinuity_bs_detach-si.p);

                    Float cosangle_bs = dot(direction_diff, direction_detach);
                    Vector3f axis_bs = cross(direction_detach, direction_diff);

                    // This rotation is in world space
                    rotation_bs = rotationFromAxisCosAngle(axis_bs, cosangle_bs);
                }

                /* Initialize the BSDF sample from the initial sample, eta and 
                   sampled_type do not change since the same component is sampled. */
                BSDFSample3 sample_bs = sample_main_bs; 

                /* Reuse one direction sampled from either the BSDF or the convolution kernel
                   around the main direction. */
                sample_bs.wo = ds_bs[0]; // Reuse the first one, could be any of them


                /* Apply the differentiable rotation */
                if (likely(any_or<true>(use_sliding_bs))) {
                    // Warning, the direction must be detached such that it follows the discontinuities
                    // Warning, this rotation in world space, but wo is in local space
                    sample_bs.wo[use_sliding_bs] = si.to_local(rotation_bs.transform_affine(
                        si.to_world(detach(sample_bs.wo)))); 
                }

                /* Compute the differentiable BSDF value for the differentiable direction */
                Spectrum bsdf_value = bsdf->eval(ctx, si, sample_bs.wo, active);

                /* Copmpute the pdf of the convolution kernel for the selected direction */
                // Warning: need to transform to a frame centered around the Z axis
                Float pdf_conv_newDir = warp::square_to_von_mises_fisher_pdf<Float>(frame_main.to_local(sample_bs.wo), 
                                                                             m_kappa_conv);




                /* Multiply the BSDF value by the convolution kernel. Use a 
                   correction term for the convolution (otherwise less energy 
                   at grazing angles) */
                Float cosangle = sample_bs.wo.z();
                Float correction_factor = m_vmf_hemisphere.eval(m_kappa_conv, cosangle, convolution);

                bsdf_value = select(convolution, bsdf_value*pdf_conv_newDir/correction_factor, bsdf_value);

                /* Compute the value of default importance sampling pdf of the BSDF.
                   Used when convolution is disabled and for MIS */
                Float bsdf_pdf_default = bsdf->pdf(ctx, si, sample_bs.wo, active);

                /* The pdf should be:
                    - When not using changes of variables, the undetached pdf
                      because the sample are sampled from the standard pdf
                    - When using changes of variables and convolution,
                      the pdf of the main direction (not detached) times 
                      the detached pdf using for sampling the convolution kernel
                      (always detached pdf because the samples don't move wrt the rotating sampling pdf)
                    - When not using the convolution, detached pdf */
                Float bsdf_pdf = select(convolution, 
                                        sample_main_bs.pdf * pdf_conv_newDir, 
                                        bsdf_pdf_default);
                if (likely(any_or<true>(use_sliding_bs))) {
                    bsdf_pdf[use_sliding_bs] = select(convolution, 
                                                      sample_main_bs.pdf * detach(pdf_conv_newDir), 
                                                      detach(bsdf_pdf_default));
                }
                Spectrum bsdf_value_pdf = bsdf_value / bsdf_pdf;

                /* Compute weights for variance reduction
                   These weights should be:
                    - just 1 if no change of variable is used
                    - Weights whose expected gradient is 0 and value is
                      close to bsdf_value_pdf. */
                // TODO: these weights should be colors.

                Mask set_weights = use_sliding_bs && (bsdf_pdf > 0.001f);
                current_weight = select(set_weights &&  convolution, 
                    current_weight * detach(bsdf_value_pdf[0])*pdf_conv_newDir/detach(pdf_conv_newDir),
                    current_weight);
                current_weight = select(set_weights && !convolution, 
                    current_weight * detach(bsdf_value_pdf[0])*bsdf_pdf_default/detach(bsdf_pdf_default), 
                    current_weight);
                throughput *= bsdf_value_pdf;

                active &= any(neq(throughput, 0.f));


                if (none(active))
                    break;
    
                eta *= sample_bs.eta;

                /* Intersect the BSDF ray against the scene geometry */
                ray = si.spawn_ray(si.to_world(sample_bs.wo));
                SurfaceInteraction3f si_bsdf = scene->ray_intersect(ray, active);
                si_bsdf.compute_differentiable_intersection(ray);

                /* Determine probability of having sampled that same
                   direction using emitter sampling. */
                emitter = si_bsdf.emitter(scene, active);
                DirectionSample3f ds(si_bsdf, si);
                ds.object = emitter;
    
                if (any_or<true>(neq(emitter, nullptr))) {
                    Float emitter_pdf =
                        select(has_flag(sample_bs.sampled_type, BSDFFlags::Delta), 0.f,
                               scene->pdf_emitter_direction(si, ds, active));
    
                    /* Always use the standard importance sampling pdf of the BSDF,
                       since this is the pdf used for MIS weights when sampling emitters. */
                    emission_weight = mis_weight(bsdf_pdf_default, emitter_pdf);
                }
    
                si = std::move(si_bsdf);
            }

            return {result, valid_ray};

        } else {
            Throw("DiffPathIntegrator: currently this integrator must be run on the GPU.");
            return {Spectrum(0.f), Mask(false)};
        }

    }

    //! @}
    // =============================================================

    std::string to_string() const override {
        return tfm::format("DiffPathIntegrator[\n"
            "  max_depth = %i,\n"
            "  rr_depth = %i\n"
            "]", m_max_depth, m_rr_depth);
    }

    MTS_DECLARE_CLASS()

protected:


    template <typename Value> Value mis_weight(Value pdf_a, Value pdf_b) const {
        pdf_a *= pdf_a;
        pdf_b *= pdf_b;
        return select(pdf_a > 0.f, pdf_a / (pdf_a + pdf_b), Value(0.f));
    };
    
    mitsuba::Transform<Vector4f> rotationFromAxisCosAngle(Vector3f axis, Float cosangle) const {
        Float ax = axis.x(), ay = axis.y(), az = axis.z();
        Float axy = ax*ay, axz = ax*az, ayz = ay*az;

        Matrix3f ux( Float(0), -az,        ay,
                    az,        Float(0), -ax,
                   -ay,        ax,        Float(0));

        Matrix3f uu(sqr(ax), axy,     axz,
                   axy,     sqr(ay), ayz,
                   axz,     ayz,     sqr(az));

        Matrix3f R = identity<Matrix3f>() * cosangle + ux + rcp(1+cosangle) * uu;

        return mitsuba::Transform<Vector4f>(Matrix4f(R));
    };

    Point3f estimateDiscontinuity(const std::vector<RayDifferential3f> &rays, 
                                  const std::vector<SurfaceInteraction3f> &sis,
                                  const std::vector<Point3f> &p_attached_sis,
                                  const Mask &/*mask*/) const {

        using Matrix        = enoki::Matrix<Float, 3>;
        
        unsigned int nbSamples = rays.size();

        if (rays.size() < 2 || rays.size() != sis.size())
            Throw("DiffPathIntegrator::estimateDiscontinuity: invalid number of samples for discontinuity estimation");

        UInt32 is_ray1_hit_uint = select(neq(sis[1].shape, nullptr), UInt32(1), UInt32(0));
        Point3f ray1_p_attached = p_attached_sis[1];
        Vector3f ray1_n = sis[1].n;
        Vector3f ray1_d = rays[1].d;

        for (unsigned int i = 2; i < nbSamples; i++) {
            Mask diff = neq(sis[0].shape, sis[i].shape);
            Mask i_hit = neq(sis[i].shape, nullptr);
            is_ray1_hit_uint = select(diff, select(i_hit, UInt32(1), UInt32(0)), is_ray1_hit_uint);
            ray1_p_attached   = select(diff, p_attached_sis[i], ray1_p_attached);
            ray1_n            = select(diff, sis[i].n, ray1_n);
            ray1_d            = select(diff, rays[i].d, ray1_d);
        }

        Mask is_ray1_hit = is_ray1_hit_uint > 0;

        /* Guess occlusion for pairs of samples */

        Point3f res(0.0);

        // if only one hit: return this hit
        Mask only_hit_0 = neq(sis[0].shape, nullptr) && !is_ray1_hit;
        res[only_hit_0] = p_attached_sis[0];

        Mask only_hit_1 = is_ray1_hit && eq(sis[0].shape, nullptr);
        res[only_hit_1] = ray1_p_attached;

        Mask hasTwoHits = neq(sis[0].shape, nullptr) && is_ray1_hit;

        if (any_or<true>(hasTwoHits)) {

            /* Compute occlusion between planes and hitpoints: sign of 
               dot(normal, hitpoint - hitpoint). Test if the origin of the rays 
               is on the same side as the other hit. */
            Float occPlane0 = dot(sis[0].n, ray1_p_attached - p_attached_sis[0])*dot(sis[0].n, rays[0].o - p_attached_sis[0]);
            Float occPlane1 = dot(ray1_n,   p_attached_sis[0] - ray1_p_attached)*dot(sis[0].n, rays[0].o - p_attached_sis[0]);

            Mask plane0Occludes1 = hasTwoHits && (occPlane0 < 0.f);
            Mask plane1Occludes0 = hasTwoHits && (occPlane1 < 0.f);

            Mask simpleOccluder0 = plane0Occludes1 && !plane1Occludes0;
            Mask simpleOccluder1 = plane1Occludes0 && !plane0Occludes1;
            Mask planeIntersection = hasTwoHits  && !simpleOccluder1 && !simpleOccluder0;

            if (any_or<true>(simpleOccluder0)) {
                res[simpleOccluder0] = p_attached_sis[0];
            }

            if (any_or<true>(simpleOccluder1)) {
                res[simpleOccluder1] = ray1_p_attached;
            }

            Mask sameNormals = planeIntersection  && abs(dot(sis[0].n, ray1_n)) > 0.99f;
            planeIntersection &= !sameNormals;

            if (any_or<true>(sameNormals)) {
                res[sameNormals] = p_attached_sis[0];
            }

            if (any_or<true>(planeIntersection)) {

#if 1
                /* Compute the intersection between 3 planes: 
                   2 planes defined by the ray intersections and 
                   the normals at these points, and 1 plane containing
                   the ray directions. */

                Vector3f N0 = sis[0].n;
                Vector3f N1 = ray1_n;
                Vector3f P0 = p_attached_sis[0];
                Vector3f P1 = ray1_p_attached;

                /* Normal of the third plane, defined using 
                   attached positions (this prevents bad correlations
                   between the displacement of the intersection and
                   the sampled positions) */

                Vector3f N = cross(P0 - rays[0].o, P1 - rays[0].o);
                Float norm_N = norm(N);

                /* Set a default intersection if the problem is ill-defined */
                res[planeIntersection] = p_attached_sis[0];

                Mask invertible = planeIntersection && norm_N > 0.001f;

                Matrix A = Matrix::from_rows(N0, N1, N);
                Float b0 =  dot(P0, N0);
                Float b1 =  dot(P1, N1);
                Float b2 =  dot(rays[0].o, N);
                Vector3f B(b0, b1, b2);
                Matrix invA = enoki::inverse(A);
                res[invertible] = invA*B;
#else       
                /* Simply choose one of the intersections.
                   This is a good strategy in many situations. */
                res[planeIntersection] = p_attached_sis[0];
#endif
            }
        }

        return res;

    }

private:
    size_t m_disable_gradient_bounce;
    size_t m_dc_light_samples;
    size_t m_dc_bsdf_samples;
    size_t m_dc_cam_samples;
    float m_conv_threshold;
    float m_kappa_conv;
    bool m_use_variance_reduction;
    bool m_use_convolution;
    bool m_disable_gradient_diffuse;

    VMFHemisphereIntegral<Float> m_vmf_hemisphere;

};

MTS_IMPLEMENT_CLASS_VARIANT(DiffPathIntegrator, MonteCarloIntegrator);
MTS_EXPORT_PLUGIN(DiffPathIntegrator, "Differentiable Path Tracer integrator");
NAMESPACE_END(mitsuba)
