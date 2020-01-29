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
--------------------------------------------------

.. pluginparameters::

 * - max_depth
   - |int|
   - Specifies the longest path depth in the generated output image (where -1
     corresponds to :math:`\infty`). A value of 1 will only render directly
     visible light sources. 2 will lead to single-bounce (direct-only)
     illumination, and so on. (Default: -1)
 * - rr_depth
   - |int|
   - Specifies the minimum path depth, after which the implementation will
     start to use the *russian roulette* path termination criterion. (Default: 5)
 * - dc_light_samples
   - |int|
   - Specifies the number of samples for reparameterizing direct lighting
     integrals. (Default: 4)
 * - dc_bsdf_samples
   - |int|
   - Specifies the number of samples for reparameterizing BSDFs integrals.
     (Default: 4)
 * - dc_cam_samples
   - |int|
   - Specifies the number of samples for reparameterizing pixel integrals.
     (Default: 4)
 * - conv_threshold
   - |float|
   - Specifies the BSDFs roughness threshold that activates convolutions.
     (Default: 0.15f)
 * - use_convolution
   - |bool|
   - Enable convolution for rough BSDFs. (Default: yes, i.e. |true|)
 * - kappa_conv
   - |float|
   - Specifies the kappa parameter of von Mises-Fisher distributions for BSDFs
     convolutions. (Default: 1000.f)
 * - use_convolution_envmap
   - |bool|
   - Enable convolution for environment maps. (Default: yes, i.e. |true|)
 * - kappa_conv_envmap
   - |float|
   - Specifies the kappa parameter of von Mises-Fisher distributions for
     environment map convolutions. (Default: 1000.f)
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
`article <https://rgl.epfl.ch/publications/Loubet2019Reparameterizing>`_
"Reparameterizing discontinuous integrands for differentiable rendering".
It is based on the integrator :ref:`path <integrator-path>` and it applies
reparameterizations to rendering integrals in order to account for discontinuities
when pixel values are differentiated using GPU modes and the Python API.

This plugin supports environment maps and area lights with the plugin
:ref:`smootharea <emitter-smootharea>`, which is similar to the plugin
:ref:`area <emitter-area>` with smoothly decreasing radiant exitance at the
borders of the area light geometry to avoid discontinuities. The area light
geometry should be flat and it should have valid uv coordinates (see
:ref:`smootharea <emitter-smootharea>` for details). Other light
sources will lead to incorrect partial derivatives. Large area lights also
result in significant bias since the convolution technique described in the
paper is only applied to environment maps and rough/diffuse BSDF integrals.

Another limitation of this implementation is memory usage on the GPU: automatic
differentiation for an entire path tracer typically requires several GB of GPU
memory. The rendering must sometimes be split into various rendering passes with
small sample counts in order to fit into GPU memory.

.. note:: This integrator does not handle participating media

 */

template <typename Float, typename Spectrum>
class PathReparamIntegrator : public MonteCarloIntegrator<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(MonteCarloIntegrator, m_max_depth, m_rr_depth)
    MTS_IMPORT_TYPES(Scene, Sampler, Medium, Emitter, EmitterPtr, BSDF, BSDFPtr)

    PathReparamIntegrator(const Properties &props) : Base(props) {
        m_dc_light_samples = props.size_("dc_light_samples", 4);
        m_dc_bsdf_samples  = props.size_("dc_bsdf_samples",  4);
        m_dc_cam_samples   = props.size_("dc_cam_samples",   4);
        m_conv_threshold   = props.float_("conv_threshold",  0.15f);
        m_kappa_conv       = props.float_("kappa_conv",      1000.f);
        m_kappa_conv_envmap      = props.float_("kappa_conv_envmap",     100000.f);
        m_use_convolution        = props.bool_("use_convolution",        true);
        m_use_convolution_envmap = props.bool_("use_convolution_envmap", true);
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
        Log(Info, "Convolutions for envmap %s",
            m_use_convolution_envmap ? "enabled" : "disabled");
        Log(Debug, "Gradient of diffuse reflections %s",
            m_disable_gradient_diffuse ? "disabled" : "enabled");
        Log(Debug, "Disable gradients after bounce %i", m_disable_gradient_bounce);
        Log(Debug, "Reusing camera samples is %s",
            REUSE_CAMERA_RAYS ? "enabled" : "disabled");
    }

    std::pair<Spectrum, Mask> sample(const Scene *scene,
                                     Sampler *sampler,
                                     const RayDifferential3f &primary_ray_,
                                     const Medium * /* medium */,
                                     Float * /* aovs */,
                                     Mask active_primary) const override {

        RayDifferential3f primary_ray = primary_ray_;

        // Estimate kappa for the convolution of pixel integrals, based on ray
        // differentials.
        Float angle = acos(min(dot(primary_ray.d_x, primary_ray.d),
                               dot(primary_ray.d_y, primary_ray.d)));
        Float target_mean_cos =
            min(cos(angle * 0.4f /*arbitrary*/), Float(1.f - 1e-7f));

        // The vMF distribution has an analytic expression for the mean cosine:
        //                  mean = 1 + 2/(exp(2*k)-1) - 1/k.
        // For large values of kappa, 1-1/k is a precise approximation of this
        // function. It can be inverted to find k from the mean cosine.
        Float kappa_camera = Float(1.f) / (Float(1.f) - target_mean_cos);

        const size_t nb_pimary_rays = slices(primary_ray.d);
        const UInt32 arange_indices = arange<UInt32>(nb_pimary_rays);

        Spectrum result(0.f);

        if constexpr (is_cuda_array_v<Float>) {

            // ---------------- Convolution of pixel integrals -------------

            // Detect discontinuities in a small vMF kernel around each ray.

            std::vector<RayDifferential3f> rays(m_dc_cam_samples);
            std::vector<SurfaceInteraction3f> sis(m_dc_cam_samples);

            Frame<Float> frame_input = Frame<Float>(primary_ray.d);

            Vector3f dir_conv_0, dir_conv_1;

            // Sample the integrals and gather intersections
            for (size_t cs = 0; cs < m_dc_cam_samples; cs++) {
                Vector3f vMF_sample_cs = warp::square_to_von_mises_fisher<Float>(
                    sampler->next_2d(active_primary), kappa_camera);
                Vector3f dir_conv_cs = frame_input.to_world(vMF_sample_cs);

                primary_ray.d = dir_conv_cs;
                sis[cs] = scene->ray_intersect(primary_ray, HitComputeMode::Least, active_primary);
                sis[cs].compute_differentiable_shape_position(active_primary);

                rays[cs] = RayDifferential(primary_ray);

                // Keep two directions for creating pairs of paths.
                // We choose the last samples since they have less
                // chances of being used in the estimation of the
                // discontinuity.
                if (cs == m_dc_cam_samples - 2)
                    dir_conv_0 = dir_conv_cs;
                if (cs == m_dc_cam_samples - 1)
                    dir_conv_1 = dir_conv_cs;
            }

            Point3f discontinuity = estimate_discontinuity(rays, sis, active_primary);
            Vector3f discontinuity_dir = normalize(discontinuity - primary_ray.o);

            // The following rotation seems to be the identity transformation, but it actually
            // changes the partial derivatives.

            // Create the differentiable rotation
            Vector3f axis = cross<Vector3f>(detach(discontinuity_dir), discontinuity_dir);
            Float cosangle = dot(discontinuity_dir, detach(discontinuity_dir));
            Transform4f rotation = rotation_from_axis_cosangle(axis, cosangle);

            // Tracks radiance scaling due to index of refraction changes
            Float eta(1.f);

            // MIS weight for intersected emitters (set by prev. iteration)
            Float emission_weight(1.f);

            // Make pairs of rays (reuse 2 samples) and apply rotation
            Spectrum throughput(1.f);

#if !REUSE_CAMERA_RAYS
            // Resample two rays. This tends to add bias on silhouettes.
            dir_conv_0 = frame_input.to_world(
                warp::square_to_von_mises_fisher<Float, Float>(
                    sampler->next_2d(active_primary), kappa_camera));
            dir_conv_1 = frame_input.to_world(
                warp::square_to_von_mises_fisher<Float, Float>(
                    sampler->next_2d(active_primary), kappa_camera));
#endif

            // NOTE: here we detach because the rays will be passed to Optix, no need for autodiff
            Vector ray_d_0 = rotation.transform_affine(detach(dir_conv_0));
            Vector ray_d_1 = rotation.transform_affine(detach(dir_conv_1));

            Vector3f ray_d   = concatD(ray_d_0, ray_d_1);
            Point3f ray_o    = makePairD<Point3f>(primary_ray.o);
            Wavelength ray_w = makePairD(primary_ray.wavelengths);

            Ray3f ray = Ray3f(ray_o, ray_d, 0.0, ray_w);

            Mask active(true);
            set_slices(active, nb_pimary_rays * 2);

            // Recompute differentiable pdf
            Float vMF_pdf_diff_0 = warp::square_to_von_mises_fisher_pdf<Float, Float>(
                frame_input.to_local(ray_d_0), kappa_camera);
            Float vMF_pdf_diff_1 = warp::square_to_von_mises_fisher_pdf<Float, Float>(
                frame_input.to_local(ray_d_1), kappa_camera);
            Float vMF_pdf_diff = concatD<Float>(vMF_pdf_diff_0, vMF_pdf_diff_1);

            // Apply differentiable weight and keep for variance reduction
            throughput *= vMF_pdf_diff / detach(vMF_pdf_diff); // NOTE: detach here so we only divide the gradient by the pdf
            Float current_weight = vMF_pdf_diff / detach(vMF_pdf_diff);

            // ---------------------- First intersection ----------------------

            auto si = scene->ray_intersect(ray, HitComputeMode::Differentiable, active);

            Mask valid_ray_pair = si.is_valid();

            Mask valid_ray =
                gather<Mask>(valid_ray_pair, arange_indices) ||
                gather<Mask>(valid_ray_pair, arange_indices + nb_pimary_rays);

            EmitterPtr emitter = si.emitter(scene);

            for (size_t depth = 1;; ++depth) {

                // ---------------- Intersection with emitters ----------------

                Spectrum emission(0.f);
                emission[active] = emission_weight * throughput * emitter->eval(si, active);

                Spectrum emission_0 = gather<Spectrum>(emission, arange_indices);
                Spectrum emission_1 = gather<Spectrum>(emission, arange_indices + nb_pimary_rays);

                Float weights_0 = gather<Float>(current_weight, arange_indices);
                Float weights_1 = gather<Float>(current_weight, arange_indices + nb_pimary_rays);

                if (depth >= m_disable_gradient_bounce) {
                    result += detach(emission_0) * 0.5f; // NOTE: detach so nothing is added to the gradient
                    result += detach(emission_1) * 0.5f;
                } else if (m_use_variance_reduction) {
                    // Avoid numerical errors due to tiny weights
                    weights_0 = select(abs(weights_0) < 0.00001f, Float(1.f), weights_0);
                    weights_1 = select(abs(weights_1) < 0.00001f, Float(1.f), weights_1);

                    // Variance reduction, assumption that contribution = weight * constant
                    result += (emission_0 - emission_1 / weights_1 * (weights_0 - detach(weights_0))) * 0.5f; // NOTE: detach here so to only add `e_1/w_1*w_0` to the gradient (only try to reduce the variance of the gradient)
                    result += (emission_1 - emission_0 / weights_0 * (weights_1 - detach(weights_1))) * 0.5f;
                } else {
                    result += emission_0 * 0.5f;
                    result += emission_1 * 0.5f;
                }

                active &= si.is_valid();

                // Russian roulette: try to keep path weights equal to one,
                // while accounting for the solid angle compression at refractive
                // index boundaries. Stop with at least some probability to avoid
                // getting stuck (e.g. due to total internal reflection)
                if (int(depth) > m_rr_depth) {
                    Float q = min(hmax(throughput) * sqr(eta), .95f);
                    active &= sample1D(active, sampler) < q;
                    throughput *= rcp(q);
                }

                if (none(active) || (uint32_t) depth >= (uint32_t) m_max_depth)
                    break;

                // --------------------- Emitter sampling ---------------------

                BSDFContext ctx;
                BSDFPtr bsdf = si.bsdf(ray);
                Mask active_e = active && has_flag(bsdf->flags(), BSDFFlags::Smooth);

                // Sample the light integral at each active shading point.
                // Several samples are used for estimating discontinuities
                // in light visibility.
                auto [emitter_ls, emitter_pdf] = scene->sample_emitter(
                    si, samplePair1D(active_e, sampler), active_e);

                Mask is_envmap = emitter_ls->is_environment() && active_e;

                Point3f position_discontinuity(0.f);
                UInt32 hits(0);

                std::vector<DirectionSample3f> ds_ls(m_dc_light_samples);
                std::vector<Spectrum> emitter_val_ls(m_dc_light_samples);
                std::vector<Mask> is_occluded_ls(m_dc_light_samples);

                auto ds_ls_main = emitter_ls->sample_direction(si, samplePair2D(active_e, sampler), active_e).first;
                Frame<Float> frame_main_ls(ds_ls_main.d);

                for (size_t ls = 0; ls < m_dc_light_samples; ls++) {
                    std::tie(ds_ls[ls], emitter_val_ls[ls]) =
                        emitter_ls->sample_direction(
                            si, samplePair2D(active_e, sampler), active_e);

                    if (m_use_convolution_envmap) {
                        Vector3f sample_ls =
                            warp::square_to_von_mises_fisher<Float>(
                                sample2D(active_e, sampler),
                                m_kappa_conv_envmap);

                        // Update with the pdf of the convolution kernel
                        ds_ls[ls].pdf[is_envmap] = warp::square_to_von_mises_fisher_pdf<Float>(
                            sample_ls, m_kappa_conv_envmap);
                        sample_ls = frame_main_ls.to_world(sample_ls);
                        ds_ls[ls].d[is_envmap] = sample_ls;
                    }

                    Mask active_ls = active_e && neq(ds_ls[ls].pdf, 0.f);

                    // Check masking for active rays
                    Ray3f ray_ls(si.p, ds_ls[ls].d, math::RayEpsilon<Float> * (1.f + hmax(abs(si.p))),
                                ds_ls[ls].dist * (1.f - math::ShadowEpsilon<Float>),
                                si.time, si.wavelengths);
                    ray_ls.maxt[is_envmap] = math::Infinity<Float>;

                    auto si_ls = scene->ray_intersect(ray_ls, HitComputeMode::Least, active_ls);
                    si_ls.compute_differentiable_shape_position(active_ls);

                    is_occluded_ls[ls] = neq(si_ls.shape, nullptr);
                    position_discontinuity[is_occluded_ls[ls]] += si_ls.p;
                    hits = select(is_occluded_ls[ls], hits + 1, hits);

                    if (m_use_convolution_envmap) {
                        // The contribution is radiance * kernel / ds_ls_main.pdf / kernel (pdf)
                        emitter_val_ls[ls][is_envmap] = emitter_ls->eval(si_ls, is_envmap) / ds_ls_main.pdf;
                    }

                    // The contribution is 0 when the light is not visible
                    emitter_val_ls[ls][is_occluded_ls[ls]] = Spectrum(0.f);
                }

                // Compute differentiable rotations from emitter samples

                Mask use_reparam = hits > 0.f;
                position_discontinuity[use_reparam] = position_discontinuity / hits;

                Vector3f direction_discontinuity(0.f);
                direction_discontinuity[use_reparam] = normalize(position_discontinuity - si.p);
                Vector3f direction_discontinuity_detach = detach(direction_discontinuity);

                // TODO: maybe should use same logic as in BSDF sampling (detach in normalize())
                // Vector3f direction_discontinuity_detach = normalize(detach(position_discontinuity) - si.p);

                Vector3f axis_ls = cross<Vector3f>(direction_discontinuity_detach, direction_discontinuity);
                Float cosangle_ls = dot(direction_discontinuity, direction_discontinuity_detach);
                Transform4f rotation_ls = rotation_from_axis_cosangle(axis_ls, cosangle_ls);

                std::vector<Spectrum> contribs_ls(m_dc_light_samples);

                // Reuse all the emitter samples and compute differentiable contributions

                for (size_t ls = 0; ls < m_dc_light_samples; ls++) {

                    // Recompute direction
                    ds_ls[ls].d[use_reparam] = rotation_ls.transform_affine(detach(ds_ls[ls].d));

                    if (m_use_convolution_envmap) {
                        // Recompute the value of convolution kernel
                        ds_ls[ls].pdf[use_reparam && is_envmap] = warp::square_to_von_mises_fisher_pdf<Float>(
                            frame_main_ls.to_local(ds_ls[ls].d), m_kappa_conv_envmap);
                    }

                    // Recompute the contribution when a reparameterization is used
                    Mask visible_and_hit = use_reparam && (!is_occluded_ls[ls]);

                    Ray3f ray_ls(si.p, ds_ls[ls].d,
                                 math::RayEpsilon<Float> * (1.f + hmax(abs(si.p))),
                                 ds_ls[ls].dist + 1.f,
                                 si.time, si.wavelengths);

                    auto si_ls = scene->ray_intersect(
                        ray_ls, HitComputeMode::Differentiable,
                        visible_and_hit);

                    Spectrum e_val_reparam = emitter_ls->eval(si_ls, visible_and_hit) / detach(ds_ls[ls].pdf);

                    if (m_use_convolution_envmap) {
                        e_val_reparam[visible_and_hit && is_envmap] *= ds_ls[ls].pdf / ds_ls_main.pdf;
                    }

                    emitter_val_ls[ls][visible_and_hit] = e_val_reparam;

                    if (m_use_convolution_envmap) {
                        // Update emitter pdf for MIS
                        Float pdf_emitter = emitter_ls->pdf_direction(si, ds_ls[ls], is_envmap);
                        ds_ls[ls].pdf[is_envmap] = detach(pdf_emitter);
                    }

                    // Compute contribution

                    Mask active_c = active_e && neq(ds_ls[ls].pdf, 0.f);

                    // Query the BSDF for that emitter-sampled direction
                    Vector3f wo = si.to_local(ds_ls[ls].d);
                    Spectrum bsdf_val = bsdf->eval(ctx, si, wo, active_c);

                    // Determine probability of having sampled that same
                    // direction using BSDF sampling.
                    Float bsdf_pdf = bsdf->pdf(ctx, si, wo, active_c);

                    Float mis = select(ds_ls[ls].delta, 1.f, mis_weight(ds_ls[ls].pdf * emitter_pdf, bsdf_pdf));

                    contribs_ls[ls] = throughput * emitter_val_ls[ls] / emitter_pdf * bsdf_val * mis;
                }

                // Accumulate contributions and variance reduction (in pairs of paths)
                if (m_dc_light_samples > 1) {
                    Spectrum contrib(0.f);
                    for (size_t ls = 0; ls < m_dc_light_samples; ls++) {
                        contrib += contribs_ls[ls];
                    }

                    contrib /= m_dc_light_samples;

                    //  Add the contribution of this light sample.
                    //  The weight is the current weight of the throughput.
                    Spectrum emitter_sampling(0.f);
                    emitter_sampling[active_e] += contrib;

                    Spectrum emitter_sampling_0 = gather<Spectrum>(emitter_sampling, arange_indices);
                    Spectrum emitter_sampling_1 = gather<Spectrum>(emitter_sampling, arange_indices + nb_pimary_rays);

                    Float weights_0 = gather<Float>(current_weight, arange_indices);
                    Float weights_1 = gather<Float>(current_weight, arange_indices + nb_pimary_rays);

                    // Here the weights weights_0 and weights_1 come from previous
                    // bsdf sampling, their gradients are uncorrelated to the sampled emissions
                    // **of the other path** emitter_sampling_0 and emitter_sampling_1.
                    if (depth >= m_disable_gradient_bounce) {
                        result += detach(emitter_sampling_0) * 0.5f;
                        result += detach(emitter_sampling_1) * 0.5f;
                    } else if (m_use_variance_reduction) {
                        weights_0 = select(abs(weights_0) < 0.00001f, Float(1.f), weights_0);
                        weights_1 = select(abs(weights_1) < 0.00001f, Float(1.f), weights_1);

                        result += (emitter_sampling_0 - emitter_sampling_1 / weights_1 * (weights_0 - detach(weights_0))) * 0.5f;
                        result += (emitter_sampling_1 - emitter_sampling_0 / weights_0 * (weights_1 - detach(weights_1))) * 0.5f;
                    } else {
                        result += emitter_sampling_0 * 0.5f;
                        result += emitter_sampling_1 * 0.5f;
                    }
                } else {
                    Throw("PathReparamIntegrator: m_dc_light_samples < 2 not implemented!");
                }

                // ----------------------- BSDF sampling ----------------------

                Float component_sample = samplePair1D(active, sampler);

                auto sample_main_bs = bsdf->sample(ctx, si, component_sample, samplePair2D(active, sampler), active).first;

                active &= sample_main_bs.pdf > 0.f;

                // TODO: BSDFs should fill the `sampled_roughness` field
                Mask convolution = Mask(m_use_convolution) && active
                    && sample_main_bs.sampled_roughness > m_conv_threshold;

                if (any(has_flag(sample_main_bs.sampled_type, BSDFFlags::Delta)))
                    Log(Error, "This pluggin does not support perfectly specular reflections"
                        " and transmissions. Rough materials should be used instead.");

                Frame<Float> frame_main_bs(sample_main_bs.wo);
                std::vector<Vector3f> ds_bs(m_dc_bsdf_samples);

                // Compute directions to samples either from the bsdf or the
                // convolution of the bsdf. Only the first one is
                // used for the light paths.
                for (size_t bs = 0; bs < m_dc_bsdf_samples; bs++) {
                    Vector2f samples = sample2D(active, sampler);
                    // Convolution: sample a vmf lobe
                    Vector3f sample_bs = warp::square_to_von_mises_fisher<Float>(samples, m_kappa_conv);
                    sample_bs = frame_main_bs.to_world(sample_bs);

                    // Otherwise: must be uncorrelated, but can sample the same component
                    auto [sample_bs_noconv, bsdf_val_bs] = bsdf->sample(ctx, si, component_sample, samples, active);

                    ds_bs[bs] = select(convolution, sample_bs, sample_bs_noconv.wo);
                }

                // Sample all these rays for discontinuity estimation
                std::vector<RayDifferential3f> rays_bs(m_dc_bsdf_samples);
                std::vector<SurfaceInteraction3f> sis_bs(m_dc_bsdf_samples);

                Mask use_reparam_bs(false);
                for (size_t bs = 0; bs < m_dc_bsdf_samples; bs++) {
                    rays_bs[bs] = si.spawn_ray(si.to_world(ds_bs[bs]));
                    sis_bs[bs] = scene->ray_intersect(rays_bs[bs], HitComputeMode::Least, active);
                    sis_bs[bs].compute_differentiable_shape_position(active);
                    // Set use_reparam_bs to true if find hit
                    use_reparam_bs = use_reparam_bs || (active && neq(sis_bs[bs].shape, nullptr));
                }

                if (m_disable_gradient_diffuse) {
                    use_reparam_bs &= !convolution;
                    current_weight = select(use_reparam_bs, current_weight, detach(current_weight));
                }

                Point3f discontinuity_bs = estimate_discontinuity(rays_bs, sis_bs, active);

                Vector3f direction_diff   = normalize(discontinuity_bs - si.p);
                Vector3f discontinuity_bs_detach = detach(discontinuity_bs);
                Vector3f direction_detach = normalize(discontinuity_bs_detach - si.p);

                Vector3f axis_bs = cross(direction_detach, direction_diff);
                Float cosangle_bs = dot(direction_diff, direction_detach);
                Transform4f rotation_bs = rotation_from_axis_cosangle(axis_bs, cosangle_bs); // This rotation is in world space

                // Initialize the BSDF sample from the initial sample, eta and
                // sampled_type do not change since the same component is sampled.
                BSDFSample3 sample_bs = sample_main_bs;

                // Reuse one direction sampled from either the BSDF or the convolution kernel
                // around the main direction.
                sample_bs.wo = ds_bs[0]; // Reuse the first one, could be any of them

                // Apply the differentiable rotation
                // Warning, the direction must be detached such that it follows the discontinuities
                // Warning, this rotation in world space, but wo is in local space
                sample_bs.wo[use_reparam_bs] = si.to_local(rotation_bs.transform_affine(
                    si.to_world(detach(sample_bs.wo))));

                // Compute the differentiable BSDF value for the differentiable direction
                Spectrum bsdf_value = bsdf->eval(ctx, si, sample_bs.wo, active);

                // Compute the pdf of the convolution kernel for the selected direction
                // Warning: need to transform to a frame centered around the Z axis
                Float pdf_conv_new_dir = warp::square_to_von_mises_fisher_pdf<Float>(frame_main_bs.to_local(sample_bs.wo),
                                                                                     m_kappa_conv);

                // Multiply the BSDF value by the convolution kernel. Use a
                // correction term for the convolution (otherwise less energy
                // at grazing angles)
                Float cosangle = sample_bs.wo.z();
                Float correction_factor = m_vmf_hemisphere.eval(m_kappa_conv, cosangle, convolution);

                bsdf_value = select(convolution, bsdf_value * pdf_conv_new_dir / correction_factor, bsdf_value);

                // Compute the value of default importance sampling pdf of the BSDF.
                // Used when convolution is disabled and for MIS
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
                                        sample_main_bs.pdf * pdf_conv_new_dir,
                                        bsdf_pdf_default);
                bsdf_pdf[use_reparam_bs] = select(convolution,
                                                  sample_main_bs.pdf * detach(pdf_conv_new_dir),
                                                  detach(bsdf_pdf_default));
                Spectrum bsdf_value_pdf = bsdf_value / bsdf_pdf;

                /* Compute weights for variance reduction
                   These weights should be:
                    - just 1 if no change of variable is used
                    - Weights whose expected gradient is 0 and value is
                      close to bsdf_value_pdf. */
                // TODO: these weights should be colors.

                Mask set_weights = use_reparam_bs && (bsdf_pdf > 0.001f);
                current_weight = select(set_weights && convolution,
                    current_weight * detach(bsdf_value_pdf[0]) * pdf_conv_new_dir / detach(pdf_conv_new_dir),
                    current_weight);
                current_weight = select(set_weights && !convolution,
                    current_weight * detach(bsdf_value_pdf[0]) * bsdf_pdf_default / detach(bsdf_pdf_default),
                    current_weight);
                throughput *= bsdf_value_pdf;

                active &= any(neq(throughput, 0.f));

                if (none(active))
                    break;

                eta *= sample_bs.eta;

                // Intersect the BSDF ray against the scene geometry
                ray = si.spawn_ray(si.to_world(sample_bs.wo));
                auto si_bsdf = scene->ray_intersect(ray, HitComputeMode::Differentiable, active);

                // Determine probability of having sampled that same
                // direction using emitter sampling.
                emitter = si_bsdf.emitter(scene, active);
                DirectionSample3f ds(si_bsdf, si);
                ds.object = emitter;

                if (any_or<true>(neq(emitter, nullptr))) {
                    Float emitter_pdf =
                        select(has_flag(sample_bs.sampled_type, BSDFFlags::Delta), 0.f,
                               scene->pdf_emitter_direction(si, ds, active));

                    // Always use the standard importance sampling pdf of the BSDF,
                    // since this is the pdf used for MIS weights when sampling emitters.
                    emission_weight = mis_weight(bsdf_pdf_default, emitter_pdf);
                }

                si = std::move(si_bsdf);
            }

            return { result, valid_ray };
        } else {
            Throw("PathReparamIntegrator: currently this integrator must be run on the GPU.");
            return {Spectrum(0.f), Mask(false)};
        }
    }

    //! @}
    // =============================================================

    std::string to_string() const override {
        return tfm::format("PathReparamIntegrator[\n"
            "  max_depth = %i,\n"
            "  rr_depth = %i\n"
            "]", m_max_depth, m_rr_depth);
    }

    MTS_DECLARE_CLASS()

protected:
    // TODO: try power heuristic, could reduce bias in gradient with large area lights
    template <typename Value> Value mis_weight(Value pdf_a, Value pdf_b) const {
        pdf_a *= pdf_a;
        pdf_b *= pdf_b;
        return select(pdf_a > 0.f, pdf_a / (pdf_a + pdf_b), Value(0.f));
    };

    mitsuba::Transform<Vector4f> rotation_from_axis_cosangle(Vector3f axis, Float cosangle) const {
        Float ax = axis.x(),
              ay = axis.y(),
              az = axis.z();
        Float axy = ax * ay,
              axz = ax * az,
              ayz = ay * az;

        Matrix3f ux(0.f, -az,  ay,
                     az, 0.f, -ax,
                    -ay,  ax, 0.f);

        Matrix3f uu(sqr(ax),     axy,    axz,
                        axy, sqr(ay),    ayz,
                        axz,     ayz, sqr(az));

        Matrix3f R = identity<Matrix3f>() * cosangle + ux + rcp(1 + cosangle) * uu;

        return mitsuba::Transform<Vector4f>(Matrix4f(R));
    };

    Point3f estimate_discontinuity(const std::vector<RayDifferential3f> &rays,
                                   const std::vector<SurfaceInteraction3f> &sis,
                                   const Mask &/*mask*/) const {

        using Matrix = enoki::Matrix<Float, 3>;

        unsigned int nb_samples = rays.size();

        if (rays.size() < 2 || rays.size() != sis.size())
            Throw("PathReparamIntegrator::estimate_discontinuity: invalid number of samples for discontinuity estimation");

        Point3f ray0_p_attached = sis[0].p;
        Vector3f ray0_n = sis[0].n;

        UInt32 is_ray1_hit_uint = select(neq(sis[1].shape, nullptr), UInt32(1), UInt32(0));
        Point3f ray1_p_attached = sis[1].p;
        Vector3f ray1_n = sis[1].n;
        Vector3f ray1_d = rays[1].d;

        for (unsigned int i = 2; i < nb_samples; i++) {
            Mask diff  = neq(sis[0].shape, sis[i].shape);
            Mask i_hit = neq(sis[i].shape, nullptr);
            is_ray1_hit_uint = select(diff, select(i_hit, UInt32(1), UInt32(0)), is_ray1_hit_uint);
            ray1_p_attached  = select(diff, sis[i].p, ray1_p_attached);
            ray1_n           = select(diff, sis[i].n, ray1_n);
            ray1_d           = select(diff, rays[i].d, ray1_d);
        }

        Mask is_ray1_hit = is_ray1_hit_uint > 0;

        // Guess occlusion for pairs of samples

        Point3f res(0.f);

        // if only one hit: return this hit
        Mask only_hit_0 = neq(sis[0].shape, nullptr) && !is_ray1_hit;
        res[only_hit_0] = ray0_p_attached;

        Mask only_hit_1 = is_ray1_hit && eq(sis[0].shape, nullptr);
        res[only_hit_1] = ray1_p_attached;

        Mask has_two_hits = neq(sis[0].shape, nullptr) && is_ray1_hit;

        // Compute occlusion between planes and hitpoints: sign of
        // dot(normal, hitpoint - hitpoint). Test if the origin of the rays
        // is on the same side as the other hit.
        Float occ_plane_0 =
            dot(ray0_n, ray1_p_attached - ray0_p_attached) *
            dot(ray0_n, rays[0].o - ray0_p_attached);
        Float occ_plane_1 = dot(ray1_n, ray0_p_attached - ray1_p_attached) *
                            dot(ray0_n, rays[0].o - ray0_p_attached);

        Mask plane_0_occludes_1 = has_two_hits && (occ_plane_0 < 0.f);
        Mask plane_1_occludes_0 = has_two_hits && (occ_plane_1 < 0.f);

        Mask simple_occluder_0 = plane_0_occludes_1 && !plane_1_occludes_0;
        Mask simple_occluder_1 = plane_1_occludes_0 && !plane_0_occludes_1;
        Mask plane_intersection = has_two_hits  && !simple_occluder_1 && !simple_occluder_0;

        /* simple_occluder */

        res[simple_occluder_0] = ray0_p_attached;
        res[simple_occluder_1] = ray1_p_attached;

        /* same_normals */

        Mask same_normals = plane_intersection  && abs(dot(ray0_n, ray1_n)) > 0.99f;
        plane_intersection &= !same_normals;
        res[same_normals] = ray0_p_attached;

        /* plane_intersection */

#if 1
        // Compute the intersection between 3 planes:
        // 2 planes defined by the ray intersections and
        // the normals at these points, and 1 plane containing
        // the ray directions.

        Vector3f N0 = ray0_n;
        Vector3f N1 = ray1_n;
        Vector3f P0 = ray0_p_attached;
        Vector3f P1 = ray1_p_attached;

        // Normal of the third plane, defined using
        // attached positions (this prevents bad correlations
        // between the displacement of the intersection and
        // the sampled positions)

        Vector3f N = cross(P0 - rays[0].o, P1 - rays[0].o);
        Float norm_N = norm(N);

        // Set a default intersection if the problem is ill-defined
        res[plane_intersection] = ray0_p_attached;

        Mask invertible = plane_intersection && norm_N > 0.001f;

        Matrix A = Matrix::from_rows(N0, N1, N);
        Float b0 =  dot(P0, N0);
        Float b1 =  dot(P1, N1);
        Float b2 =  dot(rays[0].o, N);
        Vector3f B(b0, b1, b2);
        Matrix invA = enoki::inverse(A);
        res[invertible] = invA * B;
#else
        // Simply choose one of the intersections.
        // This is a good strategy in many situations.
        res[plane_intersection] = ray0_p_attached;
#endif

        return res;

    }

private:
    size_t m_disable_gradient_bounce;
    size_t m_dc_light_samples;
    size_t m_dc_bsdf_samples;
    size_t m_dc_cam_samples;
    ScalarFloat m_conv_threshold;
    ScalarFloat m_kappa_conv;
    ScalarFloat m_kappa_conv_envmap;
    bool m_use_variance_reduction;
    bool m_use_convolution;
    bool m_use_convolution_envmap;
    bool m_disable_gradient_diffuse;

    VMFHemisphereIntegral<Float> m_vmf_hemisphere;
};

MTS_IMPLEMENT_CLASS_VARIANT(PathReparamIntegrator, MonteCarloIntegrator);
MTS_EXPORT_PLUGIN(PathReparamIntegrator, "Differentiable Path Tracer integrator");
NAMESPACE_END(mitsuba)
