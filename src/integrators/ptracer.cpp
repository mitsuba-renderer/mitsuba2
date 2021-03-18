#include <random>

// TODO: remove dependency on TBB
#include <tbb/parallel_for.h>
#include <tbb/spin_mutex.h>

#include <enoki/morton.h>
#include <mitsuba/core/fwd.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/progress.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>
#include <mitsuba/render/sampler.h>

#include <fstream>

// #define MTS_DEBUG_PTRACER_PATHS "/tmp/ptracer.obj"
#if defined(MTS_DEBUG_PTRACER_PATHS)
#include <fstream>
namespace {
static size_t export_counter = 0;
} // namespace
#endif

NAMESPACE_BEGIN(mitsuba)

/**
 * Traces rays from the light source and attempts to connect them to the sensor
 * at each bounce.
 */
template <typename Float, typename Spectrum>
class ParticleTracerIntegrator : public Integrator<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Integrator, should_stop, aov_names, m_stop, m_timeout, m_render_timer)
    MTS_IMPORT_TYPES(Scene, Sensor, Film, Sampler, ImageBlock, Emitter,
                     EmitterPtr, BSDF, BSDFPtr)

    ParticleTracerIntegrator(const Properties &props) : Base(props) {
        // TODO: consider moving those parameters to the base class.

        m_samples_per_pass = (uint32_t) props.size_("samples_per_pass", (size_t) -1);
        m_hide_emitters = props.bool_("hide_emitters", false);

        m_rr_depth = props.int_("rr_depth", 5);
        if (m_rr_depth <= 0)
            Throw("\"rr_depth\" must be set to a value greater than zero!");

        m_max_depth = props.int_("max_depth", -1);
        if (m_max_depth < 0 && m_max_depth != -1)
            Throw("\"max_depth\" must be set to -1 (infinite) or a value >= 0");
    }

    /// Perform the main rendering job
    bool render(Scene *scene, Sensor *sensor) override  {
        ScopedPhase sp(ProfilerPhase::Render);
        m_stop = false;
        Film *film = sensor->film();
        ScalarVector2i film_size = film->crop_size();

        if (unlikely(scene->emitters().empty())) {
            Log(Warn, "Scene does not contain any emitter, returning black image.");
            normalize_film(film, 1);
            return true;
        }

        size_t samples_per_pixel = sensor->sampler()->sample_count();
        size_t samples_per_pass_per_pixel =
            (m_samples_per_pass == (size_t) -1)
                ? samples_per_pixel
                : std::min((size_t) m_samples_per_pass, samples_per_pixel);
        if ((samples_per_pixel % samples_per_pass_per_pixel) != 0)
            Throw("sample_count (%d) must be a multiple of samples_per_pass (%d).",
                  samples_per_pixel, samples_per_pass_per_pixel);

        /* Multiply sample count by pixel count to obtain a similar scale
         * to the standard path tracer. */
        size_t samples_per_pass = samples_per_pass_per_pixel * hprod(film_size);
        size_t n_passes =
            (samples_per_pixel * hprod(film_size) + samples_per_pass - 1) / samples_per_pass;
        size_t total_samples = samples_per_pass * n_passes;

        std::vector<std::string> channels = aov_names();
        bool has_aovs = !channels.empty();
        if (has_aovs)
            Throw("Not supported yet: AOVs in LightTracerIntegrator");
        // Insert default channels and set up the film
        for (size_t i = 0; i < 5; ++i)
            channels.insert(channels.begin() + i, std::string(1, "XYZAW"[i]));
        film->prepare(channels);

        size_t total_samples_done = 0;
        m_render_timer.reset();
        if constexpr (!ek::is_jit_array_v<Float>) {
            size_t n_threads = __global_thread_count;

            Log(Info, "Starting render job (%ix%i, %i sample%s,%s %i thread%s)",
                film_size.x(), film_size.y(),
                total_samples, total_samples == 1 ? "" : "s",
                n_passes > 1 ? tfm::format(" %d passes,", n_passes) : "",
                n_threads, n_threads == 1 ? "" : "s");
            if (m_timeout > 0.f)
                Log(Info, "Timeout specified: %.2f seconds.", m_timeout);

            // Split up all samples between threads
            size_t grain_count = std::max(
                (size_t) 1, (size_t) std::rint(samples_per_pass /
                                               ScalarFloat(n_threads * 2)));

            ThreadEnvironment env;
            tbb::spin_mutex mutex;
            ref<ProgressReporter> progress = new ProgressReporter("Rendering");
            size_t update_threshold = std::max(grain_count / 10, 10000lu);

            tbb::parallel_for(
                tbb::blocked_range<size_t>(0, total_samples, grain_count),
                [&](const tbb::blocked_range<size_t> &range) {
                    ScopedSetThreadEnvironment set_env(env);

                    ref<Sampler> sampler = sensor->sampler()->clone();
                    ref<ImageBlock> block = new ImageBlock(
                        film->size(), channels.size(), film->reconstruction_filter(),
                        /* warn_negative */ !has_aovs,
                        /* warn_invalid */ true, /* border */ true,
                        /* normalize */ true);

                    size_t samples_done = 0;
                    block->clear();
                    size_t seed = (size_t) range.begin();
                    sampler->seed(seed);

                    for (auto i = range.begin(); i != range.end() && !should_stop(); ++i) {
                        if (likely(m_max_depth != 0) && !m_hide_emitters) {
                            // Account for emitters directly visible from the sensor
                            sample_visible_emitters(scene, sensor, sampler, block);
                        }

                        // Primary & further bounces illumination
                        auto [ray, throughput] = prepare_ray(scene, sensor, sampler);
                        trace_light_ray(ray, scene, sensor, sampler, throughput, block);

                        // TODO: it is possible that there are more than 1 sample / it ?
                        samples_done += 1;
                        if (samples_done > update_threshold) {
                            tbb::spin_mutex::scoped_lock lock(mutex);
                            total_samples_done += samples_done;
                            samples_done = 0;
                            progress->update(total_samples_done / (ScalarFloat) total_samples);
                        }
                    }

                    // When all samples are done for this range, commit to the film
                    /* locked */ {
                        tbb::spin_mutex::scoped_lock lock(mutex);
                        total_samples_done += samples_done;
                        progress->update(total_samples_done / (ScalarFloat) total_samples);

                        film->put(block);
                    }
                });
        } else {
            // Wavefront rendering
            Log(Info, "Starting render job (%ix%i, %i sample%s,%s)",
                film_size.x(), film_size.y(),
                total_samples, total_samples == 1 ? "" : "s",
                n_passes > 1 ? tfm::format(" %d passes", n_passes) : "");

            ref<Sampler> sampler = sensor->sampler();
            // Implicitly, the sampler expects samples per pixel per pass.
            sampler->set_samples_per_wavefront((uint32_t) samples_per_pass_per_pixel);

            ScalarUInt32 wavefront_size = (uint32_t) samples_per_pass;
            if (sampler->wavefront_size() != wavefront_size)
                sampler->seed(0, wavefront_size);

            ref<ImageBlock> block = new ImageBlock(
                film_size, channels.size(), film->reconstruction_filter(),
                /* warn_negative */ !has_aovs && std::is_scalar_v<Float>,
                /* warn_invalid */ true, /* border */ true,
                /* normalize */ true);
            block->clear();

            for (size_t i = 0; i < n_passes; i++) {
                if (likely(m_max_depth != 0) && !m_hide_emitters) {
                    // Account for emitters directly visible from the sensor
                    sample_visible_emitters(scene, sensor, sampler, block);
                }

                // Primary & further bounces illumination
                auto [ray, throughput] = prepare_ray(scene, sensor, sampler);
                trace_light_ray(ray, scene, sensor, sampler, throughput, block);

                total_samples_done += samples_per_pass;
            }

            film->put(block);
        }

        // Apply proper normalization.
        double new_weight = normalize_film(film, total_samples_done);
        Log(Info, "Processed %d samples, normalization weight: %f",
            total_samples_done, new_weight);

        if (!m_stop)
            Log(Info, "Rendering finished. (took %s)",
                util::time_string((float) m_render_timer.value(), true));

        return !m_stop;
    }

    void cancel() override {
        m_stop = true;
    }

    /**
     * Samples an emitter in the scene and connects it directly to the sensor,
     * writing the emitted radiance to the given image block.
     */
    virtual Spectrum sample_visible_emitters(Scene *scene, const Sensor *sensor,
                                             Sampler *sampler,
                                             ImageBlock *block,
                                             Mask active = true) const {
        // 1. Time sampling
        Float time = sensor->shutter_open();
        if (sensor->shutter_open_time() > 0)
            time += sampler->next_1d(active) * sensor->shutter_open_time();

        // 2. Emitter sampling (select one emitter)
        Float idx_sample = sampler->next_1d(active);
        auto [emitter_idx, emitter_idx_weight, _] =
            scene->sample_emitter(idx_sample, active);
        EmitterPtr emitter =
            ek::gather<EmitterPtr>(scene->emitters_ek(), emitter_idx, active);

        // 3. Emitter position sampling
        Spectrum emitter_weight = ek::zero<Spectrum>();
        SurfaceInteraction3f si = ek::zero<SurfaceInteraction3f>();
        Point2f emitter_sample  = sampler->next_2d(active);
        Mask is_infinite = has_flag(emitter->flags(), EmitterFlags::Infinite);
        Mask active_e = active && is_infinite;
        if (ek::any_or<true>(active_e)) {
            /* We are sampling a direction toward an envmap emitter starting
             * from the center of the scene. This is because the sensor is
             * not part of the scene's bounding box, which could cause issues.
             */
            Interaction3f ref_it(0.f, time, ek::zero<Wavelength>(),
                                 sensor->world_transform()->eval(time).translation());
            auto [ds, dir_weight] =
                emitter->sample_direction(ref_it, emitter_sample, active_e);
            /* Note: `dir_weight` already includes the emitter radiance, but
             * that will be accounted for again when sampling the wavelength
             * below. Instead, we recompute just the factor due to the PDF.
             * Also, convert to area measure.
             */
            emitter_weight[active_e] =
                ek::select(ds.pdf > 0.f, 1.f / ds.pdf, 0.f) * ek::sqr(ds.dist);

            si[active_e] = SurfaceInteraction3f(ds, ref_it.wavelengths);
        }

        active_e = active && !is_infinite;
        if (ek::any_or<true>(active_e)) {
            auto [ps, pos_weight] =
                emitter->sample_position(time, emitter_sample, active_e);
            emitter_weight[active_e] = pos_weight;
            si[active_e] = SurfaceInteraction3f(ps, ek::empty<Wavelength>());
        }

        /* 4. Connect to the sensor.
        Query sensor for a direction connecting to `si.p`. This also gives
            us UVs on the sensor (for splatting).
        The resulting direction points from si.p (on the emitter) toward the sensor. */
        auto [sensor_ds, sensor_weight] = sensor->sample_direction(si, sampler->next_2d(), active);
        si.wi = sensor_ds.d;

        // 5. Sample spectrum of the emitter (accounts for its radiance)
        Float wavelength_sample = sampler->next_1d(active);
        auto [wavelengths, wav_weight] =
            emitter->sample_wavelengths(si, wavelength_sample, active);
        si.wavelengths = wavelengths;
        si.shape       = emitter->shape();

        Spectrum weight = emitter_idx_weight * emitter_weight * wav_weight * sensor_weight;

        // No BSDF passed (should not evaluate it since there's no scattering)
        return connect_sensor(scene, si, sensor_ds, nullptr, weight, block, active);
    }


    /**
     * Samples a ray from a random emitter in the scene.
     */
    std::pair<Ray3f, Spectrum> prepare_ray(const Scene *scene,
                                           const Sensor *sensor,
                                           Sampler *sampler,
                                           Mask active = true) const {
        Float time = sensor->shutter_open();
        if (sensor->shutter_open_time() > 0)
            time += sampler->next_1d(active) * sensor->shutter_open_time();

        /* ---------------------- Sample ray from emitter ------------------- */
        // Prepare random samples.
        auto wavelength_sample = sampler->next_1d(active);
        auto direction_sample  = sampler->next_2d(active);
        auto position_sample   = sampler->next_2d(active);
        // Sample one ray from an emitter in the scene.
        auto [ray, ray_weight, emitter] = scene->sample_emitter_ray(
            time, wavelength_sample, direction_sample, position_sample, active);
        return std::make_pair(ray, ray_weight);
    }

    /**
     * Intersects the given ray with the scene and recursively trace using
     * BSDF sampling. The given `throughput` should account for emitted
     * radiance from the sampled light source, wavelengths sampling weights,
     * etc. At each interaction, we attempt connecting to the sensor and add
     * throughput to the given `block`.
     *
     * Note: this will *not* account for directly visible emitters, since
     * they require a direct connection from the emitter to the sensor. See
     * \ref sample_visible_emitters.
     *
     * Returns (radiance, alpha)
     */
    std::pair<Spectrum, Float>
    trace_light_ray(Ray3f ray, const Scene *scene, const Sensor *sensor,
                    Sampler *sampler, Spectrum throughput,
                    ImageBlock *block, Mask active = true) const {
        // Tracks radiance scaling due to index of refraction changes
        Float eta(1.f);
        BSDFContext ctx(TransportMode::Importance);

        /* ---------------------- Path construction ------------------------- */
        // First intersection from the emitter to the scene
        SurfaceInteraction3f si = scene->ray_intersect(ray, active);

        // Incrementally build light path using BSDF sampling.
        for (int depth = 1;; ++depth) {
            active &= si.is_valid();

            // Russian Roulette
            if (depth > m_rr_depth) {
                Float q = ek::min(ek::hmax(throughput) * eta * eta, 0.95f);
                active &= sampler->next_1d(active) < q;
                throughput *= ek::rcp(q);
            }

            if ((uint32_t) depth >= (uint32_t) m_max_depth || ek::none_or<false>(active))
                break;

            // Connect to sensor and splat if successful.
            auto [sensor_ds, sensor_weight] =
                sensor->sample_direction(si, sampler->next_2d(), active);
            connect_sensor(scene, si, sensor_ds, si.bsdf(ray),
                           throughput * sensor_weight, block, active);

            /* ----------------------- BSDF sampling ------------------------ */
            // Sample BSDF * cos(theta).
            BSDFSample3f bs;
            Spectrum bsdf_val;
            std::tie(bs, bsdf_val) =
                si.bsdf(ray)->sample(ctx, si, sampler->next_1d(active),
                                     sampler->next_2d(active), active);
            // Using geometric normals (wo points to the camera)
            Float wi_dot_geo_n = dot(si.n, -ray.d),
                  wo_dot_geo_n = dot(si.n, si.to_world(bs.wo));
            // Prevent light leaks due to shading normals
            active &= (wi_dot_geo_n * Frame3f::cos_theta(si.wi) > 0.f) &&
                      (wo_dot_geo_n * Frame3f::cos_theta(bs.wo) > 0.f);
            // Adjoint BSDF for shading normals -- [Veach, p. 155]
            auto correction = abs((Frame3f::cos_theta(si.wi) * wo_dot_geo_n) /
                                  (Frame3f::cos_theta(bs.wo) * wi_dot_geo_n));
            throughput *= bsdf_val * correction;
            eta *= bs.eta;

            active &= any(neq(throughput, 0.f));
            if (ek::none_or<false>(active))
                break;

            // Intersect the BSDF ray against scene geometry (next vertex).
            ray = si.spawn_ray(si.to_world(bs.wo));
            si = scene->ray_intersect(ray, active);
        }

        // TODO: proper alpha support
        return { throughput, 1.f };
    }


    /**
     * From the given surface interaction, attempt connecting to the sensor
     * and splat to the given block if successful.
     */
    Spectrum connect_sensor(const Scene *scene, const SurfaceInteraction3f &si,
                            const DirectionSample3f &sensor_ds,
                            const BSDFPtr bsdf, const Spectrum &weight,
                            ImageBlock *block, Mask active) const {
        active &= (sensor_ds.pdf > 0.f) && any(neq(weight, 0.f));
        Spectrum result = 0.f;
        if (ek::none_or<false>(active))
            return result;

#if defined(MTS_DEBUG_PTRACER_PATHS)
        size_t i  = 0;
        auto mode = (export_counter == 0 ? std::ios::out : std::ios::app);
        std::ofstream f(MTS_DEBUG_PTRACER_PATHS, mode);
        auto after  = si.p + ds.d;
        auto target = si.p + ds.dist * ds.d;

        f << "v " << si.p.x() << " " << si.p.y() << " " << si.p.z() << std::endl;
        f << "v " << after.x() << " " << after.y() << " " << after.z() << std::endl;
        f << "v " << target.x() << " " << target.y() << " " << target.z() << std::endl;
        i += 3;
        // // Show normal at entry point
        // auto disp = si.p + 0.1f * si.n;
        // f << "v " << disp.x() << " " << disp.y() << " " << disp.z()
        //   << std::endl;
        // f << "v " << si.p.x() << " " << si.p.y() << " " << si.p.z()
        //   << std::endl;
        // i += 2;

        export_counter += i;

        f << "l";
        for (size_t j = 1; j <= i + 1; ++j)
            f << " " << (export_counter + j);
        f << std::endl;
        export_counter += i + 1;

        if (export_counter >= 1000)
            Throw("Done here");
#endif

        // Check that sensor is visible from current position (shadow ray).
        auto sensor_ray = si.spawn_ray_to(sensor_ds.p);
        active &= !scene->ray_test(sensor_ray, active);
        if (ek::none_or<false>(active))
            return result;

        /* Foreshortening term and BSDF value for that direction (for surface interactions)
         * Note that foreshortening is only missing for directly visible emitters associated
         * with a shape (marked by convention by bsdf != nullptr). */
        Spectrum surface_weight = 1.f;
        auto local_d            = si.to_local(sensor_ray.d);
        auto on_surface         = active && neq(si.shape, nullptr);
        if (ek::any_or<true>(on_surface)) {
            // Clamp negative cosines -> zero value if behind the surface
            surface_weight[on_surface && eq(bsdf, nullptr)] *=
                ek::max(0.f, Frame3f::cos_theta(local_d));

            on_surface &= neq(bsdf, nullptr);
            if (ek::any_or<true>(on_surface)) {
                BSDFContext ctx(TransportMode::Importance);
                // Using geometric normals
                Float wi_dot_geo_n = dot(si.n, si.to_world(si.wi)),
                      wo_dot_geo_n = dot(si.n, sensor_ray.d);
                // Prevent light leaks due to shading normals
                auto valid = (wi_dot_geo_n * Frame3f::cos_theta(si.wi) > 0.f) &&
                             (wo_dot_geo_n * Frame3f::cos_theta(local_d) > 0.f);

                // Adjoint BSDF for shading normals -- [Veach, p. 155]
                auto correction = ek::select(
                    valid,
                    ek::abs((Frame3f::cos_theta(si.wi) * wo_dot_geo_n) /
                               (Frame3f::cos_theta(local_d) * wi_dot_geo_n)),
                    ek::zero<Float>());
                surface_weight[on_surface] *= correction * bsdf->eval(ctx, si, local_d, active);
            }
        }

        // Even if the ray is not coming from a surface (no foreshortening),
        // we still don't want light coming from behind the emitter.
        auto on_infinite_emitter = active && eq(si.shape, nullptr) && eq(bsdf, nullptr);
        if (ek::any_or<true>(on_infinite_emitter)) {
            auto right_side = Frame3f::cos_theta(local_d) > 0.f;
            surface_weight[on_infinite_emitter] = select(right_side, surface_weight, 0.f);
        }

        result = weight * surface_weight;

        // Splatting
        Point2f splat_uv = ek::select(active, sensor_ds.uv, 0.f);
        block->put(splat_uv, si.wavelengths, result, Float(1.f), active);

        return result;
    }

    //! @}
    // =============================================================

    /**
     * Overwrites the accumulated per-pixel normalization weights with a
     * constant weight computed from the number of overall accumulated samples.
     *
     * \return The new normalization weight.
     */
    double normalize_film(Film *film, size_t total_samples) const {
        double new_weight = total_samples / (double) hprod(film->size());
        film->reweight(new_weight);
        return new_weight;
    }


    std::string to_string() const override {
        return tfm::format("ParticleTracerIntegrator[\n"
                           "  max_depth = %i,\n"
                           "  rr_depth = %i\n"
                           "]",
                           m_max_depth, m_rr_depth);
    }

    MTS_DECLARE_CLASS()

protected:
    /**
     * \brief Number of samples to compute for each pass over the image blocks.
     *
     * Must be a multiple of the total sample count per pixel.
     * If set to (size_t) -1, all the work is done in a single pass (default).
     */
    uint32_t m_samples_per_pass;

    /// Flag for disabling direct visibility of emitters
    bool m_hide_emitters;

    /**
     * Longest visualized path depth (\c -1 = infinite).
     * A value of \c 1 will visualize only directly visible light sources.
     * \c 2 will lead to single-bounce (direct-only) illumination, and so on.
     */
    int m_max_depth;

    /// Depth to begin using russian roulette
    int m_rr_depth;
};

MTS_IMPLEMENT_CLASS_VARIANT(ParticleTracerIntegrator, Integrator);
MTS_EXPORT_PLUGIN(ParticleTracerIntegrator, "Particle Tracer integrator");
NAMESPACE_END(mitsuba)
