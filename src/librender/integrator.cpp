#include <thread>
#include <mutex>

#include <enoki/morton.h>
#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/fwd.h>
#include <mitsuba/core/profiler.h>
#include <mitsuba/core/progress.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/timer.h>
#include <mitsuba/core/util.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/core/fstream.h>
#include <mitsuba/render/film.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/sampler.h>
#include <mitsuba/render/sensor.h>
#include <mitsuba/render/spiral.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

NAMESPACE_BEGIN(mitsuba)

// -----------------------------------------------------------------------------

MTS_VARIANT Integrator<Float, Spectrum>::Integrator(const Properties & props)
    : m_stop(false) {
    m_timeout = props.float_("timeout", -1.f);
}

// -----------------------------------------------------------------------------

MTS_VARIANT SamplingIntegrator<Float, Spectrum>::SamplingIntegrator(const Properties &props)
    : Base(props) {

    m_block_size = (uint32_t) props.size_("block_size", 0);
    uint32_t block_size = math::round_to_power_of_two(m_block_size);
    if (m_block_size > 0 && block_size != m_block_size) {
        Log(Warn, "Setting block size from %i to next higher power of two: %i", m_block_size,
            block_size);
        m_block_size = block_size;
    }

    m_samples_per_pass = (uint32_t) props.size_("samples_per_pass", (size_t) -1);

    /// Disable direct visibility of emitters if needed
    m_hide_emitters = props.bool_("hide_emitters", false);
}

MTS_VARIANT SamplingIntegrator<Float, Spectrum>::~SamplingIntegrator() { }

MTS_VARIANT void SamplingIntegrator<Float, Spectrum>::cancel() {
    m_stop = true;
}

MTS_VARIANT std::vector<std::string> SamplingIntegrator<Float, Spectrum>::aov_names() const {
    return { };
}

MTS_VARIANT bool SamplingIntegrator<Float, Spectrum>::render(Scene *scene, Sensor *sensor) {
    ScopedPhase sp(ProfilerPhase::Render);
    m_stop = false;

    ref<Film> film = sensor->film();
    ScalarVector2i film_size = film->crop_size();

    size_t total_spp        = sensor->sampler()->sample_count();
    size_t samples_per_pass = (m_samples_per_pass == (size_t) -1)
                               ? total_spp : std::min((size_t) m_samples_per_pass, total_spp);
    if ((total_spp % samples_per_pass) != 0)
        Throw("sample_count (%d) must be a multiple of samples_per_pass (%d).",
              total_spp, samples_per_pass);

    size_t n_passes = (total_spp + samples_per_pass - 1) / samples_per_pass;

    std::vector<std::string> channels = aov_names();
    bool has_aovs = !channels.empty();

    // Insert default channels and set up the film
    for (size_t i = 0; i < 5; ++i)
        channels.insert(channels.begin() + i, std::string(1, "XYZAW"[i]));
    film->prepare(channels);

    m_render_timer.reset();
    if constexpr (!ek::is_jit_array_v<Float>) {
        /// Render on the CPU using a spiral pattern
        size_t n_threads = __global_thread_count;
        Log(Info, "Starting render job (%ix%i, %i sample%s,%s %i thread%s)",
            film_size.x(), film_size.y(),
            total_spp, total_spp == 1 ? "" : "s",
            n_passes > 1 ? tfm::format(" %d passes,", n_passes) : "",
            n_threads, n_threads == 1 ? "" : "s");

        if (m_timeout > 0.f)
            Log(Info, "Timeout specified: %.2f seconds.", m_timeout);

        // Find a good block size to use for splitting up the total workload.
        if (m_block_size == 0) {
            uint32_t block_size = MTS_BLOCK_SIZE;
            while (true) {
                if (block_size == 1 || ek::hprod((film_size + block_size - 1) / block_size) >= n_threads)
                    break;
                block_size /= 2;
            }
            m_block_size = block_size;
        }

        Spiral spiral(film, m_block_size, n_passes);

        ThreadEnvironment env;
        ref<ProgressReporter> progress = new ProgressReporter("Rendering");
        std::mutex mutex;

        // Total number of blocks to be handled, including multiple passes.
        size_t total_blocks = spiral.block_count() * n_passes,
               blocks_done = 0;

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, total_blocks, 1),
            [&](const tbb::blocked_range<size_t> &range) {
                ScopedSetThreadEnvironment set_env(env);
                ref<Sampler> sampler = sensor->sampler()->clone();
                ref<ImageBlock> block = new ImageBlock(m_block_size, channels.size(),
                                                       film->reconstruction_filter(),
                                                       !has_aovs);
                std::unique_ptr<Float[]> aovs(new Float[channels.size()]);

                // For each block
                for (auto i = range.begin(); i != range.end() && !should_stop(); ++i) {
                    auto [offset, size, block_id] = spiral.next_block();
                    Assert(ek::hprod(size) != 0);
                    block->set_size(size);
                    block->set_offset(offset);

                    render_block(scene, sensor, sampler, block,
                                 aovs.get(), samples_per_pass, block_id);

                    film->put(block);

                    /* Critical section: update progress bar */ {
                        std::lock_guard<std::mutex> lock(mutex);
                        blocks_done++;
                        progress->update(blocks_done / (float) total_blocks);
                    }
                }
            }
        );
    } else {
        Log(Info, "Start rendering...");
        // jit_optix_set_launch_size(film_size.x(), film_size.y(), samples_per_pass);

        ref<Sampler> sampler = sensor->sampler();
        sampler->set_samples_per_wavefront((uint32_t) samples_per_pass);

        ScalarFloat diff_scale_factor = ek::rsqrt((ScalarFloat) sampler->sample_count());
        ScalarUInt32 wavefront_size = ek::hprod(film_size) * (uint32_t) samples_per_pass;
        if (sampler->wavefront_size() != wavefront_size)
            sampler->seed(0, wavefront_size);

        UInt32 idx = ek::arange<UInt32>(wavefront_size);
        if (samples_per_pass != 1)
            idx /= (uint32_t) samples_per_pass;

        ref<ImageBlock> block = new ImageBlock(film_size, channels.size(),
                                               film->reconstruction_filter(),
                                               !has_aovs && std::is_scalar_v<Float>);
        block->clear();
        Vector2f pos = Vector2f(Float(idx % uint32_t(film_size[0])),
                                Float(idx / uint32_t(film_size[0])));
        std::vector<Float> aovs(channels.size());

        for (size_t i = 0; i < n_passes; i++)
            render_sample(scene, sensor, sampler, block, aovs.data(),
                          pos, diff_scale_factor);

        film->put(block);
    }

    if (!m_stop)
        Log(Info, "Rendering finished. (took %s)",
            util::time_string((float) m_render_timer.value(), true));

    return !m_stop;
}

MTS_VARIANT void SamplingIntegrator<Float, Spectrum>::render_block(const Scene *scene,
                                                                   const Sensor *sensor,
                                                                   Sampler *sampler,
                                                                   ImageBlock *block,
                                                                   Float *aovs,
                                                                   size_t sample_count_,
                                                                   size_t block_id) const {
    block->clear();
    uint32_t pixel_count  = (uint32_t)(m_block_size * m_block_size),
             sample_count = (uint32_t)(sample_count_ == (size_t) -1
                                           ? sampler->sample_count()
                                           : sample_count_);

    ScalarFloat diff_scale_factor = ek::rsqrt((ScalarFloat) sampler->sample_count());

    if constexpr (!ek::is_array_v<Float>) {
        for (uint32_t i = 0; i < pixel_count && !should_stop(); ++i) {
            sampler->seed(block_id * pixel_count + i);

            ScalarPoint2u pos = ek::morton_decode<ScalarPoint2u>(i);
            if (ek::any(pos >= block->size()))
                continue;

            pos += block->offset();
            for (uint32_t j = 0; j < sample_count && !should_stop(); ++j) {
                render_sample(scene, sensor, sampler, block, aovs,
                              pos, diff_scale_factor);
            }
        }
    } else if constexpr (ek::is_array_v<Float> && !ek::is_cuda_array_v<Float>) {
        // Ensure that the sample generation is fully deterministic
        sampler->seed(block_id);

        for (auto [index, active] : ek::range<UInt32>(pixel_count * sample_count)) {
            if (should_stop())
                break;
            Point2u pos = ek::morton_decode<Point2u>(index / UInt32(sample_count));
            active &= !any(pos >= block->size());
            pos += block->offset();
            render_sample(scene, sensor, sampler, block, aovs, pos, diff_scale_factor, active);
        }
    } else {
        ENOKI_MARK_USED(scene);
        ENOKI_MARK_USED(sensor);
        ENOKI_MARK_USED(aovs);
        ENOKI_MARK_USED(diff_scale_factor);
        ENOKI_MARK_USED(pixel_count);
        ENOKI_MARK_USED(sample_count);
        ENOKI_MARK_USED(block_id);
        Throw("Not implemented for CUDA arrays.");
    }
}

MTS_VARIANT void
SamplingIntegrator<Float, Spectrum>::render_sample(const Scene *scene,
                                                   const Sensor *sensor,
                                                   Sampler *sampler,
                                                   ImageBlock *block,
                                                   Float *aovs,
                                                   const Vector2f &pos,
                                                   ScalarFloat diff_scale_factor,
                                                   Mask active) const {
    std::conditional_t<ek::is_jit_array_v<Float>, Timer, int> timer;

    Vector2f position_sample = pos + sampler->next_2d(active);

    Point2f aperture_sample(.5f);
    if (sensor->needs_aperture_sample())
        aperture_sample = sampler->next_2d(active);

    Float time = sensor->shutter_open();
    if (sensor->shutter_open_time() > 0.f)
        time += sampler->next_1d(active) * sensor->shutter_open_time();

    Float wavelength_sample = sampler->next_1d(active);

    Vector2f adjusted_position =
        (position_sample - sensor->film()->crop_offset()) /
        sensor->film()->crop_size();

    auto [ray, ray_weight] = sensor->sample_ray_differential(
        time, wavelength_sample, adjusted_position, aperture_sample);

    ray.scale_differential(diff_scale_factor);

    const Medium *medium = sensor->medium();
    std::pair<Spectrum, Mask> result = sample(scene, sampler, ray, medium, aovs + 5, active);
    result.first = ray_weight * result.first;

    UnpolarizedSpectrum spec_u = depolarize(result.first);

    Color3f xyz;
    if constexpr (is_monochromatic_v<Spectrum>) {
        xyz = spec_u.x();
    } else if constexpr (is_rgb_v<Spectrum>) {
        xyz = srgb_to_xyz(spec_u, active);
    } else {
        static_assert(is_spectral_v<Spectrum>);
        xyz = spectrum_to_xyz(spec_u, ray.wavelengths, active);
    }

    aovs[0] = xyz.x();
    aovs[1] = xyz.y();
    aovs[2] = xyz.z();
    aovs[3] = ek::select(result.second, Float(1.f), Float(0.f));
    aovs[4] = 1.f;

    block->put(position_sample, aovs, active);

    if constexpr (ek::is_jit_array_v<Float>) {
        if (jit_flag(JitFlag::VCallRecord) && jit_flag(JitFlag::LoopRecord)) {
            Log(Info, "Computation graph recorded. (took %s)",
                util::time_string((float) timer.reset(), true));
        }

        auto &out = Base::m_graphviz_output;
        if (!out.empty()) {
            ref<FileStream> out_stream =
                new FileStream(out, FileStream::ETruncReadWrite);
            const char *graph = jit_var_graphviz();
            out_stream->write(graph, strlen(graph));
        }

        ek::eval(block->data());
        if (jit_flag(JitFlag::VCallRecord) && jit_flag(JitFlag::LoopRecord)) {
            Log(Info, "Code generation finished. (took %s)",
                util::time_string((float) timer.reset(), true));
        }

        ek::sync_thread();
    } else {
        ENOKI_MARK_USED(timer);
    }

    sampler->advance();
}

MTS_VARIANT std::pair<Spectrum, typename SamplingIntegrator<Float, Spectrum>::Mask>
SamplingIntegrator<Float, Spectrum>::sample(const Scene * /* scene */,
                                            Sampler * /* sampler */,
                                            const RayDifferential3f & /* ray */,
                                            const Medium * /* medium */,
                                            Float * /* aovs */,
                                            Mask /* active */) const {
    NotImplementedError("sample");
}

// -----------------------------------------------------------------------------

MTS_VARIANT MonteCarloIntegrator<Float, Spectrum>::MonteCarloIntegrator(const Properties &props)
    : Base(props) {
    /// Depth to begin using russian roulette
    m_rr_depth = props.int_("rr_depth", 5);
    if (m_rr_depth <= 0)
        Throw("\"rr_depth\" must be set to a value greater than zero!");

    /*  Longest visualized path depth (``-1 = infinite``). A value of \c 1 will
        visualize only directly visible light sources. \c 2 will lead to
        single-bounce (direct-only) illumination, and so on. */
    m_max_depth = props.int_("max_depth", -1);
    if (m_max_depth < 0 && m_max_depth != -1)
        Throw("\"max_depth\" must be set to -1 (infinite) or a value >= 0");
}

MTS_VARIANT MonteCarloIntegrator<Float, Spectrum>::~MonteCarloIntegrator() { }

// -----------------------------------------------------------------------------

MTS_VARIANT LightTracerIntegrator<Float, Spectrum>::LightTracerIntegrator(const Properties &props)
    : Base(props) {
    // TODO: consider moving those parameters to the base class.
    m_rr_depth = props.int_("rr_depth", 5);
    if (m_rr_depth <= 0)
        Throw("\"rr_depth\" must be set to a value greater than zero!");

    m_max_depth = props.int_("max_depth", -1);
    if (m_max_depth < 0 && m_max_depth != -1)
        Throw("\"max_depth\" must be set to -1 (infinite) or a value >= 0");
}

MTS_VARIANT LightTracerIntegrator<Float, Spectrum>::~LightTracerIntegrator() { }

MTS_VARIANT bool LightTracerIntegrator<Float, Spectrum>::render(Scene *scene, Sensor *sensor) {
    static_assert(std::is_scalar_v<Float>); // TODO: support other modes
    ScopedPhase sp(ProfilerPhase::Render);
    m_stop = false;
    m_render_timer.reset();
    Film *film = sensor->film();

    if (scene->emitters().empty()) {
        Log(Warn, "Scene does not contain any emitter, returning black image.");
        normalize_film(film, 1);
        return true;
    }

    ThreadEnvironment env;
    tbb::spin_mutex mutex;
    ref<ProgressReporter> progress = new ProgressReporter("Rendering");

    size_t n_cores = util::core_count();
    /* Multiply sample count by pixel count to obtain a similar scale
     * to the standard path tracer. */
    size_t total_samples =
        sensor->sampler()->sample_count() * hprod(film->size());
    // TODO: handle different modes
    size_t iteration_count = total_samples;
    // size_t iteration_count = IsVectorized
    //                              ? std::rint(total_samples / (Float) PacketSize)
    //                              : total_samples;
    size_t grain_count = std::max(
        (size_t) 1, (size_t) std::rint(iteration_count / Float(n_cores * 2)));

    // Insert default channels and set up the film
    std::vector<std::string> channels = { "X", "Y", "Z", "A", "W" };
    film->prepare(channels);
    bool has_aovs = false;
    if (has_aovs)
        Throw("Not supported yet: AOVs in LightTracerIntegrator");

    Log(Info, "Starting render job (%ix%i, %i sample%s, %i core%s)",
        film->crop_size().x(), film->crop_size().y(), total_samples,
        total_samples == 1 ? "" : "s", n_cores, n_cores == 1 ? "" : "s");
    if (m_timeout > 0.0f)
        Log(Info, "Timeout specified: %.2f seconds.", m_timeout);

    size_t total_samples_done = 0;
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, iteration_count, grain_count),
        [&](const tbb::blocked_range<size_t> &range) {
            ScopedSetThreadEnvironment set_env(env);

            // Thread-specific accumulators.
            // Enable warning, border and normalization for the image block.
            ref<ImageBlock> block = new ImageBlock(
                film->size(), channels.size(), film->reconstruction_filter(),
                /* warn_negative */ !has_aovs,
                /* warn_invalid */ true, /* border */ true,
                /* normalize */ true);
            size_t samples_done = 0;

            // Ensure that sample generation is fully deterministic.
            ref<Sampler> sampler = sensor->sampler()->clone();
            size_t seed          = (size_t) range.begin();
            sampler->seed(seed);

            for (auto i = range.begin(); i != range.end() && !should_stop();
                 ++i) {
                // Account for visible emitters
                sample_visible_emitters(scene, sensor, sampler, block);

                // Primary & further bounces illumination
                auto [ray, throughput] = prepare_ray(scene, sensor, sampler);

                // TODO(!): how to create a correct initial si?
                SurfaceInteraction3f si;
                trace_light_ray(ray, scene, sensor, sampler, si, throughput, /*depth*/ 0, block);

                // TODO: support other modes
                samples_done += 1;
                // samples_done += (enoki::is_array_v<Value> ? PacketSize : 1);
                if (samples_done > 10000) {
                    tbb::spin_mutex::scoped_lock lock(mutex);
                    total_samples_done += samples_done;
                    samples_done = 0;
                    progress->update(total_samples_done /
                                     (Float) total_samples);
                    // notify(block->bitmap());
                }
            }

            /* locked */ {
                tbb::spin_mutex::scoped_lock lock(mutex);
                total_samples_done += samples_done;

                film->put(block);
            }
        });

    // Apply proper normalization.
    auto new_weight = normalize_film(film, total_samples_done);
    Log(Info, "Processed %d samples, normalization weight: %f",
        total_samples_done, new_weight);

    return !m_stop;
}

MTS_VARIANT void LightTracerIntegrator<Float, Spectrum>::cancel() {
    m_stop = true;
}

MTS_VARIANT Spectrum
LightTracerIntegrator<Float, Spectrum>::sample_visible_emitters(
    Scene *scene, const Sensor *sensor, Sampler *sampler, ImageBlock *block,
    Mask active) const {

    Float time = sensor->shutter_open();
    if (sensor->shutter_open_time() > 0)
        time += sampler->next_1d(active) * sensor->shutter_open_time();

    Float idx_sample = sampler->next_1d(active);
    auto [emitter_idx, emitter_weight] =
        scene->sample_emitter(idx_sample, active);
    EmitterPtr emitter = enoki::gather<EmitterPtr>(scene->emitters().data(),
                                                   emitter_idx, active);

    Point2f emitter_sample = sampler->next_2d(active);
    auto [ps, pos_weight] =
        emitter->sample_position(time, emitter_sample, active);
    SurfaceInteraction3f si(ps, ek::empty<Wavelength>());

    Float wavelength_sample = sampler->next_1d(active);
    auto [wavelengths, wav_weight] =
        emitter->sample_wavelengths(si, wavelength_sample, active);
    si.wavelengths = wavelengths;
    si.shape       = emitter->shape();

    Spectrum weight = emitter_weight * pos_weight * wav_weight;

    // No BSDF passed (should not evaluate it since there's no scattering)
    return connect_sensor(scene, sensor, sampler, si, nullptr, weight, block, active);
}

MTS_VARIANT Float LightTracerIntegrator<Float, Spectrum>::normalize_block(
    ImageBlock *block, size_t total_samples) const {
    Float new_weight = total_samples / Float(hprod(block->size()));

    // Overwrite the weight channel
    auto &data         = block->data();
    size_t border      = block->border_size();
    size_t pixel_count = (block->width() + border) * (block->height() + border);
    size_t weight_channel = 4;
    Assert(block->channel_count() == 5);
    Assert(data.size() == block->channel_count() * pixel_count);
    enoki::scatter(data, new_weight,
                   block->channel_count() * enoki::arange<UInt64>(pixel_count) +
                       weight_channel);
    return new_weight;
}

MTS_VARIANT Float LightTracerIntegrator<Float, Spectrum>::normalize_film(
    Film *film, size_t total_samples) const {
    double new_weight = total_samples / (double) hprod(film->size());
    film->bitmap()->overwrite_channel(4, new_weight);
    return new_weight;
}

MTS_IMPLEMENT_CLASS_VARIANT(Integrator, Object, "integrator")
MTS_IMPLEMENT_CLASS_VARIANT(SamplingIntegrator, Integrator)
MTS_IMPLEMENT_CLASS_VARIANT(MonteCarloIntegrator, SamplingIntegrator)
MTS_IMPLEMENT_CLASS_VARIANT(LightTracerIntegrator, Integrator)

MTS_INSTANTIATE_CLASS(Integrator)
MTS_INSTANTIATE_CLASS(SamplingIntegrator)
MTS_INSTANTIATE_CLASS(MonteCarloIntegrator)
MTS_INSTANTIATE_CLASS(LightTracerIntegrator)
NAMESPACE_END(mitsuba)
