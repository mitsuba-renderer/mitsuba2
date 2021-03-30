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
class ParticleTracerIntegrator : public LightTracerIntegrator<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(LightTracerIntegrator, m_samples_per_pass, m_hide_emitters,
                    m_rr_depth, m_max_depth)
    MTS_IMPORT_TYPES(Scene, Sensor, Film, Sampler, ImageBlock, Emitter,
                     EmitterPtr, BSDF, BSDFPtr)

    ParticleTracerIntegrator(const Properties &props) : Base(props) { }

    /**
     * Samples a ray from a random emitter in the scene.
     */
    std::pair<Ray3f, Spectrum> prepare_ray(const Scene *scene,
                                           const Sensor *sensor,
                                           Sampler *sampler,
                                           Mask active = true) const override {
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
                    ImageBlock *block, Mask active = true) const override {
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
            auto [bs, bsdf_val] =
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

            // TODO: shouldn't need this?
            ek::schedule(throughput, si, ray, eta, active);
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
                            ImageBlock *block, Mask active) const override {
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

    std::string to_string() const override {
        return tfm::format("ParticleTracerIntegrator[\n"
                           "  max_depth = %i,\n"
                           "  rr_depth = %i\n"
                           "]",
                           m_max_depth, m_rr_depth);
    }

    MTS_DECLARE_CLASS()

protected:
};

MTS_IMPLEMENT_CLASS_VARIANT(ParticleTracerIntegrator, LightTracerIntegrator);
MTS_EXPORT_PLUGIN(ParticleTracerIntegrator, "Particle Tracer integrator");
NAMESPACE_END(mitsuba)
