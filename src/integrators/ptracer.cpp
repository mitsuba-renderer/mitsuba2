#include <random>

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
    MTS_IMPORT_BASE(LightTracerIntegrator, m_max_depth, m_rr_depth)
    MTS_IMPORT_TYPES(Scene, Sensor, Sampler, ImageBlock, Medium, Emitter,
                     EmitterPtr, BSDF, BSDFPtr)

    explicit ParticleTracerIntegrator(const Properties &props) : Base(props) {}

    std::pair<Ray3f, Spectrum> prepare_ray(const Scene *scene,
                                           const Sensor *sensor,
                                           Sampler *sampler,
                                           Mask active) const override {
        Float time = sensor->shutter_open();
        if (sensor->shutter_open_time() > 0)
            time += sampler->next_1d(active) * sensor->shutter_open_time();

        /* ---------------------- Sample ray from emitter ------------------- */
        // Prepare random samples.
        auto wavelength_sample = sampler->next_1d(active);
        auto direction_sample  = sampler->next_2d(active);
        auto position_sample   = sampler->next_2d(active);
        // Sample one ray from an emitter in the scene.
        auto [ray, ray_weight, unused] = scene->sample_emitter_ray(
            time, wavelength_sample, direction_sample, position_sample, active);
        return std::make_pair(ray, ray_weight);
    }

    std::pair<Spectrum, Float>
    trace_light_ray(Ray3f ray, const Scene *scene, const Sensor *sensor,
                    Sampler *sampler, SurfaceInteraction3f &si,
                    Spectrum throughput, int depth, ImageBlock *block,
                    Mask active) const override {
        // Tracks radiance scaling due to index of refraction changes
        Float eta(1.f);
        BSDFContext ctx(TransportMode::Importance);

        // TODO: support ray differentials
        // ray.scale_differential(diff_scale_factor);

        /* ---------------------- Path construction ------------------------- */
        if (m_max_depth != -1 && depth >= m_max_depth)
            return { throughput, 1.f };
        // First intersection
        if (depth == 1)
            si = scene->ray_intersect(ray, active);

        // Incrementally build light path using BSDF sampling.
        for (;; ++depth) {
            active &= si.is_valid();

            // Russian Roulette
            if (depth > m_rr_depth) {
                Float q = ek::min(ek::hmax(throughput) * eta * eta, 0.95f);
                active &= sampler->next_1d(active) < q;
                throughput *= ek::rcp(q);
            }
            if (ek::none(active) || (uint32_t) depth >= (uint32_t) m_max_depth)
                break;

            // Connect to sensor and splat if successful.
            connect_sensor(scene, sensor, sampler, si, si.bsdf(ray), throughput,
                           block, active);

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

    Spectrum connect_sensor(const Scene *scene, const Sensor *sensor,
                            Sampler *sampler, const SurfaceInteraction3f &si,
                            const BSDF *bsdf, const Spectrum &weight,
                            ImageBlock *block, Mask active) const override {
        /* Query sensor for a direction connecting to `si.p`. This also gives
           us UVs on the sensor (for splatting). */
        auto [ds, sensor_val] = sensor->sample_direction(si, sampler->next_2d(), active);

        active &= (ds.pdf > 0.f) && any(neq(sensor_val, 0.f));
        Spectrum result = 0.f;
        if (ek::none_or<false>(active))
            return result;
        ds.uv[!active] = 0.f;

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
        auto sensor_ray = si.spawn_ray_to(ds.p);
        active &= !scene->ray_test(sensor_ray, active);
        if (ek::none_or<false>(active))
            return result;

        /* Foreshortening term and BSDF value for that direction (for surface interactions)
         * Note that foreshortening is only missing for directly visible emitters associated
         * with a shape (marked by convention by bsdf == nullptr). */
        Spectrum surface_weight = 1.f;
        auto on_surface         = active && neq(si.shape, nullptr);
        if (ek::any_or<true>(on_surface)) {
            auto local_d = si.to_local(sensor_ray.d);
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

        result = weight * sensor_val * surface_weight;

        // Splatting
        if (block != nullptr) {
            block->put(ds.uv, si.wavelengths, result, Float(1.f), active);
        }

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
};

MTS_IMPLEMENT_CLASS_VARIANT(ParticleTracerIntegrator, LightTracerIntegrator);
MTS_EXPORT_PLUGIN(ParticleTracerIntegrator, "Particle Tracer integrator");
NAMESPACE_END(mitsuba)
