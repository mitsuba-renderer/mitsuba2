#include <mitsuba/core/properties.h>
#include <mitsuba/core/string.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/texture.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _bsdf-bumpmap:

Bump map BSDF adapter (:monosp:`bumpmap`)
-----------------------------------------

.. pluginparameters::

 * - bumpmap
   - |texture|
   - Specifies the bump map texture.
 * - (Nested plugin)
   - |bsdf|
   - A BSDF model that should be affected by the bump map
 * - strength
   - |float|
   - Bump map gradient multiplier. (Default: 1.0)
 * - epsilon
   - |float|
   - Finite difference delta to compute the bump map gradient. This is a
     multiplier on the pixel size extent in texture space (Default: 0.5)

Bump mapping is a simple technique for cheaply adding surface detail to a rendering. This is done
by perturbing the shading coordinate frame based on a displacement height field provided as a
texture. This method can lend objects a highly realistic and detailed appearance (e.g. wrinkled
or covered by scratches and other imperfections) without requiring any changes to the input geometry.

The implementation in Mitsuba uses the common approach of ignoring the usually negligible
texture-space derivative of the base mesh surface normal. As side effect of this decision,
it is invariant to constant offsets in the height field texture: only variations in its luminance
cause changes to the shading frame.

Note that the magnitude of the height field variations influences
the strength of the displacement.

.. subfigstart::
.. subfigure:: ../../resources/data/docs/images/render/bsdf_bumpmap_without.jpg
   :caption: Roughplastic BSDF
.. subfigure:: ../../resources/data/docs/images/render/bsdf_bumpmap_with.jpg
   :caption: Roughplastic BSDF with bump mapping
.. subfigend::
   :label: fig-bsdf-bumpmap

The following XML snippet describes a rough plastic material affected by a bump
map. Note the we set the ``raw`` properties of the bump map ``bitmap`` object to
``true`` in order to disable the transformation from sRGB to linear encoding:

.. code-block:: xml
    :name: bumpmap

    <bsdf type="bumpmap">
        <texture name="bumpmap" type="bitmap">
            <boolean name="raw" value="true"/>
            <string name="filename" value="textures/bumpmap.jpg"/>
        </texture>
        <bsdf type="roughplastic"/>
    </bsdf>

 */

template <typename Float, typename Spectrum>
class BumpMap final : public BSDF<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(BSDF, m_flags, m_components)
    MTS_IMPORT_TYPES(Texture)

    BumpMap(const Properties &props) : Base(props) {
        for (auto &kv : props.objects()) {
            auto bsdf = dynamic_cast<Base *>(kv.second.get());

            if (bsdf) {
                if (m_nested_bsdf)
                    Throw("Only a single BSDF child object can be specified.");
                m_nested_bsdf = bsdf;
            }
        }
        if (!m_nested_bsdf)
            Throw("Exactly one BSDF child object must be specified.");

        m_strength = props.float_("strength", 1.f);
        m_bumpmap  = props.texture<Texture>("bumpmap", 0.f);

        m_epsilon = props.float_("epsilon", .5f);

        m_components.clear();
        for (size_t i = 0; i < m_nested_bsdf->component_count(); ++i)
            m_components.push_back(m_nested_bsdf->flags(i));
        m_flags = m_nested_bsdf->flags();

        if (m_bumpmap->is_spatially_varying())
            m_flags = m_flags | BSDFFlags::NeedsDifferentials;
    }

    SurfaceInteraction3f evaluate_bump(const SurfaceInteraction3f &si,
                                       Mask active) const {

        SurfaceInteraction3f perturbed_si(si);

        // Compute pixel differential based uv delta for the gradient
        Float du = m_epsilon * max(abs(perturbed_si.duv_dx[0]),
                                   abs(perturbed_si.duv_dy[0]));
        Float dv = m_epsilon * max(abs(perturbed_si.duv_dx[1]),
                                   abs(perturbed_si.duv_dy[1]));

        // Might not have differentials (or be too far from camera...)
        du = max(0.0001f, du);
        dv = max(0.0001f, dv);

        // Save current uv
        Vector2f uv_org = perturbed_si.uv;

        // Compute bump map 3-tap gradient
        Float displace          = m_bumpmap->eval_1(perturbed_si, active);
        perturbed_si.uv[active] = uv_org + Vector2f(du, 0.f);
        Float u_displace        = m_bumpmap->eval_1(perturbed_si, active);
        perturbed_si.uv[active] = uv_org + Vector2f(0.f, dv);
        Float v_displace        = m_bumpmap->eval_1(perturbed_si, active);

        Float grad_u = m_strength * (u_displace - displace) / du;
        Float grad_v = m_strength * (v_displace - displace) / dv;

        // Restore uv
        perturbed_si.uv[active] = uv_org;

        // Compute surface differentials with map gradient
        Vector3f dp_du = select(
            active,
            fmadd(perturbed_si.sh_frame.n,
                  grad_u - dot(perturbed_si.sh_frame.n, perturbed_si.dp_du),
                  perturbed_si.dp_du),
            .0f);
        Vector3f dp_dv = select(
            active,
            fmadd(perturbed_si.sh_frame.n,
                  grad_v - dot(perturbed_si.sh_frame.n, perturbed_si.dp_dv),
                  perturbed_si.dp_dv),
            .0f);

        // Bump-mapped shading normal
        perturbed_si.sh_frame.n[active] = normalize(cross(dp_du, dp_dv));

        // Flip if not aligned with geometric normal
        perturbed_si.sh_frame
            .n[active &&
               (dot(perturbed_si.n, perturbed_si.sh_frame.n) < .0f)] *= -1.f;

        // Gram-schmidt orthogonalization to compute local shading frame
        perturbed_si.sh_frame.s[active] =
            normalize(fnmadd(perturbed_si.sh_frame.n,
                             dot(perturbed_si.sh_frame.n, perturbed_si.dp_du),
                             perturbed_si.dp_du));
        perturbed_si.sh_frame.t[active] =
            cross(perturbed_si.sh_frame.n, perturbed_si.sh_frame.s);

        // Express wi in the new the shading frame change
        perturbed_si.wi[active] = perturbed_si.to_local(si.to_world(si.wi));

        return perturbed_si;
    }

    std::pair<BSDFSample3f, Spectrum> sample(const BSDFContext &ctx,
                                             const SurfaceInteraction3f &si,
                                             Float sample1,
                                             const Point2f &sample2,
                                             Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFSample, active);

        // Sample nested BSDF with perturbed shading frame
        const SurfaceInteraction3f perturbed_si = evaluate_bump(si, active);
        auto [bs, weight] = m_nested_bsdf->sample(ctx, perturbed_si,
                                                  sample1, sample2, active);
        active &= any(neq(weight, 0.f));
        if (none(active))
            return { bs, 0.f };

        // Transform sampled 'wo' back to original frame and check orientation
        const Vector3f perturbed_wo = si.to_local(perturbed_si.to_world(bs.wo));
        active &= Frame3f::cos_theta(bs.wo) *
                  Frame3f::cos_theta(perturbed_wo) > 0.f;
        bs.wo = perturbed_wo;

        return { bs, weight & active };
    }

    Spectrum eval(const BSDFContext &ctx, const SurfaceInteraction3f &si,
                  const Vector3f &wo, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        // Evaluate nested BSDF with perturbed shading frame
        const SurfaceInteraction3f perturbed_si = evaluate_bump(si, active);

        Vector3f perturbed_wo = perturbed_si.to_local(si.to_world(wo));

        active &= Frame3f::cos_theta(wo) *
                  Frame3f::cos_theta(perturbed_wo) > 0.f;

        return m_nested_bsdf->eval(ctx, perturbed_si, perturbed_wo, active);
    }

    Float pdf(const BSDFContext &ctx, const SurfaceInteraction3f &si,
              const Vector3f &wo, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        // Evaluate nested BSDF with perturbed shading frame
        const SurfaceInteraction3f perturbed_si = evaluate_bump(si, active);

        const Vector3f perturbed_wo = perturbed_si.to_local(si.to_world(wo));

        active &= Frame3f::cos_theta(wo) *
                  Frame3f::cos_theta(perturbed_wo) > 0.f;

        return m_nested_bsdf->pdf(ctx, perturbed_si, perturbed_wo, active);
    }

    void traverse(TraversalCallback *callback) override {
        callback->put_object("nested_bsdf", m_nested_bsdf.get());
        callback->put_object("bumpmap", m_bumpmap.get());
        callback->put_parameter("strength", m_strength);
        callback->put_parameter("epsilon", m_epsilon);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "BumpMap[" << std::endl
            << "  nested_bsdf = " << string::indent(m_nested_bsdf) << std::endl
            << "  bumpmap = " << string::indent(m_bumpmap) << "," << std::endl
            << "  strength = " << string::indent(m_strength) << "," << std::endl
            << "  epsilon = " << string::indent(m_epsilon) << "," << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
protected:
    ScalarFloat m_strength;
    ScalarFloat m_epsilon;
    ref<Texture> m_bumpmap;
    ref<Base> m_nested_bsdf;
};

MTS_IMPLEMENT_CLASS_VARIANT(BumpMap, BSDF)
MTS_EXPORT_PLUGIN(BumpMap, "Bump map material adapter")
NAMESPACE_END(mitsuba)
