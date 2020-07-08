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
        for (auto &[name, obj] : props.objects(false)) {
            auto bsdf = dynamic_cast<Base *>(obj.get());

            if (bsdf) {
                if (m_nested_bsdf)
                    Throw("Only a single BSDF child object can be specified.");
                m_nested_bsdf = bsdf;
                props.mark_queried(name);
            }
        }
        if (!m_nested_bsdf)
            Throw("Exactly one BSDF child object must be specified.");

        m_strength = props.float_("strength", 1.f);
        m_bumpmap  = props.texture<Texture>("bumpmap", 0.f);

        m_components.clear();
        for (size_t i = 0; i < m_nested_bsdf->component_count(); ++i)
            m_components.push_back(m_nested_bsdf->flags(i));
        m_flags = m_nested_bsdf->flags();
    }

    std::pair<BSDFSample3f, Spectrum> sample(const BSDFContext &ctx,
                                             const SurfaceInteraction3f &si,
                                             Float sample1,
                                             const Point2f &sample2,
                                             Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFSample, active);

        // Sample nested BSDF with perturbed shading frame
        SurfaceInteraction3f perturbed_si(si);
        perturbed_si.sh_frame = frame(si, active);
        perturbed_si.wi       = perturbed_si.to_local(si.to_world(si.wi));

        auto [bs, weight] = m_nested_bsdf->sample(ctx, perturbed_si,
                                                  sample1, sample2, active);
        active &= any(neq(depolarize(weight), 0.f));
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
        SurfaceInteraction3f perturbed_si(si);
        perturbed_si.sh_frame = frame(si, active);
        perturbed_si.wi       = perturbed_si.to_local(si.to_world(si.wi));
        Vector3f perturbed_wo = perturbed_si.to_local(si.to_world(wo));

        active &= Frame3f::cos_theta(wo) *
                  Frame3f::cos_theta(perturbed_wo) > 0.f;

        return m_nested_bsdf->eval(ctx, perturbed_si, perturbed_wo, active);
    }

    Float pdf(const BSDFContext &ctx, const SurfaceInteraction3f &si,
              const Vector3f &wo, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        // Evaluate nested BSDF pdf with perturbed shading frame
        SurfaceInteraction3f perturbed_si(si);
        perturbed_si.sh_frame = frame(si, active);
        perturbed_si.wi       = perturbed_si.to_local(si.to_world(si.wi));
        Vector3f perturbed_wo = perturbed_si.to_local(si.to_world(wo));

        active &= Frame3f::cos_theta(wo) *
                  Frame3f::cos_theta(perturbed_wo) > 0.f;

        return m_nested_bsdf->pdf(ctx, perturbed_si, perturbed_wo, active);
    }

    Frame3f frame(const SurfaceInteraction3f& si, Mask active) const {
        Vector2f grad_uv = m_strength * m_bumpmap->eval_1_grad(si, active);
        Vector3f dp_du =
            select(active,
                fmadd(si.sh_frame.n,
                    grad_uv.x() - dot(si.sh_frame.n, si.dp_du), si.dp_du),
                .0f);
        Vector3f dp_dv =
            select(active,
                fmadd(si.sh_frame.n,
                    grad_uv.y() - dot(si.sh_frame.n, si.dp_dv), si.dp_dv),
                .0f);

        // Bump-mapped shading normal
        Frame3f result;
        result.n[active] = normalize(cross(dp_du, dp_dv));

        // Flip if not aligned with geometric normal
        result.n[active && (dot(si.n, result.n) < .0f)] *= -1.f;

        // Gram-schmidt orthogonalization to compute local shading frame
        result.s[active] =
            normalize(fnmadd(result.n, dot(result.n, si.dp_du), si.dp_du));
        result.t[active] = cross(result.n, result.s);

        return result;
    }

    void traverse(TraversalCallback *callback) override {
        callback->put_object("nested_bsdf", m_nested_bsdf.get());
        callback->put_object("bumpmap", m_bumpmap.get());
        callback->put_parameter("strength", m_strength);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "BumpMap[" << std::endl
            << "  nested_bsdf = " << string::indent(m_nested_bsdf) << std::endl
            << "  bumpmap = " << string::indent(m_bumpmap) << "," << std::endl
            << "  strength = " << string::indent(m_strength) << "," << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
protected:
    ScalarFloat m_strength;
    ref<Texture> m_bumpmap;
    ref<Base> m_nested_bsdf;
};

MTS_IMPLEMENT_CLASS_VARIANT(BumpMap, BSDF)
MTS_EXPORT_PLUGIN(BumpMap, "Bump map material adapter")
NAMESPACE_END(mitsuba)
