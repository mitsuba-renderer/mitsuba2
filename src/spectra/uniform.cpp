#include <mitsuba/core/properties.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/texture.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _spectrum-uniform:

Uniform spectrum (:monosp:`uniform`)
------------------------------------

In its default form, this spectrum returns a constant reflectance or emission 
value between 360 and 830nm. When using spectral variants, the covered spectral 
interval can be specified tuned using its full XML specification; the plugin 
will return 0 outside of the covered spectral range.

.. pluginparameters::

 * - value
   - |float|
   - Returned value
 * - lambda_min
   - |float|
   - Lower bound of the covered spectral interval. Default: MTS_WAVELENGTH_MIN
 * - lambda_max
   - |float|
   - Upper bound of the covered spectral interval. Default: MTS_WAVELENGTH_MAX
 */

template <typename Float, typename Spectrum>
class UniformSpectrum final : public Texture<Float, Spectrum> {
public:
    MTS_IMPORT_TYPES(Texture)

    UniformSpectrum(const Properties &props)
        : Texture(props), m_value(1.f), m_lambda_min(MTS_WAVELENGTH_MIN),
          m_lambda_max(MTS_WAVELENGTH_MAX) {
        if (props.has_property("value"))
            m_value = props.float_("value");

        if (props.has_property("lambda_min"))
            m_lambda_min = max(props.float_("lambda_min"), MTS_WAVELENGTH_MIN);

        if (props.has_property("lambda_max"))
            m_lambda_max = min(props.float_("lambda_max"), MTS_WAVELENGTH_MAX);

        if (!(m_lambda_min < m_lambda_max))
            Throw(
                "UniformSpectrum: 'lambda_min' must be less than 'lambda_max'");
    }

    UnpolarizedSpectrum eval(const SurfaceInteraction3f &si,
                             Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);

        if constexpr (is_spectral_v<Spectrum>) {
            auto active_w = (si.wavelengths >= m_lambda_min) &&
                            (si.wavelengths <= m_lambda_max);

            return select(active_w, UnpolarizedSpectrum(m_value),
                          UnpolarizedSpectrum(0.f));
        } else {
            return m_value;
        }
    }

    Float eval_1(const SurfaceInteraction3f & /* it */,
                 Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);
        return m_value;
    }

    Wavelength pdf_spectrum(const SurfaceInteraction3f &si, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);

        if constexpr (is_spectral_v<Spectrum>) {
            auto active_w = (si.wavelengths >= m_lambda_min) &&
                            (si.wavelengths <= m_lambda_max);

            return select(active_w,
                          Wavelength(1.f / (m_lambda_max - m_lambda_min)),
                          Wavelength(0.f));
        } else {
            NotImplementedError("pdf_spectrum");
        }
    }

    std::pair<Wavelength, UnpolarizedSpectrum>
    sample_spectrum(const SurfaceInteraction3f & /*si*/,
                    const Wavelength &sample, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::TextureSample, active);

        if constexpr (is_spectral_v<Spectrum>) {
            return { m_lambda_min + (m_lambda_max - m_lambda_min) * sample,
                     m_value * (m_lambda_max - m_lambda_min) };
        } else {
            ENOKI_MARK_USED(sample);
            NotImplementedError("sample_spectrum");
        }
    }

    ScalarFloat mean() const override { return scalar_cast(hmean(m_value)); }

    void traverse(TraversalCallback *callback) override {
        callback->put_parameter("value", m_value);
        callback->put_parameter("lambda_min", m_lambda_min);
        callback->put_parameter("lambda_max", m_lambda_max);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "UniformSpectrum[" << std::endl
            << "  value = " << m_value << std::endl
            << "  lambda_min = " << m_lambda_min << std::endl
            << "  lambda_max = " << m_lambda_max << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    Float m_value;
    ScalarFloat m_lambda_min;
    ScalarFloat m_lambda_max;
};

MTS_IMPLEMENT_CLASS_VARIANT(UniformSpectrum, Texture)
MTS_EXPORT_PLUGIN(UniformSpectrum, "Uniform spectrum")
NAMESPACE_END(mitsuba)
