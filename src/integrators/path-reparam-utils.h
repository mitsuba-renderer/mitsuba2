#include <random>
#include <enoki/stl.h>
#include <enoki/transform.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/core/frame.h>
#include <list>

NAMESPACE_BEGIN(mitsuba)

/* Helper for the correction factor in convolution sampling */

template<typename Float>
class VMFHemisphereIntegral {

public:

    VMFHemisphereIntegral() {
        m_k_res = 0;
        m_t_res = 0;

        // TODO clean this
        auto fs = Thread::thread()->file_resolver();
        fs::path file_path = fs->resolve("data/vmf-hemisphere.data");
        auto name = file_path.filename().string();

        auto fail = [&](const char *descr, auto... args) {
            Throw(("Error while loading file \"%s\": " + std::string(descr))
                      .c_str(), name, args...);
        };

        Log(Debug, "Loading data from \"%s\" ..", name);
        if (!fs::exists(file_path))
            fail("file not found");

        ref<MemoryMappedFile> mmap = new MemoryMappedFile(file_path);

        const char *ptr = (const char *) mmap->data();
        const char *eof = ptr + mmap->size();
        char buf[1025];

        size_t nb_data_line = 0;

        std::unique_ptr<float[]> tmp_data;

        while (ptr < eof) {
            // Determine the offset of the next newline
            const char *next = ptr;
            advance<false>(&next, eof, "\n");

            // Copy buf into a 0-terminated buffer
            size_t size = next - ptr;
            if (size >= sizeof(buf) - 1)
                fail("file contains an excessively long line! (%i characters)", size);
            memcpy(buf, ptr, size);
            buf[size] = '\0';

            // Skip whitespace
            const char *cur = buf, *eol = buf + size;
            advance<true>(&cur, eol, " \t\r");

            bool parse_error = false;

            if (cur[0] == 'R') {
                cur += 2;
                const char *next2;
                m_k_res = strtoul(cur, (char **) &next2, 10);
                cur = next2;
                m_t_res = strtoul(cur, (char **) &next2, 10);
                tmp_data = std::unique_ptr<float[]>(new float[m_k_res*m_t_res]);
            } else if (cur[0] != '#' && cur[0] != '\0') {
                if (m_t_res == 0)
                    fail("the resolution of the data must be specified before");

                if (nb_data_line >= m_k_res)
                    fail("too much data, found more than %i lines", m_k_res);

                for (size_t i = 0; i < m_t_res; ++i) {
                    const char *orig = cur;
                    float flt = strtof(cur, (char **) &cur);
                    parse_error |= cur == orig;
                    tmp_data[i + (nb_data_line) * m_k_res] = flt;
                }

                nb_data_line += 1;
            }

            if (unlikely(parse_error))
                fail("could not parse line \"%s\"", buf);

            ptr = next + 1;
        }

        m_data = DynamicBuffer<Float>::copy(&tmp_data[0], m_k_res * m_t_res);

        Log(Debug, "Loaded VMFHemisphereIntegral data, %ix%i.", m_k_res, m_t_res);

    }

    Float eval(float k, Float costheta, mask_t<Float> active) const {
        using UInt32   = uint32_array_t<Float>;
        using Point2u  = Point<UInt32, 2>;
        using Point2f  = Point<Float, 2>;
        using Vector2f = Vector<Float, 2>;
        using Vector2u = Vector<uint32_t, 2>;

        Vector2u size(m_t_res, m_k_res);

        Vector2f uv(costheta, mapping_K_U(Float(k)));

        uv = min(uv, 1.f);
        uv = max(uv, 0.f);
        uv *= Vector2f(size - 1u);

        Point2u pos = min(Point2u(uv), size - 2u);

        Point2f w1 = uv - Point2f(pos),
                w0 = 1.f - w1;

        UInt32 index = pos.x() + pos.y() * (uint32_t) size.x();

        uint32_t width = (uint32_t) size.x();
        Float v00 = gather<Float>(m_data, index, active);
        Float v10 = gather<Float>(m_data, index + 1u, active);
        Float v01 = gather<Float>(m_data, index + width, active);
        Float v11 = gather<Float>(m_data, index + width + 1u, active);

        Float s0 = fmadd(w0.x(), v00, w1.x() * v10),
              s1 = fmadd(w0.x(), v01, w1.x() * v11);

        return fmadd(w0.y(), s0, w1.y() * s1);
    }

    Float mapping_U_K(Float u) const {
        Float u_max = 6.f;
        return 0.1f * pow(10.f, u * u_max) - 0.1f;
    }

    Float mapping_K_U(Float k) const {
        Float u_max = 6.f;
        return log(10.f * k + 1.f) / (log(10.f) * u_max);
    }

private:

    template <bool Negate, size_t N>
    void advance(const char **start_, const char *end, const char (&delim)[N]) {
        const char *start = *start_;

        while (true) {
            bool is_delim = false;
            for (size_t i = 0; i < N; ++i)
                if (*start == delim[i])
                    is_delim = true;
            if ((is_delim ^ Negate) || start == end)
                break;
            ++start;
        }

        *start_ = start;
    }

    size_t m_k_res;
    size_t m_t_res;

    DynamicBuffer<Float> m_data;
};

// Helpers for duplicating data in large CUDA arrays

template <typename Value> Value concatD(const Value &a, const Value &b) {
    using T =
        std::conditional_t<is_static_array_v<Value>, value_t<Value>, Value>;
    using UInt = uint_array_t<T>;
    using Mask = mask_t<T>;
    if constexpr (is_cuda_array_v<Value>) {
        size_t N = slices(a);
        if (slices(a) != slices(b)) {
            Throw("DiffPathIntegrator::concatD: cannot concat arrays with "
                  "different sizes (not implemented).");
        }
        UInt index = arange<UInt>(N * 2);
        Mask m     = index < N;
        index      = select(m, index, index - N);
        return select(m, gather<Value>(a, index, m),
                         gather<Value>(b, index, !m));
    } else {
        Throw("DiffPathIntegrator::concatD: can only concat cuda arrays.");
    }
}

template <typename Value> Value makePairD(const Value &a) {
    using T =
        std::conditional_t<is_static_array_v<Value>, value_t<Value>, Value>;
    using UInt = uint_array_t<T>;
    using Mask = mask_t<T>;
    if constexpr (is_cuda_array_v<Value>) {
        size_t N = slices(a);
        if (N > 0) {
            UInt index = arange<UInt>(N * 2);
            Mask m     = index < N;
            index      = select(m, index, index - N);
            return gather<Value>(a, index);
        } else {
            return Value();
        }
    } else {
        Throw("DiffPathIntegrator::makePairD: can only makePairD cuda arrays.");
    }
}

// Helpers for sampling large CUDA Arrays

template <typename Float, typename Spectrum, typename Mask = mask_t<Float>>
Float samplePair1D(const Mask &m, Sampler<Float, Spectrum> *sampler) {
    size_t N = slices(m) / 2;
    using UInt = uint_array_t<Float>;
    UInt indices = arange<UInt>(N);
    Mask m0 = gather<Mask>(m, indices);
    Mask m1 = gather<Mask>(m, indices + N);
    Float sample = sampler->next_1d(m0 || m1);
    return makePairD(sample);
}

template <typename Float, typename Spectrum, typename Mask = mask_t<Float>,
          typename Point2 = Point<Float, 2>>
Point2 samplePair2D(const Mask &m, Sampler<Float, Spectrum> *sampler) {
    using UInt = uint_array_t<Float>;
    size_t N = slices(m) / 2;
    UInt indices = arange<UInt>(N);
    Mask m0 = gather<Mask>(m, indices);
    Mask m1 = gather<Mask>(m, indices + N);
    Point2 sample = sampler->next_2d(m0 || m1);
    return makePairD(sample);
}

template <typename Float, typename Spectrum, typename Mask = mask_t<Float>>
Float sample1D(const Mask &m, Sampler<Float, Spectrum> *sampler) {
    using UInt = uint_array_t<Float>;
    size_t N = slices(m) / 2;
    UInt indices = arange<UInt>(N);
    Mask m0 = gather<Mask>(m, indices);
    Mask m1 = gather<Mask>(m, indices + N);
    return concatD(sampler->next_1d(m0), sampler->next_1d(m1));
}

template <typename Float, typename Spectrum, typename Mask = mask_t<Float>,
          typename Point2 = Point<Float, 2>>
Point2 sample2D(const Mask &m, Sampler<Float, Spectrum> *sampler) {
    using UInt = uint_array_t<Float>;
    size_t N = slices(m) / 2;
    UInt indices = arange<UInt>(N);
    Mask m0 = gather<Mask>(m, indices);
    Mask m1 = gather<Mask>(m, indices + N);
    return concatD(sampler->next_2d(m0), sampler->next_2d(m1));
}

NAMESPACE_END(mitsuba)
