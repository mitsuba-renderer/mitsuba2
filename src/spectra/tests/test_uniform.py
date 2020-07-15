import pytest

import enoki as ek
import mitsuba


def make_spectrum(value=None, lambda_min=None, lambda_max=None):
    from mitsuba.core.xml import load_dict

    spectrum_dict = {"type": "uniform"}
    
    if value is not None:
        spectrum_dict["value"] = value
    if lambda_min is not None:
        spectrum_dict["lambda_min"] = lambda_min
    if lambda_max is not None:
        spectrum_dict["lambda_max"] = lambda_max
    return load_dict(spectrum_dict)


def test_construct(variant_scalar_spectral):
    assert make_spectrum() is not None
    assert make_spectrum(value=2.) is not None
    assert make_spectrum(value=2., lambda_min=400., lambda_max=500.) is not None

    with pytest.raises(RuntimeError):
        make_spectrum(lambda_min=500., lambda_max=400.)


def test_eval(variant_scalar_spectral):
    from mitsuba.render import SurfaceInteraction3f
    si = SurfaceInteraction3f()

    value = 2.
    lambda_min = 400.
    lambda_max = 500.
    dlambda = lambda_max - lambda_min

    s = make_spectrum(value, lambda_min, lambda_max)
    wavelengths = [390., 400., 450., 510.]
    values = [0, value, value, 0]

    si.wavelengths = wavelengths
    assert ek.allclose(s.eval(si), values)
    assert ek.allclose(
        s.pdf_spectrum(si), [1. / dlambda if value else 0. for value in values])

    assert ek.allclose(s.eval_1(si), value)

    with pytest.raises(RuntimeError) as excinfo:
        s.eval_3(si)
    assert 'not implemented' in str(excinfo.value)


def test_sample_spectrum(variant_scalar_spectral):
    from mitsuba.render import SurfaceInteraction3f
    from mitsuba.core import MTS_WAVELENGTH_MIN, MTS_WAVELENGTH_MAX
    dlambda = MTS_WAVELENGTH_MAX - MTS_WAVELENGTH_MIN

    value = 0.5
    s = make_spectrum(value)

    si = SurfaceInteraction3f()
    assert ek.allclose(s.sample_spectrum(si, 0), [MTS_WAVELENGTH_MIN, value * dlambda])
    assert ek.allclose(s.sample_spectrum(si, .5), [
                       MTS_WAVELENGTH_MIN + .5 * dlambda, value * dlambda])
    assert ek.allclose(s.sample_spectrum(si, 1), [MTS_WAVELENGTH_MAX, value * dlambda])
