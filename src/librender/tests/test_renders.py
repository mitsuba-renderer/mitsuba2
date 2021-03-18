import os
from os.path import join, realpath, dirname, basename, splitext, exists
import argparse
import glob
import mitsuba
import pytest
import enoki as ek
import numpy as np
from enoki.scalar import ArrayXf as Float


color_modes = ['mono', 'rgb', 'spectral_polarized', 'spectral']


TEST_SCENE_DIR = realpath(join(os.path.dirname(
    __file__), '../../../resources/data/tests/scenes'))
scenes = glob.glob(join(TEST_SCENE_DIR, '*', '*.xml'))


# List of test scene folders to exclude
EXCLUDE_FOLDERS = [
    'orthographic_sensor', #TODO remove this after rebase onto master
]


# Don't test participating media in GPU modes
# to reduce the time needed to run all tests
JIT_EXCLUDE_FOLDERS = [
    'instancing', #TODO remove this once nested vcalls are supported
    'participating_media',
]


def get_ref_fname(scene_fname):
    for color_mode in color_modes:
        if color_mode in mitsuba.variant():
            ref_fname = join(dirname(scene_fname), 'refs', splitext(
                basename(scene_fname))[0] + '_ref_' + color_mode + '.exr')
            var_fname = ref_fname.replace('.exr', '_var.exr')
            return ref_fname, var_fname
    assert False


def xyz_to_rgb_bmp(arr):
    from mitsuba.core import Bitmap, Struct
    xyz_bmp = Bitmap(arr, Bitmap.PixelFormat.XYZ)
    return xyz_bmp.convert(Bitmap.PixelFormat.RGB, Struct.Type.Float32, False)


def read_rgb_bmp_to_xyz(fname):
    from mitsuba.core import Bitmap, Struct
    return Bitmap(fname).convert(Bitmap.PixelFormat.XYZ, Struct.Type.Float32, False)


def bitmap_extract(bmp):
    # AVOs from the moment integrator are in XYZ (float32)
    split = bmp.split()
    img = np.array(split[1][1], copy=False)
    img_m2 = np.array(split[2][1], copy=False)
    return img, img_m2 - img * img


def z_test(mean, sample_count, reference, reference_var):
    # Sanitize the variance images
    reference_var = np.maximum(reference_var, 1e-4)

    # Compute Z statistic
    z_stat = np.abs(mean - reference) * np.sqrt(sample_count / reference_var)

    # Cumulative distribution function of the standard normal distribution
    def stdnormal_cdf(x):
        shape = x.shape
        cdf = (1.0 - ek.erf(-Float(x.flatten()) / ek.sqrt(2.0))) * 0.5
        return np.array(cdf).reshape(shape)

    # Compute p-value
    p_value = 2.0 * (1.0 - stdnormal_cdf(z_stat))

    return p_value


if hasattr(ek, 'JitFlag'):
    jit_flags_options = [
        {ek.JitFlag.VCallRecord : 0, ek.JitFlag.VCallOptimize : 0, ek.JitFlag.LoopRecord : 0, ek.JitFlag.VCallBranch: 0},
        {ek.JitFlag.VCallRecord : 1, ek.JitFlag.VCallOptimize : 0, ek.JitFlag.LoopRecord : 0, ek.JitFlag.VCallBranch: 0},
        {ek.JitFlag.VCallRecord : 1, ek.JitFlag.VCallOptimize : 1, ek.JitFlag.LoopRecord : 1, ek.JitFlag.VCallBranch: 0},
        {ek.JitFlag.VCallRecord : 1, ek.JitFlag.VCallOptimize : 1, ek.JitFlag.LoopRecord : 0, ek.JitFlag.VCallBranch: 1},
    ]
else:
    jit_flags_options = ["dummy"]


@pytest.mark.slow
@pytest.mark.parametrize(*['scene_fname', scenes])
@pytest.mark.parametrize("jit_flags", jit_flags_options)
def test_render(variants_all, gc_collect, scene_fname, jit_flags):
    from mitsuba.core import Bitmap, Struct, Thread

    if hasattr(ek, 'JitFlag'):
        if 'scalar' in mitsuba.variant() and not jit_flags == jit_flags_options[0]:
            pytest.skip('no need to test the other jit flags in scalar mode')

        # Set enoki JIT flags
        for k, v in jit_flags.items():
            ek.set_flag(k, v)

    scene_dir = dirname(scene_fname)

    if os.path.split(scene_dir)[1] in EXCLUDE_FOLDERS:
        pytest.skip(f"Skip rendering scene {scene_fname}")

    is_jit = 'cuda' in mitsuba.variant() or 'llvm' in mitsuba.variant()
    if is_jit and os.path.split(scene_dir)[1] in JIT_EXCLUDE_FOLDERS:
        pytest.skip(f"Skip rendering scene {scene_fname} in JIT mode")

    ref_fname, ref_var_fname = get_ref_fname(scene_fname)
    if not (exists(ref_fname) and exists(ref_var_fname)):
        pytest.skip("Non-existent reference data.")

    ref_bmp = read_rgb_bmp_to_xyz(ref_fname)
    ref_img = np.array(ref_bmp, copy=False)

    ref_var_bmp = read_rgb_bmp_to_xyz(ref_var_fname)
    ref_var_img = np.array(ref_var_bmp, copy=False)

    significance_level = 0.01

    # Compute spp budget
    sample_budget = int(2e6)
    pixel_count = ek.hprod(ref_bmp.size())
    spp = sample_budget // pixel_count

    # Load and render
    scene = mitsuba.core.xml.load_file(scene_fname, spp=spp)
    scene.integrator().render(scene, scene.sensors()[0])

    # Compute variance image
    bmp = scene.sensors()[0].film().bitmap(raw=False)
    img, var_img = bitmap_extract(bmp)

    # Compute Z-test p-value
    p_value = z_test(img, spp, ref_img, ref_var_img)

    # Apply the Sidak correction term, since we'll be conducting multiple independent
    # hypothesis tests. This accounts for the fact that the probability of a failure
    # increases quickly when several hypothesis tests are run in sequence.
    alpha = 1.0 - (1.0 - significance_level) ** (1.0 / pixel_count)

    success = (p_value > alpha)

    if (np.count_nonzero(success) / 3) >= (0.9975 * pixel_count):
        print('Accepted the null hypothesis (min(p-value) = %f, significance level = %f)' %
              (np.min(p_value), alpha))
    else:
        print('Reject the null hypothesis (min(p-value) = %f, significance level = %f)' %
              (np.min(p_value), alpha))

        output_dir = join(scene_dir, 'error_output')

        if not exists(output_dir):
            os.makedirs(output_dir)

        output_prefix = join(output_dir, splitext(
            basename(scene_fname))[0] + '_' + mitsuba.variant())

        img_rgb_bmp = xyz_to_rgb_bmp(img)
        ref_img_rgb_bmp = xyz_to_rgb_bmp(ref_img)

        fname = output_prefix + '_img.exr'
        img_rgb_bmp.write(fname)
        print('Saved rendered image to: ' + fname)

        fname = output_prefix + '_ref.exr'
        ref_img_rgb_bmp.write(fname)
        print('Saved reference image to: ' + fname)

        var_fname = output_prefix + '_var.exr'
        xyz_to_rgb_bmp(var_img).write(var_fname)
        print('Saved variance image to: ' + var_fname)

        err_fname = output_prefix + '_error.exr'
        err_img = 0.02 * np.array(img_rgb_bmp)
        err_img[~success] = 1.0
        err_bmp = Bitmap(err_img).write(err_fname)
        print('Saved error image to: ' + err_fname)

        pvalue_fname = output_prefix + '_pvalue.exr'
        xyz_to_rgb_bmp(p_value).write(pvalue_fname)
        print('Saved error image to: ' + pvalue_fname)

        assert False


if __name__ == '__main__':
    """
    Generate reference images for all the scenes contained within the TEST_SCENE_DIR directory,
    and for all the color mode having their `scalar_*` mode enabled.
    """
    parser = argparse.ArgumentParser(prog='RenderReferenceImages')
    parser.add_argument('--overwrite', action='store_true',
                        help='Force rerendering of all reference images. Otherwise, only missing references will be rendered.')
    parser.add_argument('--spp', default=32000, type=int,
                        help='Samples per pixel. Default value: 32000')
    args = parser.parse_args()

    ref_spp = args.spp
    overwrite = args.overwrite

    for scene_fname in scenes:
        scene_dir = dirname(scene_fname)

        if os.path.split(scene_dir)[1] in EXCLUDE_FOLDERS:
            continue

        for variant in mitsuba.variants():
            if not variant.split('_')[0] == 'scalar' or variant.endswith('double'):
                continue

            mitsuba.set_variant(variant)
            from mitsuba.core import Bitmap, Struct, Thread

            ref_fname, var_fname = get_ref_fname(scene_fname)
            if exists(ref_fname) and exists(var_fname) and not overwrite:
                continue

            scene = mitsuba.core.xml.load_file(scene_fname, spp=ref_spp)
            scene.integrator().render(scene, scene.sensors()[0])

            bmp = scene.sensors()[0].film().bitmap(raw=False)
            img, var_img = bitmap_extract(bmp)

            # Write rendered image to a file
            os.makedirs(dirname(ref_fname), exist_ok=True)
            xyz_to_rgb_bmp(img).write(ref_fname)
            print('Saved rendered image to: ' + ref_fname)

            # Write variance image to a file
            xyz_to_rgb_bmp(var_img).write(var_fname)
            print('Saved variance image to: ' + var_fname)
