import mitsuba
import pytest
import enoki as ek
import numpy as np

from mitsuba.python.test.util import fresolver_append_path

# Convert flat array into a vector of arrays (will be included in next enoki release)
def ravel(buf, dim = 3):
    from mitsuba.core import UInt32, Point2f, Point3f
    idx = dim * UInt32.arange(ek.slices(buf) // dim)
    if dim == 2:
        return Point2f(ek.gather(buf, idx), ek.gather(buf, idx + 1))
    elif dim == 3:
        return Point3f(ek.gather(buf, idx), ek.gather(buf, idx + 1), ek.gather(buf, idx + 2))

# Return contiguous flattened array (will be included in next enoki release)
def unravel(source, target, dim = 3):
    from mitsuba.core import UInt32
    idx = UInt32.arange(ek.slices(source))
    for i in range(dim):
        ek.scatter(target, source[i], dim * idx + i)


def write_gradient_image(grad, name):
    """Convert signed floats to blue/red gradient exr image"""
    from mitsuba.core import Bitmap

    convert_to_rgb = True

    if convert_to_rgb:
        # Compute RGB channels for .exr image (no grad = black)
        grad_R = grad.copy()
        grad_R[grad_R < 0] = 0.0
        grad_B = grad.copy()
        grad_B[grad_B > 0] = 0.0
        grad_B *= -1.0
        grad_G = grad.copy() * 0.0

        grad_np = np.concatenate((grad_R, grad_G, grad_B), axis=2)
    else:
        grad_np = np.concatenate((grad, grad, grad), axis=2)

    print('Writing', name + ".exr")
    Bitmap(grad_np).write(name + ".exr")


def render_gradient(scene, passes, diff_params):
    """Render radiance and gradient image using forward autodiff"""
    from mitsuba.python.autodiff import render

    fsize = scene.sensors()[0].film().size()

    img  = np.zeros((fsize[1], fsize[0], 3), dtype=np.float32)
    grad = np.zeros((fsize[1], fsize[0], 1), dtype=np.float32)
    for i in range(passes):
        img_i = render(scene)
        ek.forward(diff_params, i == passes - 1)

        grad_i = ek.gradient(img_i).numpy().reshape(fsize[1], fsize[0], -1)[:, :, [0]]
        img_i = img_i.numpy().reshape(fsize[1], fsize[0], -1)

        # Remove NaNs
        grad_i[grad_i != grad_i] = 0
        img_i[img_i != img_i] = 0

        grad += grad_i
        img += img_i

    return img / passes, grad / passes


def compute_groundtruth(make_scene, integrator, spp, passes, epsilon):
    """Render groundtruth radiance and gradient image using finite difference"""
    from mitsuba.python.autodiff import render

    def render_offset(offset):
        scene = make_scene(integrator, spp, offset)
        fsize = scene.sensors()[0].film().size()

        values = render(scene)
        for i in range(passes-1):
            values += render(scene)
        values /= passes

        return values.numpy().reshape(fsize[1], fsize[0], -1)

    gradient = (render_offset(epsilon) - render_offset(-epsilon)) / (2.0 * ek.norm(epsilon))

    image = render_offset(0.0)

    return image, gradient[:, :, [0]]


diff_integrator_default = { "type" : "pathreparam", "max_depth" : 2 }
ref_integrator_default = { "type" : "path", "max_depth" : 2 }


def check_finite_difference(test_name,
                            make_scene,
                            get_diff_params,
                            diff_integrator=diff_integrator_default,
                            diff_spp=4,
                            diff_passes=8,
                            ref_integrator=ref_integrator_default,
                            ref_spp=128,
                            ref_passes=10,
                            ref_eps=0.002,
                            error_threshold=0.05):
    """Compare resulting image and image gradient with finite difference method"""
    from mitsuba.core import Bitmap, Struct
    from mitsuba.python.autodiff import render

    # Render groundtruth image and gradients (using finite difference)
    img_ref, grad_ref = compute_groundtruth(make_scene, ref_integrator, ref_spp, ref_passes, ref_eps)

    ek.cuda_malloc_trim()

    scene = make_scene(diff_integrator, diff_spp, 0.0)
    fsize = scene.sensors()[0].film().size()
    img, grad = render_gradient(scene, diff_passes, get_diff_params(scene))

    error_img = np.abs(img_ref - img).mean()
    error_grad = np.abs(grad_ref - grad).mean() / grad_ref.max()

    if error_img > error_threshold:
        print("error_img:", error_img)
        Bitmap(img_ref).write('%s_img_ref.exr' % test_name)
        Bitmap(img).write('%s_img.exr' % test_name)
        assert False

    if error_grad > error_threshold:
        print("error_grad:", error_grad)
        scale = np.abs(grad_ref).max()
        write_gradient_image(grad_ref / scale, '%s_grad_ref' % test_name)
        write_gradient_image(grad / scale, '%s_grad' % test_name)
        Bitmap(img_ref).write('%s_img_ref.exr' % test_name)
        Bitmap(img).write('%s_img.exr' % test_name)
        assert False


def update_vertex_buffer(scene, object_name, diff_trafo):
    """Apply the given transformation to mesh vertex positions and call update scene"""
    from mitsuba.python.util import traverse

    params = traverse(scene)
    key = object_name + '.vertex_positions_buf'

    vertex_positions_buf = params[key]
    vertex_positions = ravel(vertex_positions_buf)

    vertex_positions_t = diff_trafo.transform_point(vertex_positions)

    unravel(vertex_positions_t, params[key])
    params.set_dirty(key)

    params.update()


# ----------------
# Template scene


mitsuba.set_variant("scalar_rgb")
from mitsuba.core import ScalarTransform4f

scene_template_dict = {
    "type" : "scene",
    "integrator" : { "type" : "path", "max_depth" : 2 },
    "sensor" : {
        "type" : "perspective",
        "fov_axis" : "smaller",
        "near_clip" : 0.1,
        "far_clip" : 2800,
        "focus_distance" : 1000,
        "fov" : 10,
        "to_world" : ScalarTransform4f.look_at([0, 0, 10], [0, 0, 0], [0, 1, 0]),
        "sampler" : {
            "type" : "independent",
            "sample_count" : 4
        },
        "film" : {
            "type" : "hdrfilm",
            "width" :  48,
            "height" : 48,
            "filter" : { "type" : "box" }
        }
    },
    "planemesh" :  {
        "type" : "obj",
        "id" : "planemesh",
        "to_world" : ScalarTransform4f.scale(2.0),
        "filename" : "resources/data/obj/xy_plane.obj",
        "bsdf" : { "type" : "diffuse" }
    },
    "light_shape" : {
        "type" : "obj",
        "id" : "light_shape",
        "to_world" : ScalarTransform4f.translate([10, 0, 15]) * ScalarTransform4f.rotate([1, 0, 0], 180),
        "filename" : "resources/data/obj/xy_plane.obj",
        "smooth_area_light" : {
            "type" : "smootharea",
            "radiance" : { "type": "spectrum", "value": 100 }
        }
    },
    "object" : {
        "type" : "obj",
        "id" : "object",
        "to_world" : ScalarTransform4f.translate([0, 0, 1.0]),
        "filename" : "resources/data/obj/smooth_empty_cube.obj"
    }
}


# ----------------
# Tests


# @pytest.mark.slow
# def test01_light_position(variant_gpu_autodiff_rgb):
#     from mitsuba.core import Float, Vector3f, Transform4f, ScalarTransform4f, ScalarVector3f, xml

#     if ek.cuda_mem_get_info()[1] < int(1e9):
#         pytest.skip('Insufficient GPU memory')

#     @fresolver_append_path
#     def make_scene(integrator, spp, param):
#         scene_dict = dict(scene_template_dict)
#         scene_dict["integrator"] = integrator
#         scene_dict["sensor"]["sampler"]["sample_count"] = spp
#         scene_dict["light_shape"]["to_world"] = ScalarTransform4f.translate(ScalarVector3f(10, 0, 15) + param) * ScalarTransform4f.rotate([1, 0, 0], 180)
#         return xml.load_dict(scene_dict)

#     def get_diff_param(scene):
#         diff_param = Float(0.0)
#         ek.set_requires_gradient(diff_param)
#         diff_trafo = Transform4f.translate(diff_param)
#         update_vertex_buffer(scene, 'light_shape', diff_trafo)
#         return diff_param

#     check_finite_difference("light_position", make_scene, get_diff_param)


# @pytest.mark.slow
# def test02_object_position(variant_gpu_autodiff_rgb):
#     from mitsuba.core import Float, Transform4f, ScalarTransform4f, ScalarVector3f, xml
#     from mitsuba.python.util import traverse

#     if ek.cuda_mem_get_info()[1] < int(1e9):
#         pytest.skip('Insufficient GPU memory')

#     @fresolver_append_path
#     def make_scene(integrator, spp, param):
#         scene_dict = dict(scene_template_dict)
#         scene_dict["integrator"] = integrator
#         scene_dict["sensor"]["sampler"]["sample_count"] = spp
#         scene_dict["object"]["to_world"] = ScalarTransform4f.translate(ScalarVector3f(0, 0, 1) + param)
#         return xml.load_dict(scene_dict)

#     def get_diff_param(scene):
#         diff_param = Float(0.0)
#         ek.set_requires_gradient(diff_param)
#         diff_trafo = Transform4f.translate(diff_param)
#         update_vertex_buffer(scene, 'object', diff_trafo)
#         return diff_param

#     check_finite_difference("object_position", make_scene, get_diff_param)


# @pytest.mark.slow
# def test03_object_rotation(variant_gpu_autodiff_rgb):
#     from mitsuba.core import Float, Transform4f, ScalarTransform4f, ScalarVector3f, xml
#     from mitsuba.python.util import traverse

#     if ek.cuda_mem_get_info()[1] < int(1e9):
#         pytest.skip('Insufficient GPU memory')

#     @fresolver_append_path
#     def make_scene(integrator, spp, param):
#         scene_dict = dict(scene_template_dict)
#         scene_dict["integrator"] = integrator
#         scene_dict["sensor"]["sampler"]["sample_count"] = spp
#         scene_dict["object"]["to_world"] = ScalarTransform4f.rotate([1, 0, 0], param)
#         return xml.load_dict(scene_dict)

#     def get_diff_param(scene):
#         diff_param = Float(0.0)
#         ek.set_requires_gradient(diff_param)
#         diff_trafo = Transform4f.rotate([1, 0, 0], diff_param)
#         update_vertex_buffer(scene, 'object', diff_trafo)
#         return diff_param

#     check_finite_difference("object_rotation", make_scene, get_diff_param, ref_eps=0.05)


# @pytest.mark.slow
# def test04_object_scaling(variant_gpu_autodiff_rgb):
#     from mitsuba.core import Float, Transform4f, ScalarTransform4f, xml
#     from mitsuba.python.util import traverse

#     if ek.cuda_mem_get_info()[1] < int(1e9):
#         pytest.skip('Insufficient GPU memory')

#     @fresolver_append_path
#     def make_scene(integrator, spp, param):
#         scene_dict = dict(scene_template_dict)
#         scene_dict["integrator"] = integrator
#         scene_dict["sensor"]["sampler"]["sample_count"] = spp
#         scene_dict["object"]["to_world"] = ScalarTransform4f.scale([1 + param, 1, 1])
#         return xml.load_dict(scene_dict)

#     def get_diff_param(scene):
#         diff_param = Float(0.0)
#         ek.set_requires_gradient(diff_param)
#         diff_trafo = Transform4f.scale([1 + diff_param, 1, 1])
#         update_vertex_buffer(scene, 'object', diff_trafo)
#         return diff_param

#     check_finite_difference("object_scaling", make_scene, get_diff_param)


@pytest.mark.slow
def test05_glossy_reflection(variant_gpu_autodiff_rgb):
    from mitsuba.core import Float, Transform4f, ScalarTransform4f, ScalarVector3f, xml
    from mitsuba.python.util import traverse

    if ek.cuda_mem_get_info()[1] < int(1e9):
        pytest.skip('Insufficient GPU memory')

    @fresolver_append_path
    def make_scene(integrator, spp, param):
        scene_dict = dict(scene_template_dict)
        scene_dict["integrator"] = integrator
        scene_dict["integrator"]["max_depth"] = 3
        scene_dict["sensor"]["sampler"]["sample_count"] = spp
        scene_dict["sensor"]["fov"] = 15
        scene_dict["planemesh"]["bsdf"] = { "type" : "roughconductor", "alpha" : 0.05 }
        scene_dict["planemesh"]["filename"] = "resources/data/obj/xy_plane_rough.obj"
        scene_dict["planemesh"]["to_world"] = ScalarTransform4f.rotate([1, 0, 0], -25)
        scene_dict["object"]["to_world"] = ScalarTransform4f.translate(ScalarVector3f(0, 0.6, 1) + param)
        return xml.load_dict(scene_dict)

    def get_diff_param(scene):
        diff_param = Float(0.0)
        ek.set_requires_gradient(diff_param)
        diff_trafo = Transform4f.translate(diff_param)
        update_vertex_buffer(scene, 'object', diff_trafo)
        return diff_param

    check_finite_difference("glossy_reflection", make_scene, get_diff_param, diff_passes=16, ref_eps=0.015)


# TODO fix this test
@pytest.mark.skip
@pytest.mark.slow
def test06_envmap(variant_gpu_autodiff_rgb):
    from mitsuba.core import Float, Transform4f, ScalarTransform4f, xml
    from mitsuba.python.util import traverse

    if ek.cuda_mem_get_info()[1] < int(1e9):
        pytest.skip('Insufficient GPU memory')

    @fresolver_append_path
    def make_scene(integrator, spp, param):
        scene_dict = dict(scene_template_dict)
        scene_dict["integrator"] = integrator
        scene_dict["sensor"]["sampler"]["sample_count"] = spp
        del scene_dict["light_shape"]
        scene_dict["envmap"] = {
            "type" : "envmap",
            "scale" : 1.0,
            "filename" : "resources/data/envmap/park.hdr",
            "to_world" : ScalarTransform4f.rotate([1, 0, 0], 90)
        }
        scene_dict["object"]["to_world"] = ScalarTransform4f.translate(param) * scene_dict["object"]["to_world"]
        return xml.load_dict(scene_dict)

    def get_diff_param(scene):
        diff_param = Float(0.0)
        ek.set_requires_gradient(diff_param)
        diff_trafo = Transform4f.translate(diff_param)
        update_vertex_buffer(scene, 'object', diff_trafo)
        return diff_param

    diff_integrator = {
        "type" : "pathreparam",
        "max_depth" : 2,
        "kappa_conv_envmap" : 10000000
    }

    check_finite_difference("envmap", make_scene, get_diff_param, diff_integrator=diff_integrator, error_threshold=0.1)

# TODO add tests for area+envmap