import mitsuba
import pytest
import enoki as ek
import numpy as np

from mitsuba.python.test.util import fresolver_append_path

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



diff_integrator_default = """<integrator type="pathreparam">
                                 <integer name="max_depth" value="2"/>
                             </integrator>"""

ref_integrator_default = """<integrator type="path">
                                <integer name="max_depth" value="2"/>
                            </integrator>"""


def check_finite_difference(test_name, make_scene, get_diff_params,
                            diff_integrator=diff_integrator_default, diff_spp=4, diff_passes=10,
                            ref_integrator=ref_integrator_default, ref_spp=128, ref_passes=10, ref_eps=0.001):

    from mitsuba.python.autodiff import render

    # Render groundtruth image and gradients (using finite difference)
    img_ref, grad_ref = compute_groundtruth(make_scene, ref_integrator, ref_spp, ref_passes, ref_eps)

    ek.cuda_malloc_trim()

    scene = make_scene(diff_integrator, diff_spp, 0.0)
    fsize = scene.sensors()[0].film().size()
    img, grad = render_gradient(scene, diff_passes, get_diff_params(scene))

    error_img = np.abs(img_ref - img).mean()
    error_grad = np.abs(grad_ref - grad).mean()

    if error_img > 0.1:
        print("error_img:", error_img)
        from mitsuba.core import Bitmap, Struct
        Bitmap(img_ref).write('%s_img_ref.exr' % test_name)
        Bitmap(img).write('%s_img.exr' % test_name)
        assert False

    if error_grad > 0.1:
        print("error_grad:", error_grad)
        scale = np.abs(grad_ref).max()
        write_gradient_image(grad_ref / scale, '%s_grad_ref' % test_name)
        write_gradient_image(grad / scale, '%s_grad' % test_name)
        assert False

# -----------------------------------------------------------------------
# -------------------------------- TESTS --------------------------------
# -----------------------------------------------------------------------

@pytest.mark.slow
def test01_light_position(variant_gpu_autodiff_rgb):
    from mitsuba.core import Float, Vector3f, Transform4f
    from mitsuba.core.xml import load_string
    from mitsuba.python.util import traverse

    @fresolver_append_path
    def make_scene(integrator, spp, param):
        return load_string("""
            <scene version="2.0.0">
                {integrator}
                <sensor type="perspective">
                    <string name="fov_axis" value="smaller"/>
                    <float name="near_clip" value="0.1"/>
                    <float name="far_clip" value="2800"/>
                    <float name="focus_distance" value="1000"/>
                    <transform name="to_world">
                        <lookat origin="0, 0, 10" target="0, 0, 0" up="0, 1, 0"/>
                    </transform>
                    <float name="fov" value="10"/>
                    <sampler type="independent">
                        <integer name="sample_count" value="{spp}"/>
                    </sampler>
                    <film type="hdrfilm">
                        <integer name="width" value="64"/>
                        <integer name="height" value="64"/>
                        <rfilter type="box"/>
                    </film>
                </sensor>

                <shape type="obj" id="light_shape">
                    <transform name="to_world">
                        <rotate x="1" angle="180"/>
                        <translate x="10.0" y="0.0" z="15.0"/>
                        <translate x="{param}" y="{param}" z="{param}"/>
                    </transform>
                    <string name="filename" value="resources/data/obj/xy_plane.obj"/>
                    <emitter type="smootharea" id="smooth_area_light">
                        <spectrum name="radiance" value="100"/>
                    </emitter>
                </shape>

                <shape type="obj" id="object">
                    <string name="filename" value="resources/data/obj/smooth_empty_cube.obj"/>
                    <transform name="to_world">
                        <translate z="1.0"/>
                    </transform>
                </shape>

                <shape type="obj" id="planemesh">
                    <string name="filename" value="resources/data/obj/xy_plane.obj"/>
                    <transform name="to_world">
                        <scale value="2.0"/>
                    </transform>
                </shape>
            </scene>
        """.format(integrator=integrator, spp=spp, param=param))

    def get_diff_param(scene):
        # Create a differentiable hyperparameter
        diff_param = Float(0.0)
        ek.set_requires_gradient(diff_param)

        # Update vertices so that they depend on diff_param
        params = traverse(scene)
        t = Transform4f.translate(Vector3f(1.0) * diff_param)
        vertex_positions = params['light_shape.vertex_positions']
        vertex_positions_t = t.transform_point(vertex_positions)
        params['light_shape.vertex_positions'] = vertex_positions_t

        # Update the scene
        params.update()

        return diff_param

    # Run the test
    check_finite_difference("light_position", make_scene, get_diff_param)


@pytest.mark.slow
def test02_object_position(variant_gpu_autodiff_rgb):
    from mitsuba.core import Float, Vector3f, Transform4f
    from mitsuba.core.xml import load_string
    from mitsuba.python.util import traverse

    @fresolver_append_path
    def make_scene(integrator, spp, param):
        return load_string("""
            <scene version="2.0.0">
                {integrator}
                <sensor type="perspective">
                    <string name="fov_axis" value="smaller"/>
                    <float name="near_clip" value="0.1"/>
                    <float name="far_clip" value="2800"/>
                    <float name="focus_distance" value="1000"/>
                    <transform name="to_world">
                        <lookat origin="0, 0, 10" target="0, 0, 0" up="0, 1, 0"/>
                    </transform>
                    <float name="fov" value="10"/>
                    <sampler type="independent">
                        <integer name="sample_count" value="{spp}"/>
                    </sampler>
                    <film type="hdrfilm">
                        <integer name="width" value="64"/>
                        <integer name="height" value="64"/>
                        <rfilter type="box"/>
                    </film>
                </sensor>

                <shape type="obj" id="light_shape">
                    <transform name="to_world">
                        <rotate x="1" angle="180"/>
                        <translate x="10.0" y="0.0" z="15.0"/>
                    </transform>
                    <string name="filename" value="resources/data/obj/xy_plane.obj"/>
                    <emitter type="smootharea" id="smooth_area_light">
                        <spectrum name="radiance" value="100"/>
                    </emitter>
                </shape>

                <shape type="obj" id="object">
                    <string name="filename" value="resources/data/obj/smooth_empty_cube.obj"/>
                    <transform name="to_world">
                        <translate z="1.0"/>
                        <translate x="{param}"/>
                    </transform>
                </shape>

                <shape type="obj" id="planemesh">
                    <string name="filename" value="resources/data/obj/xy_plane.obj"/>
                    <transform name="to_world">
                        <scale value="2.0"/>
                    </transform>
                </shape>
            </scene>
        """.format(integrator=integrator, spp=spp, param=param))

    def get_diff_param(scene):

        # Create a differentiable hyperparameter
        diff_param = mitsuba.core.Float(0.0)
        ek.set_requires_gradient(diff_param)

        # Update vertices so that they depend on diff_param
        properties = traverse(scene)
        t = mitsuba.core.Transform4f.translate(mitsuba.core.Vector3f(1.0,0.0,0.0) * diff_param)
        vertex_positions = properties['object.vertex_positions']
        vertex_positions_t = t.transform_point(vertex_positions)
        properties['object.vertex_positions'] = vertex_positions_t

        # Update the scene
        properties.update()

        return diff_param

    # Run the test
    check_finite_difference("object_position", make_scene, get_diff_param)

# TODO fix this test
@pytest.mark.skip
@pytest.mark.slow
def test03_envmap(variant_gpu_autodiff_rgb):
    from mitsuba.core import Float, Vector3f, Transform4f
    from mitsuba.core.xml import load_string
    from mitsuba.python.util import traverse

    @fresolver_append_path
    def make_scene(integrator, spp, param):
        return load_string("""
            <scene version="2.0.0">
                {integrator}
                <sensor type="perspective">
                    <string name="fov_axis" value="smaller"/>
                    <float name="near_clip" value="0.1"/>
                    <float name="far_clip" value="2800"/>
                    <float name="focus_distance" value="1000"/>
                    <transform name="to_world">
                        <lookat origin="0, 0, 10" target="0, 0, 0" up="0, 1, 0"/>
                    </transform>
                    <float name="fov" value="10"/>
                    <sampler type="independent">
                        <integer name="sample_count" value="{spp}"/>
                    </sampler>
                    <film type="hdrfilm">
                        <integer name="width" value="64"/>
                        <integer name="height" value="64"/>
                        <rfilter type="box"/>
                    </film>
                </sensor>

                <emitter type="envmap">
                    <float name="scale" value="1"/>
                    <string name="filename" value="resources/data/envmap/museum.exr"/>
                    <transform name="to_world">
                        <rotate x="1.0" angle="90"/>
                    </transform>
                </emitter>

                <shape type="obj" id="object">
                    <string name="filename" value="resources/data/obj/smooth_empty_cube.obj"/>
                    <transform name="to_world">
                        <translate z="0.6"/>
                        <translate x="{param}"/>
                    </transform>
                </shape>

                <shape type="obj" id="planemesh">
                    <string name="filename" value="resources/data/obj/xy_plane.obj"/>
                    <bsdf type="diffuse">
                        <rgb name="reflectance" value="0.8 0.8 0.8"/>
                    </bsdf>
                    <transform name="to_world">
                        <scale value="2.0"/>
                    </transform>
                </shape>
            </scene>
        """.format(integrator=integrator, spp=spp, param=param))

    def get_diff_param(scene):

        # Create a differentiable hyperparameter
        diff_param = mitsuba.core.Float(0.0)
        ek.set_requires_gradient(diff_param)

        # Update vertices so that they depend on diff_param
        properties = traverse(scene)
        t = mitsuba.core.Transform4f.translate(mitsuba.core.Vector3f(1.0, 0.0, 0.0) * diff_param)
        vertex_positions = properties['object.vertex_positions']
        vertex_positions_t = t.transform_point(vertex_positions)
        properties['object.vertex_positions'] = vertex_positions_t

        # Update the scene
        properties.update()

        return diff_param

    # Run the test

    diff_integrator = """<integrator type="pathreparam">
                             <integer name="max_depth" value="2"/>
                             <boolean name="use_convolution_envmap" value="true"/>
                         </integrator>"""

    check_finite_difference("envmap_conv", make_scene, get_diff_param, diff_integrator=diff_integrator)

    diff_integrator = """<integrator type="pathreparam">
                             <integer name="max_depth" value="2"/>
                             <boolean name="use_convolution_envmap" value="false"/>
                         </integrator>"""

    check_finite_difference("envmap_no_conv", make_scene, get_diff_param, diff_integrator=diff_integrator)

# TODO add tests for area+envmap, rotation, scaling, glossy reflection