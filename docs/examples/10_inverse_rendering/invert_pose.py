import os
import time
import enoki as ek

import mitsuba
mitsuba.set_variant('gpu_autodiff_rgb')

from mitsuba.core import xml, Thread, Transform4f, Bitmap, Float, Vector3f, UInt32
from mitsuba.python.util import traverse
from mitsuba.python.autodiff import render, write_bitmap, Adam

# Return contiguous flattened array (will be included in next enoki release)
def ravel(buf):
    idx = 3 * UInt32.arange(int(len(buf) / 3))
    return Vector3f(ek.gather(buf, idx), ek.gather(buf, idx + 1), ek.gather(buf, idx + 2))

# Convert flat array into a vector of arrays (will be included in next enoki release)
def unravel(source, target, dim = 3):
    idx = UInt32.arange(ek.slices(source))
    for i in range(dim):
        ek.scatter(target, source[i], dim * idx + i)

# Prepare output folder
output_path = "output/invert_pose/"
if not os.path.isdir(output_path):
    os.makedirs(output_path)

# Load example scene
scene_folder = '../../../resources/data/docs/examples/invert_pose/'
Thread.thread().file_resolver().append(scene_folder)
scene = xml.load_file(scene_folder + 'scene.xml')

params = traverse(scene)
positions_buf = params['object.vertex_positions_buf']
positions_initial = ravel(positions_buf)

# Create differential parameter to be optimized
translate_ref = Vector3f(0.0)

# Create a new ParameterMap (or dict)
params_optim = {
    "translate" : translate_ref,
}

# Construct an Adam optimizer that will adjust the translation parameters
opt = Adam(params_optim, lr=0.02)

# Apply the transformation to mesh vertex position and update scene (e.g. Optix BVH)
def apply_transformation():
    trasfo = Transform4f.translate(params_optim["translate"])
    new_positions = trasfo.transform_point(positions_initial)
    unravel(new_positions, positions_buf)
    params['object.vertex_positions_buf'] = positions_buf
    params.update()

# Render a reference image (no derivatives used yet)
apply_transformation()
image_ref = render(scene, spp=32)
crop_size = scene.sensors()[0].film().crop_size()
write_bitmap(output_path + 'out_ref.exr', image_ref, crop_size)
print("Write " + output_path + "out_ref.exr")

# Move object before starting the optimization process
params_optim["translate"] = Vector3f(0.5, 0.2, -0.2)

time_a = time.time()

iterations = 100
for it in range(iterations):
    # Perform a differentiable rendering of the scene
    image = render(scene,
                   optimizer=opt,
                   spp=4,
                   unbiased=True,
                   pre_render_callback=apply_transformation)

    write_bitmap(output_path + 'out_%03i.exr' % it, image, crop_size)

    # Objective: MSE between 'image' and 'image_ref'
    ob_val = ek.hsum(ek.sqr(image - image_ref)) / len(image)

    # Back-propagate errors to input parameters
    ek.backward(ob_val)

    # Optimizer: take a gradient step -> update displacement map
    opt.step()

    print('Iteration %03i: error=%g' % (it, ob_val[0]), end='\r')

time_b = time.time()

print()
print('%f ms per iteration' % (((time_b - time_a) * 1000) / iterations))