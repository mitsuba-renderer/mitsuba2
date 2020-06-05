import os
import time
import enoki as ek

import mitsuba
mitsuba.set_variant('gpu_autodiff_rgb')

from mitsuba.core import Thread, xml, UInt32, Float, Vector2f, Vector3f, Transform4f, ScalarTransform4f
from mitsuba.render import SurfaceInteraction3f
from mitsuba.python.util import traverse
from mitsuba.python.autodiff import render, write_bitmap, Adam

# Convert flat array into a vector of arrays (will be included in next enoki release)
def ravel(buf, dim = 3):
    idx = dim * UInt32.arange(ek.slices(buf) // dim)
    if dim == 2:
        return Vector2f(ek.gather(buf, idx), ek.gather(buf, idx + 1))
    elif dim == 3:
        return Vector3f(ek.gather(buf, idx), ek.gather(buf, idx + 1), ek.gather(buf, idx + 2))

# Return contiguous flattened array (will be included in next enoki release)
def unravel(source, target, dim = 3):
    idx = UInt32.arange(ek.slices(source))
    for i in range(dim):
        ek.scatter(target, source[i], dim * idx + i)

# Prepare output folder
output_path = "output/invert_heightfield/"
if not os.path.isdir(output_path):
    os.makedirs(output_path)

# Load example scene
scene_folder = '../../../resources/data/docs/examples/invert_heightfield/'
Thread.thread().file_resolver().append(scene_folder)
scene = xml.load_file(scene_folder + 'scene.xml')

params = traverse(scene)
positions_buf = params['grid_mesh.vertex_positions_buf']
positions_initial = ravel(positions_buf)
normals_initial = ravel(params['grid_mesh.vertex_normals_buf'])
vertex_count = ek.slices(positions_initial)

# Create a texture with the reference displacement map
disp_tex = xml.load_dict({
    "type" : "bitmap",
    "filename" : "mitsuba_coin.jpg",
    "to_uv" : ScalarTransform4f.scale([1, -1, 1]) # texture is upside-down
}).expand()[0]

# Create a fake surface interaction with an entry per vertex on the mesh
mesh_si = SurfaceInteraction3f.zero(vertex_count)
mesh_si.uv = ravel(params['grid_mesh.vertex_texcoords_buf'], dim=2)

# Evaluate the displacement map for the entire mesh
disp_tex_data_ref = disp_tex.eval_1(mesh_si)

# Apply displacement to mesh vertex positions and update scene (e.g. OptiX BVH)
def apply_displacement(amplitude = 0.05):
    new_positions = disp_tex.eval_1(mesh_si) * normals_initial * amplitude + positions_initial
    unravel(new_positions, params['grid_mesh.vertex_positions_buf'])
    params.set_dirty('grid_mesh.vertex_positions_buf')
    params.update()

# Apply displacement before generating reference image
apply_displacement()

# Render a reference image (no derivatives used yet)
image_ref = render(scene, spp=32)
crop_size = scene.sensors()[0].film().crop_size()
write_bitmap(output_path + 'out_ref.exr', image_ref, crop_size)
print("Write " + output_path + "out_ref.exr")

# Reset texture data to a constant
disp_tex_params = traverse(disp_tex)
disp_tex_params.keep(['data'])
disp_tex_params['data'] = ek.full(Float, 0.25, len(disp_tex_params['data']))
disp_tex_params.update()

# Construct an Adam optimizer that will adjust the texture parameters
opt = Adam(disp_tex_params, lr=0.002)

time_a = time.time()

iterations = 100
for it in range(iterations):
    # Perform a differentiable rendering of the scene
    image = render(scene,
                   optimizer=opt,
                   spp=4,
                   unbiased=True,
                   pre_render_callback=apply_displacement)

    write_bitmap(output_path + 'out_%03i.exr' % it, image, crop_size)

    # Objective: MSE between 'image' and 'image_ref'
    ob_val = ek.hsum(ek.sqr(image - image_ref)) / len(image)

    # Back-propagate errors to input parameters
    ek.backward(ob_val)

    # Optimizer: take a gradient step -> update displacement map
    opt.step()

    # Compare iterate against ground-truth value
    err_ref = ek.hsum(ek.sqr(disp_tex_data_ref - disp_tex.eval_1(mesh_si)))
    print('Iteration %03i: error=%g' % (it, err_ref[0]), end='\r')

time_b = time.time()

print()
print('%f ms per iteration' % (((time_b - time_a) * 1000) / iterations))
