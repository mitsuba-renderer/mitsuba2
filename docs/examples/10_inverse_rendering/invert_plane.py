# Simple inverse rendering example: render a cornell box reference image, then
# then replace one of the scene parameters and try to recover it using
# differentiable rendering and gradient-based optimization.

from os.path import join

import enoki as ek
import mitsuba
mitsuba.set_variant('gpu_autodiff_rgb')

from mitsuba.core import Thread, Color3f, Bitmap
from mitsuba.core.xml import load_file
from mitsuba.python.util import traverse
from mitsuba.python.autodiff import render, write_bitmap, Adam
import time

# Load the scene
#directory = '/home/merlinnd/Code/munich/scenes/glossy-plane'
directory = '/home/merlin/Code/mitsuba-data/glossy-plane'
Thread.thread().file_resolver().append(directory)
scene = load_file(join(directory, 'glossy-plane-master.xml'))

# Find differentiable scene parameters
params = traverse(scene)

# Discard all parameters except for one we want to differentiate
key = 'glossy.diffuse_reflectance.data'
print(params)
params.keep([key])

# Print the current value and keep a backup copy
param_ref = type(params[key])(params[key])
print(type(param_ref))

# Render a reference image (no derivatives used yet)
crop_size = scene.sensors()[0].film().crop_size()
if False:
    image_ref = render(scene, 1234, spp=128)
    write_bitmap('out_ref.exr', image_ref, crop_size)
else:
    import numpy as np
    image_ref = Bitmap('out_ref.exr')
    image_ref = mitsuba.core.Float(np.array(image_ref).ravel())

# Set initial value
params[key] = type(params[key]).full(0.5, ek.slices(params[key]))
params.update()

# Construct an Adam optimizer that will adjust the parameters 'params'
opt = Adam(params, lr=2e-2)

time_a = time.time()

iterations = 1000
start_avg_i = 250
avg_count = 0
avg_value = type(params[key]).full(0., ek.slices(params[key]))
sampler = scene.sensors()[0].sampler()

for it in range(iterations):
    seed = 2 * it

    # Perform a differentiable rendering of the scene
    image = render(scene, seed, optimizer=opt, unbiased=True, spp=32)
	
    if it % 10 == 0:
        write_bitmap('out_%03i.exr' % it, image, crop_size)

    # Objective: MSE between 'image' and 'image_ref'
    ob_val = ek.hsum(ek.sqr(image - image_ref)) / ek.slices(image)

    # Back-propagate errors to input parameters
    ek.backward(ob_val)

    # Optimizer: take a gradient step
    opt.step()

    # Compare iterate against ground-truth value
    err_ref = ek.hsum(ek.sqr(param_ref - params[key]))
    print('Iteration %03i: error=%g' % (it, err_ref[0]), end='\r')

    if it > start_avg_i:
        avg_value += ek.detach(params[key])
        avg_count += 1

time_b = time.time()

print()
print('%f ms per iteration' % (((time_b - time_a) * 1000) / iterations))

output = params[key]
write_bitmap('result.exr', output, [512, 512])
average = avg_value / avg_count
write_bitmap('average.exr', average, [512, 512])

