# Simple inverse rendering example: render a cornell box reference image, then
# then replace one of the scene parameters and try to recover it using
# differentiable rendering and gradient-based optimization.

import enoki as ek
import mitsuba
import numpy as np

import time

def load_scene():
    from mitsuba.core import ScalarTransform4f
    from mitsuba.core.xml import load_dict

    sensor_to_world = ScalarTransform4f.look_at(
        target=(0, 0, 0),
        origin=(1, -5, 0),
        up=(0, 0, 1)
    )
    sensor = {
        'type': 'perspective',
        'near_clip': 0.1,
        'far_clip': 100,
        'fov_axis': 'smaller',
        'fov': 60,
        'to_world': sensor_to_world,

        'sampler': {
            'type': 'independent',
            'sample_count': 512
        },
        'film': {
            'type': 'hdrfilm',
            'width': 256,
            'height': 256,
            'pixel_format': 'rgb',
            'rfilter': {
                # Important: reconstruction filter with a footprint
                # larger than 1 pixel.
                'type': 'gaussian'
            }
        },
    }
    integrator = {
        'type': 'ptracer',
        'samples_per_pass': 256,
        'max_depth': 4,
        'hide_emitters': False,
    }
    scene = {
        'type': 'scene',
        'sensor': sensor,
        'integrator': integrator,
        'shape': {
            'type': 'sphere',
            'shape-bsdf': {
                'type': 'diffuse',
                'reflectance': {
                    "type" : "rgb",
                    "value" : [0.5, 0.7, 0.9]
                },
            }
        },
        'emitter': {
            'type': 'constant',
            'radiance': 1.
        }
    }
    scene = load_dict(scene)
    return scene, scene.integrator(), scene.sensors()[0]

def main():
    from mitsuba.python.util import traverse
    from mitsuba.python.autodiff import render, write_bitmap, Adam

    scene, integrator, sensor = load_scene()

    crop_size = sensor.film().crop_size()
    integrator.render(scene, sensor)
    write_bitmap('test.exr', sensor.film().bitmap(), crop_size)

    params = traverse(scene)
    params.keep([
        'Sphere.bsdf.reflectance.value'
    ])
    opt = Adam(lr=.2, params=params)

    # Load or create the reference image
    image_ref = np.zeros(shape=(*crop_size, 3))
    image_ref[..., :2] = 0.5
    write_bitmap('out_ref.exr', image_ref, crop_size)

    start_time = time.time()
    iterations = 100
    for it in range(iterations):
        # Perform a differentiable rendering of the scene
        image = render(scene, optimizer=opt, unbiased=True, spp=1)
        write_bitmap('out_{:03d}.png'.format(it), image, crop_size)

        # Objective: MSE between 'image' and 'image_ref'
        loss = ek.hsum(ek.sqr(image - image_ref)) / len(image)

        # Back-propagate errors to input parameters
        ek.backward(loss)

        # Optimizer: take a gradient step
        opt.step()

        # Compare iterate against ground-truth value
        print('Iteration {:03d}: loss={:g}'.format(it, loss), end='\r')


    end_time = time.time()
    print()
    print('{:f} ms per iteration'.format(((end_time - start_time) * 1000) / iterations))


if __name__ == '__main__':
    mitsuba.set_variant('cuda_ad_rgb')
    # TODO: enable more options once stable
    ek.set_flags(ek.JitFlag.ADOptimize)
    main()
