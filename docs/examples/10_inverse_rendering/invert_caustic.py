# Simple inverse rendering example: render a cornell box reference image, then
# then replace one of the scene parameters and try to recover it using
# differentiable rendering and gradient-based optimization.

import os
import time

import enoki as ek
import mitsuba
import numpy as np


SCENE_DIR = os.path.realpath(os.path.join(
    os.path.dirname(__file__), '../../../resources/data/scenes/caustic-optimization'))

def load_scene():
    from mitsuba.core import ScalarTransform4f, Thread, Bitmap
    from mitsuba.core.xml import load_dict

    fr = Thread.thread().file_resolver()
    fr.append(SCENE_DIR)

    # Looking at the receiving plane, not looking through the lens
    sensor_to_world = ScalarTransform4f.look_at(
        target=(0, -20, 0),
        origin=(0, -4.65, 0),
        up=(0, 0, 1)
    )
    sensor = {
        'type': 'perspective',
        'near_clip': 1,
        'far_clip': 1000,
        'fov': 45,
        'to_world': sensor_to_world,

        'sampler': {
            'type': 'independent',
            'sample_count': 512  # Not really used
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
        # Glass BSDF
        'simple-glass': {
            'type': 'dielectric',
            'id': 'simple-glass-bsdf',
            'ext_ior': 'air',
            'int_ior': 1.5,
            'specular_reflectance': { 'type': 'spectrum', 'value': 0 },
        },
        'white-bsdf': {
            'type': 'diffuse',
            'id': 'white-bsdf',
            'reflectance': { 'type': 'rgb', 'value': (1, 1, 1) },
        },
        'black-bsdf': {
            'type': 'diffuse',
            'id': 'black-bsdf',
            'reflectance': { 'type': 'spectrum', 'value': 0 },
        },

        # Receiving plane
        'receiving-plane': {
            'type': 'obj',
            'id': 'receiving-plane',
            'filename': 'meshes/rectangle.obj',
            'to_world': \
                ScalarTransform4f.look_at(
                    target=(0, 1, 0),
                    origin=(0, -7, 0),
                    up=(0, 0, 1)
                ) \
                * ScalarTransform4f.scale((5, 5, 5)),
            'bsdf': {'type': 'ref', 'id': 'white-bsdf'},
        },
        # Glass slab, excluding the 'exit' face (added separately below)
        'slab': {
            'type': 'obj',
            'id': 'slab',
            'filename': 'meshes/slab.obj',
            'to_world': ScalarTransform4f.rotate(axis=(1, 0, 0), angle=90),
            'bsdf': {'type': 'ref', 'id': 'simple-glass'},
        },
        # Glass rectangle, to be optimized
        'lens': {
            'type': 'obj',
            'id': 'lens',
            'filename': 'meshes/rectangle-fine.obj',  # TODO: resolution
            'to_world': ScalarTransform4f.rotate(axis=(1, 0, 0), angle=90),
            'bsdf': {'type': 'ref', 'id': 'simple-glass'},
        },

        # Directional area emitter placed behind the glass slab
        # TODO: try with a spotlight emitter
        'focused-emitter-shape': {
            'type': 'obj',
            'filename': 'meshes/rectangle.obj',
            'to_world': ScalarTransform4f.look_at(
                target=(0, 0, 0),
                origin=(0, 5, 0),
                up=(0, 0, 1)
            ),
            'bsdf': {'type': 'ref', 'id': 'black-bsdf'},
            'focused-emitter': {
                'type':'directionalarea',
                'radiance': {'type': 'spectrum', 'value': 0.8},
            },
        },
    }
    scene = load_dict(scene)
    return scene, scene.integrator(), scene.sensors()[0]

def load_ref_image(resolution, output_dir):
    from mitsuba.python.autodiff import write_bitmap

    if False:
        image_ref = np.zeros(shape=(*resolution, 3), dtype=np.float32)
        image_ref[..., 1:] = 0.5
    else:
        from mitsuba.core import Bitmap

        b = Bitmap(os.path.join(SCENE_DIR, 'references/wave-1024.exr'))
        b = b.convert(Bitmap.PixelFormat.RGB, Bitmap.Float32, False)
        if b.size() != resolution:
            # TODO(!): how to avoid this mode change?
            #          `Bitmap::resample` doesn't work without it.
            bak = mitsuba.variant()
            mitsuba.set_variant('scalar_rgb')
            b = b.resample(resolution)
            mitsuba.set_variant(bak)

        image_ref = np.array(b)

    write_bitmap(os.path.join(output_dir, 'out_ref.exr'), image_ref, resolution)
    return mitsuba.core.Float(image_ref.ravel())

def main():
    from mitsuba.core import Bitmap, Vector3f
    from mitsuba.core.xml import load_dict
    from mitsuba.render import SurfaceInteraction3f
    from mitsuba.python.util import traverse
    from mitsuba.python.autodiff import render, write_bitmap, Adam

    scene, integrator, sensor = load_scene()

    crop_size = sensor.film().crop_size()

    # Heightmap (displacement texture) that will actually be optimized
    texture_res = (2048, 2048)
    heightmap_texture = load_dict({
        'type': 'bitmap',
        'id': 'heightmap_texture',
        'bitmap': Bitmap(np.zeros(texture_res, dtype=np.float32)),
        'raw': True,
    }).expand()[0]

    params_scene = traverse(scene)
    # We will always apply displacements along the original normals
    positions_initial = ek.unravel(Vector3f, params_scene['lens.vertex_positions'])
    normals_initial = ek.unravel(Vector3f, params_scene['lens.vertex_normals'])
    lens_si = ek.zero(SurfaceInteraction3f, ek.width(positions_initial))
    lens_si.uv = ek.unravel(type(lens_si.uv), params_scene['lens.vertex_texcoords'])

    def apply_displacement(amplitude = 1.):
        # Enforce reasonable range
        # TODO: range based on scene scale (mm)
        params['data'] = ek.clamp(params['data'], -0.5, 0.5)
        ek.enable_grad(params['data'])

        new_positions = (heightmap_texture.eval_1(lens_si) * normals_initial * amplitude
                         + positions_initial)
        params_scene['lens.vertex_positions'] = ek.ravel(new_positions)
        params_scene.set_dirty('lens.vertex_positions')
        params_scene.update()

    # Actually optimized: the heightmap texture
    params = traverse(heightmap_texture)
    params.keep(['data'])
    opt = Adam(lr=5e-5, params=params)
    opt.load()

    # Load or create the reference image
    image_ref = load_ref_image(crop_size, output_dir='.')

    start_time = time.time()
    iterations = 500
    for it in range(iterations):
        t0 = time.time()

        apply_displacement()

        # Perform a differentiable rendering of the scene
        # TODO: support unbiased=True
        image = render(scene, optimizer=opt, unbiased=False, spp=32)
        # image = render(scene, optimizer=opt, unbiased=True, spp=32)

        if it % 5 == 0:
            write_bitmap('out_{:03d}.exr'.format(it), image, crop_size)

        # Objective: MSE between 'image' and 'image_ref'
        loss = ek.hsum_async(ek.sqr(image - image_ref)) / len(image)

        # Back-propagate errors to input parameters and take an optimizer step
        ek.backward(loss)
        opt.step()

        # TODO: shouldn't need to do this (?)
        opt.update()
        sensor.sampler().schedule_state()

        elapsed_ms = 1000. * (time.time() - t0)
        print('Iteration {:03d}: loss={:g} (took {:.0f}ms)'.format(
            it, loss[0], elapsed_ms), end='\r')


    end_time = time.time()
    print()
    print('{:f} ms per iteration'.format(((end_time - start_time) * 1000) / iterations))


if __name__ == '__main__':
    mitsuba.set_variant('cuda_ad_rgb')
    # TODO: enable more options once stable
    ek.set_flags(ek.JitFlag.ADOptimize)
    main()
