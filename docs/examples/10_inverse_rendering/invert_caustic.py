# Simple inverse rendering example: render a cornell box reference image, then
# then replace one of the scene parameters and try to recover it using
# differentiable rendering and gradient-based optimization.

import os
from os.path import join, realpath, dirname
import time

import enoki as ek
import mitsuba
import numpy as np


SCENE_DIR = os.path.realpath(os.path.join(
    os.path.dirname(__file__), '../../../resources/data/scenes/caustic-optimization'))
CONFIGS = {
    'wave': {
        'emitter': 'gray',
        'reference': os.path.join(SCENE_DIR, 'references/wave-1024.exr'),
        'render_resolution': (512, 512),
        'heightmap_resolution': (4096, 4096),
        'spp': 8,
        'max_iterations': 500,
        'learning_rate': 5e-5,
    },
    'sunday': {
        'emitter': 'bayer',
        'reference': os.path.join(SCENE_DIR, 'references/sunday-512.exr'),
        # 'render_resolution': (256, 256),
        # 'heightmap_resolution': (1024, 1024),
        # 'spp': 16,
        'render_resolution': (512, 512),
        'heightmap_resolution': (2048, 2048),
        'spp': 32,
        'max_iterations': 500,
        'learning_rate': 2e-5,
    },
}


def load_scene(config, output_dir):
    from mitsuba.core import ScalarTransform4f, Thread, Bitmap
    from mitsuba.core.xml import load_dict

    fr = Thread.thread().file_resolver()
    fr.append(SCENE_DIR)

    lens_res = config.get('lens_res', config['heightmap_resolution'])
    lens_fname = join(output_dir, 'lens_{}_{}.ply'.format(*lens_res))
    if not os.path.isfile(lens_fname):
        m = create_flat_lens_mesh(lens_res)
        m.write_ply(lens_fname)
        print('[+] Wrote lens mesh ({}x{} tesselation) file to: {}'.format(*lens_res, lens_fname))

    emitter = None
    if config['emitter'] == 'gray':
        emitter = {
            'type':'directionalarea',
            'radiance': {'type': 'spectrum', 'value': 0.8},
        }
    elif config['emitter'] == 'bayer':
        bayer = np.zeros(shape=(32, 32, 3))
        bayer[::2, ::2, 2] = 2
        bayer[::2, 1::2, 1] = 2
        # bayer[1::2, ::2, 1] = 1  # Avoid too much green
        bayer[1::2, 1::2, 0] = 2

        bayer = Bitmap(bayer.astype(np.float32), pixel_format=Bitmap.PixelFormat.RGB)
        # bayer.write('emitter.exr')
        emitter = {
            'type':'directionalarea',
            'radiance': {
                'type': 'bitmap',
                'bitmap': bayer,
                'raw': True,
                'filter_type': 'nearest'
            },
        }
    # TODO: suppport spotlight emitter type
    assert emitter is not None

    # Looking at the receiving plane, not looking through the lens
    sensor_to_world = ScalarTransform4f.look_at(
        target=(0, -20, 0),
        origin=(0, -4.65, 0),
        up=(0, 0, 1)
    )
    resx, resy = config['render_resolution']
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
            'width': resx,
            'height': resy,
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
            'type': 'ply',
            'id': 'lens',
            'filename': lens_fname,
            'to_world': ScalarTransform4f.rotate(axis=(1, 0, 0), angle=90),
            'bsdf': {'type': 'ref', 'id': 'simple-glass'},
        },

        # Directional area emitter placed behind the glass slab
        'focused-emitter-shape': {
            'type': 'obj',
            'filename': 'meshes/rectangle.obj',
            'to_world': ScalarTransform4f.look_at(
                target=(0, 0, 0),
                origin=(0, 5, 0),
                up=(0, 0, 1)
            ),
            'bsdf': {'type': 'ref', 'id': 'black-bsdf'},
            'focused-emitter': emitter,
        },
    }
    scene = load_dict(scene)
    return scene, scene.integrator(), scene.sensors()[0]

def load_ref_image(config, resolution, output_dir):
    from mitsuba.core import Bitmap
    from mitsuba.python.autodiff import write_bitmap

    b = Bitmap(config['reference'])
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


def create_flat_lens_mesh(resolution):
    from mitsuba.core import Float, UInt32, ScalarTransform4f
    from mitsuba.render import Mesh

    assert resolution[0] == resolution[1], 'Not supported yet: non-symmetric lens mesh resolution'
    X, Y = np.mgrid[:resolution[0], :resolution[1]]
    # Scale UVs to [-1, 1]
    U = X / (resolution[0]-1)
    V = Y / (resolution[1]-1)
    # Scale actual positions to [-1, 1]
    X = 2 * (U - 0.5)
    Y = 2 * (V - 0.5)

    n_vertices = resolution[0] * resolution[1]

    vertices = np.stack([
        X.ravel(), Y.ravel(), np.zeros((n_vertices))
    ], axis=1)
    # # Rotate to match the `slab.obj` positioning
    # to_world = ScalarTransform4f.rotate(axis=(1, 0, 0), angle=90)
    # to_world = np.array(to_world.matrix)[:3, :3]
    # vertices = (to_world @ vertices.T).T
    assert vertices.shape == (n_vertices, 3)

    texcoords =  np.stack([
        U.ravel(), V.ravel()
    ], axis=1)
    assert texcoords.shape == (n_vertices, 2)

    # Create two triangles per grid cell
    faces = []
    for i in range(resolution[0] - 1):
        for j in range(resolution[1] - 1):
            v00 = i * resolution[1] + j
            v01 = v00 + 1
            v10 = (i + 1) * resolution[1] + j
            v11 = v10 + 1

            faces.append((v00, v10, v01))
            faces.append((v01, v10, v11))
    n_faces = len(faces)
    faces = np.array(faces).astype(np.uint32)
    assert faces.shape == (n_faces, 3)

    m = Mesh("lens-mesh", n_vertices, n_faces, has_vertex_texcoords=True)
    ek.scatter(
        m.vertex_positions_buffer(),
        Float(vertices.ravel().astype(np.float32)),
        ek.arange(UInt32, n_vertices * 3)
    )
    ek.scatter(
        m.faces_buffer(),
        UInt32(faces.ravel().astype(np.uint32)),
        ek.arange(UInt32, n_faces * 3)
    )
    ek.scatter(
        m.vertex_texcoords_buffer(),
        Float(texcoords.ravel().astype(np.float32)),
        ek.arange(UInt32, n_vertices * 2)
    )
    ek.eval()
    m.parameters_changed()

    return m


def main(config, output=None):
    from mitsuba.core import Bitmap, Vector3f
    from mitsuba.core.xml import load_dict
    from mitsuba.render import SurfaceInteraction3f
    from mitsuba.python.util import traverse
    from mitsuba.python.autodiff import render, write_bitmap, Adam

    output_dir = realpath(output or '.')
    os.makedirs(output_dir, exist_ok=True)

    config = CONFIGS[config]
    scene, integrator, sensor = load_scene(config, output_dir)

    crop_size = sensor.film().crop_size()

    # Heightmap (displacement texture) that will actually be optimized
    # Note: the resolution of displacement is also limited by the number
    #       of vertices that can be displaced ('lens' mesh).
    heightmap_texture = load_dict({
        'type': 'bitmap',
        'id': 'heightmap_texture',
        'bitmap': Bitmap(np.zeros(config['heightmap_resolution'], dtype=np.float32)),
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
    opt = Adam(lr=config['learning_rate'], params=params)
    opt.load()

    # Load or create the reference image
    image_ref = load_ref_image(config, crop_size, output_dir='.')
    ref_scale = ek.hsum(image_ref) / ek.width(image_ref)

    start_time = time.time()
    iterations = config['max_iterations']
    for it in range(iterations):
        t0 = time.time()

        apply_displacement()

        # Perform a differentiable rendering of the scene
        # TODO: support unbiased=True
        image = render(scene, optimizer=opt, unbiased=False, spp=config['spp'])
        # image = render(scene, optimizer=opt, unbiased=True, spp=config['spp'])

        if it % 5 == 0:
            write_bitmap('out_{:03d}.exr'.format(it), image, crop_size)

        # Loss function: scale-independent L2
        current_scale = ek.hsum(ek.detach(image)) / ek.width(image)
        loss = ek.hsum_async(ek.sqr(
            (image / current_scale) - (image_ref / ref_scale)
        )) / len(image)

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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, choices=list(CONFIGS.keys()), default='wave',
                        help='Configuration name')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output directory (defaults to current directory)')
    parsed = parser.parse_args()

    mitsuba.set_variant('cuda_ad_rgb')
    # TODO: enable more options once stable
    ek.set_flags(ek.JitFlag.ADOptimize)
    main(**vars(parsed))
