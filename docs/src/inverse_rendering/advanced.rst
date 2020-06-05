.. _sec-differentiable-rendering-advanced:

More advanced examples
======================

Differentiating textures
------------------------

The previous example demonstrated differentiation of a scalar color parameter.
We will now show how to work with textured parameters, focusing on the example
of an environment map emitter.

.. subfigstart::
.. subfigure:: ../../../resources/data/docs/images/autodiff/bunny.jpg
   :caption: A metallic Stanford bunny surrounded by an environment map.
.. subfigure :: ../../../resources/data/docs/images/autodiff/museum.jpg
   :caption: The ground-truth environment map that we will attempt to recover from the image on the left.
.. subfigend::
   :label: fig-bunny-autodiff

The example scene can be downloaded `here
<http://mitsuba-renderer.org/scenes/bunny.zip>`_ and contains a metallic
Stanford bunny surrounded by a museum-like environment. As before, we load
the scene and enumerate its differentiable parameters:

.. code-block:: python

    import enoki as ek
    import mitsuba
    mitsuba.set_variant('gpu_autodiff_rgb')

    from mitsuba.core import Float, Thread
    from mitsuba.core.xml import load_file
    from mitsuba.python.util import traverse
    from mitsuba.python.autodiff import render, write_bitmap, Adam

    # Load example scene
    Thread.thread().file_resolver().append('bunny')
    scene = load_file('bunny/bunny.xml')

    # Find differentiable scene parameters
    params = traverse(scene)

We then make a backup copy of the ground-truth environment map and generate a
reference rendering

.. code-block:: python

    # Make a backup copy
    param_res = params['my_envmap.resolution']
    param_ref = Float(params['my_envmap.data'])

    # Discard all parameters except for one we want to differentiate
    params.keep(['my_envmap.data'])

    # Render a reference image (no derivatives used yet)
    image_ref = render(scene, spp=16)
    crop_size = scene.sensors()[0].film().crop_size()
    write_bitmap('out_ref.png', image_ref, crop_size)

Let's now change the environment map into a uniform white lighting environment.
The ``my_envmap.data`` parameter is a RGBA bitmap linearized into a 1D array of
size ``param_res[0] x param_res[1] x 4`` (125'000).

.. code-block:: python

    # Change to a uniform white lighting environment
    params['my_envmap.data'] = ek.full(Float, 1.0, len(param_ref))
    params.update()

Finally, we jointly estimate all 125K parameters using gradient-based
optimization. The optimization loop is identical to previous examples except
that we can now also write out the current environment image in each iteration.

.. code-block:: python

    # Construct an Adam optimizer that will adjust the parameters 'params'
    opt = Adam(params, lr=.02)

    for it in range(100):
        # Perform a differentiable rendering of the scene
        image = render(scene, optimizer=opt, unbiased=True, spp=1)
        write_bitmap('out_%03i.png' % it, image, crop_size)
        write_bitmap('envmap_%03i.png' % it, params['my_envmap.data'],
                     (param_res[1], param_res[0]))

        # Objective: MSE between 'image' and 'image_ref'
        ob_val = ek.hsum(ek.sqr(image - image_ref)) / len(image)

        # Back-propagate errors to input parameters
        ek.backward(ob_val)

        # Optimizer: take a gradient step
        opt.step()

        # Compare iterate against ground-truth value
        err_ref = ek.hsum(ek.sqr(param_ref - params['my_envmap.data']))
        print('Iteration %03i: error=%g' % (it, err_ref[0]))

The following video shows the convergence behavior during the first 100
iterations. The image rapidly resolves to the target image. The small black
regions in the image correspond to parts of the mesh where inter-reflection was
ignored due to a limit on the maximum number of light bounces.

.. raw:: html

    <center>
        <video controls loop autoplay muted
        src="https:////rgl.s3.eu-central-1.amazonaws.com/media/uploads/wjakob/2020/03/03/bunny_render.mp4"></video>
    </center>

The following image shows the reconstructed environment map at each step.
Unobserved regions are unaffected by gradient steps and remain white.

.. raw:: html

    <center>
        <video controls loop autoplay muted
        src="https://rgl.s3.eu-central-1.amazonaws.com/media/uploads/wjakob/2020/03/03/bunny_envmap.mp4"></video>
    </center>

This image is still fairly noisy and even contains some negative (!) regions.
This is because the optimization problem defined above is highly ambiguous due
to the loss of information that occurs in the forward rendering model above.
The solution we found optimizes the objective well (i.e. the rendered image
matches the target), but the reconstructed texture may not match our
expectation. In such a case, it may be advisable to introduce further
regularization (non-negativity, smoothness, etc.).

.. note::

    The full Python script of this tutorial can be found in the file:
    :file:`docs/examples/10_inverse_rendering/invert_bunny.py`.


Pose optimization
------------------------

This advanced example demonstrates how to optimize the pose of an object so that it matches a
target reference image. One key aspect of this example is that it requires the differentiation of
mesh parameters, such as vertex positions. Computing derivatives for parameters that affect
visibility is a complex problem as it would normally make the integrants of the rendering equation
non-differentiable. For this reason, this example requires the use of the specialized
:ref:`pathreparam <integrator-pathreparam>` integrator, described in this
`article <https://rgl.epfl.ch/publications/Loubet2019Reparameterizing>`_.

The example scene can be found in ``resource/data/docs/examples/invert_pose/`` and contains a
simple object illuminated by a rectangular light source. To avoid discontinuities around the
area light, we use the :ref:`smootharea <emitter-smootharea>` plugin.

.. subfigstart::
.. subfigure:: ../../../resources/data/docs/images/autodiff/invert_pose_ref_image.png
   :caption: Reference image with the object at the target position.
.. subfigure :: ../../../resources/data/docs/images/autodiff/invert_pose_init_image.png
   :caption: Initial image with the object at the wrong position.
.. subfigend::
   :label: fig-pose-autodiff

First, we define two helper functions that we will use to transform the mesh
parameter buffers (flatten arrays) into ``VectorXf`` type (and the other way around).
Note that those functions will be natively supported by ``enoki`` in a futur release.

.. code-block:: python

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

Using those functions, we can get the initial vertex positions of the mesh after loading the scene:

.. code-block:: python

    # Load example scene
    scene_folder = '../../../resources/data/docs/examples/invert_pose/'
    Thread.thread().file_resolver().append(scene_folder)
    scene = xml.load_file(scene_folder + 'scene.xml')

    params = traverse(scene)
    positions_buf = params['object.vertex_positions_buf']
    positions_initial = ravel(positions_buf)

In this example, we are going to optimize a translation transform in order to align an object given
a reference image. To do so, we need to create a differential parameter that represents the
amount of translation along the 3 axis. After inserting this parameter in a Python dictionary
(similar structure as the :py:obj:`mitsuba.python.util.ParameterMap`), we construct an Adam
optimizer that will adjust those translation parameters during the optimization loop.

.. code-block:: python

    # Create differential parameter to be optimized
    translate_ref = Vector3f(0.0)

    # Create a new ParameterMap (or dict)
    params_optim = {
        "translate" : translate_ref,
    }

    # Construct an Adam optimizer that will adjust the translation parameters
    opt = Adam(params_optim, lr=0.02)

We define a helper function that creates a translation transform from this parameter and
applies it to every vertex position of the mesh. This function will be called at every iteration
of the optimization loop to update the mesh (and the scene's data structure) after the parameter was
adjusted by the optimizer.

.. code-block:: python

    # Apply the transformation to mesh vertex position and update scene (e.g. Optix BVH)
    def apply_transformation():
        trasfo = Transform4f.translate(params_optim["translate"])
        new_positions = trasfo.transform_point(positions_initial)
        unravel(new_positions, params['object.vertex_positions_buf'])
        params.set_dirty('object.vertex_positions_buf')
        params.update()

For the sake of this example, we synthetize the reference image with the null tranform:

.. code-block:: python

    # Render a reference image (no derivatives used yet)
    apply_transformation()
    image_ref = render(scene, spp=32)
    crop_size = scene.sensors()[0].film().crop_size()
    write_bitmap(output_path + 'out_ref.exr', image_ref, crop_size)
    print("Write " + output_path + "out_ref.exr")

Starting with a translated object, we can now perform a similar optimization loop as in the
previous examples. The ``apply_transformation()`` function is passed to the ``render`` function
as a pre-render callback, which will be called before every render.

.. code-block:: python

    # Move object before starting the optimization process
    params_optim["translate"] = Vector3f(0.5, 0.2, -0.2)

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


The following video show the result of this optimization, with the object slowy aligning with the
reference image:


Heightfield optimization
------------------------

In this example, we will optimize for the heightfield values on a grid mesh. This will be done by
adjusting the pixel values of displacement map applied on that mesh. We also need the
:ref:`pathreparam <integrator-pathreparam>` here as we will differentiate the vertex positions
of the grid.

The example scene can be found in ``resource/data/docs/examples/invert_heightfield/`` and contains a
simple grid mesh illuminated by a rectangular light source. To avoid discontinuities around the
area light, we use the :ref:`smootharea <emitter-smootharea>` plugin.

.. subfigstart::
.. subfigure:: ../../../resources/data/docs/images/autodiff/invert_heightfield_ref_image.png
   :caption: Reference render of the heightfield.
.. subfigure :: ../../../resources/data/docs/examples/invert_heightfield/mitsuba_coin.jpg
   :caption: Target displacement map.
.. subfigend::
   :label: fig-heightfield-autodiff

As in the previous example, we can load the scene and read the initial grid mesh parameters
(vertex positions, normals and texture coordinates), which we will use later in the script.

.. code-block:: python

    import enoki as ek
    import mitsuba
    mitsuba.set_variant('gpu_autodiff_rgb')

    from mitsuba.core import UInt32, Float, Thread, xml, Vector2f, Vector3f, Transform4f
    from mitsuba.render import SurfaceInteraction3f
    from mitsuba.python.util import traverse
    from mitsuba.python.autodiff import render, write_bitmap, Adam

    # Load example scene
    scene_folder = '../../../resources/data/docs/examples/invert_heightfield/'
    Thread.thread().file_resolver().append(scene_folder)
    scene = xml.load_file(scene_folder + 'scene.xml')

    params = traverse(scene)
    positions_buf = params['grid_mesh.vertex_positions_buf']
    positions_initial = ravel(positions_buf)
    normals_initial = ravel(params['grid_mesh.vertex_normals_buf'])
    vertex_count = ek.slices(positions_initial)

In this example, we implement displacement mapping directly in Python instead of using a C++ plugin.
This showcases the flexibility of the framework, and the ability to fully control the optimization
process. For instance, one could want to add constraints on the displacement values range, ...

We first create a :ref:`Bitmap <texture-bitmap>` texture instance using
:py:func:`mitsuba.core.xml.load_dict`, which will load the displacement map image file from disk.
We also create a :py:class:`mitsuba.render.SurfaceInteraction3f` with one entry per vertex on the
mesh. By properly setting the texture coordinates on this surface interaction, we can now evaluate
the displacement map for the entire mesh in one line of code.

.. code-block:: python

    # Create a texture with the reference displacement map
    disp_tex = xml.load_dict({
        "type" : "bitmap",
        "filename" : "mitsuba_coin.jpg"
    }).expand()[0]

    # Create a fake surface interaction with an entry per vertex on the mesh
    mesh_si = SurfaceInteraction3f.zero(vertex_count)
    mesh_si.uv = ravel(params['grid_mesh.vertex_texcoords_buf'], dim=2)

    # Evaluate the displacement map for the entire mesh
    disp_tex_data_ref = disp_tex.eval_1(mesh_si)

Finally, we define a function that applies the displacement map onto the original mesh. This will
be called at every iteration of the optimization loop to update the mesh data everytime the
displacement map is refined.

.. code-block:: python

    # Apply displacement to mesh vertex positions and call update scene
    def apply_displacement(amplitude = 0.05):
        new_positions = disp_tex.eval_1(mesh_si) * normals_initial * amplitude + positions_initial
        unravel(new_positions, params['grid_mesh.vertex_positions_buf'])
        params.set_dirty('grid_mesh.vertex_positions_buf')
        params.update()

We can now generate a reference image.

.. code-block:: python

    # Apply displacement before generating reference image
    apply_displacement()

    # Render a reference image (no derivatives used yet)
    image_ref = render(scene, spp=32)
    crop_size = scene.sensors()[0].film().crop_size()
    write_bitmap('out_ref.exr', image_ref, crop_size)
    print("Write out_ref.exr")

Before runing the optimization loop, we need to change the displacement data to a constant value
(here ``0.25``). This can be done using the :py:func:`mitsuba.python.util.traverse` function
on the texture object directly. We can then create an optimizer that will adjust those texture
parameters during the optimization process.

.. code-block:: python

    # Reset texture data to a constant
    disp_tex_params = traverse(disp_tex)
    disp_tex_params['data'] = ek.full(Float, 0.25, len(disp_tex_params['data']))
    disp_tex_params.update()

    # Construct an Adam optimizer that will adjust the texture parameters
    disp_tex_params.keep(['data'])
    opt = Adam(disp_tex_params, lr=0.002)

The optimization loop is very similar to the previous example, to the exception that it needs to
manually apply the displacement mapping to the mesh at every iteration.

.. code-block:: python

    iterations = 100
    for it in range(iterations):
        # Apply displacement to mesh and update scene (e.g. OptiX BVH)
        apply_displacement()

        # Perform a differentiable rendering of the scene
        image = render(scene, optimizer=opt, spp=4)
        write_bitmap('out_%03i.exr' % it, image, crop_size)

        # Objective: MSE between 'image' and 'image_ref'
        ob_val = ek.hsum(ek.sqr(image - image_ref)) / len(image)

        # Back-propagate errors to input parameters
        ek.backward(ob_val)

        # Optimizer: take a gradient step -> update displacement map
        opt.step()

        # Compare iterate against ground-truth value
        err_ref = ek.hsum(ek.sqr(disp_tex_data_ref - disp_tex.eval_1(mesh_si)))
        print('Iteration %03i: error=%g' % (it, err_ref[0]), end='\r')

Here we can see the result of the heightfield optimization:

.. note::

    The full Python script of this tutorial can be found in the file:
    :file:`docs/examples/10_inverse_rendering/invert_heightfield.py`.
