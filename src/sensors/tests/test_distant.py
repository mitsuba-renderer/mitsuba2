import numpy as np
import pytest

import enoki as ek


def sensor_dict(ray_target=None, ray_origin=None, film="1x1",
                direction=None, orientation=None, flip_directions=None):
    result = {"type": "distant"}

    if film == "1x1":
        result.update({
            "film": {
                "type": "hdrfilm",
                "width": 1,
                "height": 1,
                "rfilter": {"type": "box"}
            }
        })

    elif film == "32x1":
        result.update({
            "film": {
                "type": "hdrfilm",
                "width": 32,
                "height": 1,
                "rfilter": {"type": "box"}
            }
        })

    elif film == "32x32":
        result.update({
            "film": {
                "type": "hdrfilm",
                "width": 32,
                "height": 32,
                "rfilter": {"type": "box"}
            }
        })

    if direction is not None:
        result["direction"] = direction

    if orientation is not None:
        result["orientation"] = orientation

    if ray_target == "point":
        result.update({"ray_target": [0, 0, 0]})

    elif ray_target == "shape":
        result.update({"ray_target": {"type": "rectangle"}})

    elif isinstance(ray_target, dict):
        result.update({"ray_target": ray_target})

    if ray_origin == "shape":
        result.update({"ray_origin": {"type": "rectangle"}})

    elif isinstance(ray_origin, dict):
        result.update({"ray_origin": ray_origin})

    if flip_directions is not None:
        result.update({"flip_directions": flip_directions})

    return result


def make_sensor(d):
    from mitsuba.core.xml import load_dict

    return load_dict(d).expand()[0]


def test_construct(variant_scalar_rgb):
    # Construct without parameters
    sensor = make_sensor({"type": "distant"})
    assert sensor is not None
    assert not sensor.bbox().valid()  # Degenerate bounding box

    # Construct with direction, check transform setup correctness
    for direction, orientation, expected in [
        ([0, 0, 1], None, [[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]]),
        ([0, 0, 2], None, [[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]]),
        ([0, 0, 1], [1, 0, 0], [[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]]),
        ([0, 0, 1], [0, 1, 0], [[0, -1, 0, 0],
                                [1, 0, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]]),
        ([0, 0, 1], [0, -1, 0], [[0, 1, 0, 0],
                                 [-1, 0, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]]),
        ([0, 0, 1], [1, 1, 0], [[0.707107, -0.707107, 0, 0],
                                [0.707107, 0.707107, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]]),
    ]:
        sensor = make_sensor(sensor_dict(direction=direction,
                                         orientation=orientation))
        result = sensor.world_transform().eval(0.).matrix
        assert np.allclose(result, expected)
        # Couldn't get ek.allclose() to work here

    # Test different combinations of target and origin values
    # -- No target, no origin
    sensor = make_sensor(sensor_dict())
    assert sensor is not None

    # -- Point target
    sensor = make_sensor(sensor_dict(ray_target="point"))
    assert sensor is not None

    # -- Shape target
    sensor = make_sensor(sensor_dict(ray_target="shape"))
    assert sensor is not None

    # -- Random object target (we expect to raise)
    with pytest.raises(RuntimeError):
        make_sensor(sensor_dict(ray_target={"type": "constant"}))

    # -- Shape origin
    sensor = make_sensor(sensor_dict(ray_target="shape"))
    assert sensor is not None

    # -- Random object origin (we expect to raise)
    with pytest.raises(RuntimeError):
        make_sensor(sensor_dict(ray_origin={"type": "constant"}))


@pytest.mark.parametrize(
    "direction", [[0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [2.0, 0.0, 0.0]]
)
def test_sample_ray_direction_1x1(variant_scalar_rgb, direction):
    sensor = make_sensor(sensor_dict(film="1x1", direction=direction))

    # Check that directions are appropriately set
    for (sample1, sample2) in [
        [[0.32, 0.87], [0.16, 0.44]],
        [[0.17, 0.44], [0.22, 0.81]],
        [[0.12, 0.82], [0.99, 0.42]],
        [[0.72, 0.40], [0.01, 0.61]],
    ]:
        ray, _ = sensor.sample_ray(1.0, 1.0, sample1, sample2, True)

        # Check that ray direction is what is expected
        assert ek.allclose(ray.d, -ek.normalize(direction))


@pytest.mark.parametrize("orientation", [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, 1.0, 0.0],
])
def test_sample_ray_direction_32x1(variant_scalar_rgb, orientation):
    direction = [0.0, 0.0, 1.0]
    sensor = make_sensor(
        sensor_dict(film="32x1", direction=direction, orientation=orientation)
    )

    excluded = ek.cross(direction, orientation)

    # Check that directions are appropriately set
    for (sample1, sample2) in [
        [[0.32, 0.87], [0.16, 0.44]],
        [[0.17, 0.44], [0.22, 0.81]],
        [[0.12, 0.82], [0.99, 0.42]],
        [[0.72, 0.40], [0.01, 0.61]],
    ]:
        ray, _ = sensor.sample_ray(1.0, 1.0, sample1, sample2, True)

        # Check that ray direction is what is expected
        assert ek.allclose(ek.dot(ray.d, excluded), 0.0)


def test_flip_direction(variant_scalar_rgb):
    time = 1.0
    sample_wav = 1.0
    samples = [
        [[0.32, 0.87], [0.16, 0.44]],
        [[0.17, 0.44], [0.22, 0.81]],
        [[0.12, 0.82], [0.99, 0.42]],
        [[0.72, 0.40], [0.01, 0.61]],
    ]

    direction = [0.0, 0.0, 1.0]
    sensor1 = make_sensor(
        sensor_dict(film="32x32", direction=direction, flip_directions=False)
    )
    sensor2 = make_sensor(
        sensor_dict(film="32x32", direction=direction, flip_directions=True)
    )

    for (sample1, sample2) in samples:
        ray1, _ = sensor1.sample_ray(time, sample_wav, sample1, sample2, True)
        ray2, _ = sensor2.sample_ray(time, sample_wav, sample1, sample2, True)

        assert ek.allclose(ray1.d, -ray2.d)


def bsphere(bbox):
    c = bbox.center()
    return c, ek.norm(c - bbox.max)


@pytest.mark.parametrize("ray_kind", ["regular", "differential"])
def test_sample_ray_origin(variant_scalar_rgb, ray_kind):
    time = 1.0
    sample_wav = 1.0
    samples = [
        [[0.32, 0.87], [0.16, 0.44]],
        [[0.17, 0.44], [0.22, 0.81]],
        [[0.12, 0.82], [0.99, 0.42]],
        [[0.72, 0.40], [0.01, 0.61]],
    ]

    from mitsuba.core.xml import load_dict
    from mitsuba.core import ScalarTransform4f

    # Default origin: use bounding sphere to compute ray origins
    scene_dict = {
        "type": "scene",
        "shape": {"type": "rectangle"},
        "sensor": sensor_dict(direction=[0, 0, 1]),
    }
    scene = load_dict(scene_dict)
    sensor = scene.sensors()[0]

    _, bsphere_radius = bsphere(scene.bbox())

    for (sample1, sample2) in samples:
        if ray_kind == "regular":
            ray, _ = sensor.sample_ray(time, sample_wav, sample1, sample2, True)
        elif ray_kind == "differential":
            ray, _ = sensor.sample_ray_differential(
                time, sample_wav, sample1, sample2, True
            )
            assert not ray.has_differentials

        assert ek.allclose(ray.o.z, bsphere_radius, rtol=1e-3)

    # Shape origin: we set ray origins to be located on a rectangle located at
    # an arbitrary z altitude
    z_offset = 3.42

    for direction in [[0, 0, 1], [0, 2, 1]]:
        scene_dict = {
            "type": "scene",
            "shape": {"type": "rectangle"},
            "sensor": sensor_dict(
                direction=direction,
                ray_origin={
                    "type": "rectangle",
                    "to_world":
                        ScalarTransform4f.translate([0, 0, z_offset]) *
                        ScalarTransform4f.scale(10)
                        # ^-- scaling ensures that the square covers the entire
                        #     area where target points can be located
                }),
        }
        scene = load_dict(scene_dict)
        sensor = scene.sensors()[0]

        for (sample1, sample2) in samples:
            ray, _ = sensor.sample_ray(time, sample_wav, sample1, sample2, True)
            assert ek.allclose(ray.o.z, z_offset)

    # Check that wrong origins will lead to invalid rays
    direction = [0, 0, 1]
    scene_dict = {
        "type": "scene",
        "shape": {"type": "rectangle"},
        "sensor": sensor_dict(
            direction=direction,
            ray_origin={
                "type": "rectangle",
                "to_world":
                    ScalarTransform4f.translate([0, 0, -1.])
                    # ^-- origin surface cannot be reached given ray direction:
                    #     this will always produce invalid origins
            }),
    }
    scene = load_dict(scene_dict)
    sensor = scene.sensors()[0]

    for (sample1, sample2) in samples:
        ray, weights = sensor.sample_ray(time, sample_wav, sample1, sample2, True)
        assert any(ek.isnan(ray.o))
        assert ek.allclose(weights, 0.0)


@pytest.mark.parametrize(
    "sensor_setup",
    [
        "default",
        "target_square",
        "target_square_small",
        "target_square_large",
        "target_disk",
        "target_point",
    ],
)
@pytest.mark.parametrize("w_e", [[0, 0, -1], [0, 1, -1]])
@pytest.mark.parametrize("w_o", [[0, 0, 1], [0, 1, 1]])
def test_sample_ray_target(variant_scalar_rgb, sensor_setup, w_e, w_o):
    # This test checks if targeting works as intended by rendering a basic scene
    from mitsuba.core import ScalarTransform4f, Struct, Bitmap
    from mitsuba.core.xml import load_dict

    # Basic illumination and sensing parameters
    l_e = 1.0  # Emitted radiance
    w_e = list(w_e / np.linalg.norm(w_e))  # Emitter direction
    w_o = list(w_o / np.linalg.norm(w_o))  # Sensor direction
    cos_theta_e = abs(np.dot(w_e, [0, 0, 1]))
    cos_theta_o = abs(np.dot(w_o, [0, 0, 1]))

    # Reflecting surface specification
    surface_scale = 1.0
    rho = 1.0  # Surface reflectance

    # Sensor definitions
    sensors = {
        "default": {  # No target, origin projected to bounding sphere
            "type": "distant",
            "direction": w_o,
            "sampler": {
                "type": "independent",
                "sample_count": 100000,
            },
            "film": {
                "type": "hdrfilm",
                "height": 1,
                "width": 1,
                "rfilter": {"type": "box"},
            },
        },
        "target_square": {  # Targeting square, origin projected to bounding sphere
            "type": "distant",
            "direction": w_o,
            "ray_target": {
                "type": "rectangle",
                "to_world": ScalarTransform4f.scale(surface_scale),
            },
            "sampler": {
                "type": "independent",
                "sample_count": 100000,
            },
            "film": {
                "type": "hdrfilm",
                "height": 1,
                "width": 1,
                "rfilter": {"type": "box"},
            },
        },
        "target_square_small": {  # Targeting small square, origin projected to bounding sphere
            "type": "distant",
            "direction": w_o,
            "ray_target": {
                "type": "rectangle",
                "to_world": ScalarTransform4f.scale(0.5 * surface_scale),
            },
            "sampler": {
                "type": "independent",
                "sample_count": 100000,
            },
            "film": {
                "type": "hdrfilm",
                "height": 1,
                "width": 1,
                "rfilter": {"type": "box"},
            },
        },
        "target_square_large": {  # Targeting large square, origin projected to bounding sphere
            "type": "distant",
            "direction": w_o,
            "ray_target": {
                "type": "rectangle",
                "to_world": ScalarTransform4f.scale(2.0 * surface_scale),
            },
            "sampler": {
                "type": "independent",
                "sample_count": 100000,
            },
            "film": {
                "type": "hdrfilm",
                "height": 1,
                "width": 1,
                "rfilter": {"type": "box"},
            },
        },
        "target_point": {  # Targeting point, origin projected to bounding sphere
            "type": "distant",
            "direction": w_o,
            "ray_target": [0, 0, 0],
            "sampler": {
                "type": "independent",
                "sample_count": 100000,
            },
            "film": {
                "type": "hdrfilm",
                "height": 1,
                "width": 1,
                "rfilter": {"type": "box"},
            },
        },
        "target_disk": {  # Targeting disk, origin projected to bounding sphere
            "type": "distant",
            "direction": w_o,
            "ray_target": {
                "type": "disk",
                "to_world": ScalarTransform4f.scale(surface_scale),
            },
            "sampler": {
                "type": "independent",
                "sample_count": 100000,
            },
            "film": {
                "type": "hdrfilm",
                "height": 1,
                "width": 1,
                "rfilter": {"type": "box"},
            },
        },
    }

    # Scene setup
    scene_dict = {
        "type": "scene",
        "shape": {
            "type": "rectangle",
            "to_world": ScalarTransform4f.scale(surface_scale),
            "bsdf": {
                "type": "diffuse",
                "reflectance": rho,
            },
        },
        "emitter": {"type": "directional", "direction": w_e, "irradiance": l_e},
        "integrator": {"type": "path"},
    }

    scene = load_dict({**scene_dict, "sensor": sensors[sensor_setup]})

    # Run simulation
    sensor = scene.sensors()[0]
    scene.integrator().render(scene, sensor)

    # Check result
    result = np.array(
        sensor.film()
        .bitmap()
        .convert(Bitmap.PixelFormat.RGB, Struct.Type.Float32, False)
    ).squeeze()

    surface_area = 4.0 * surface_scale ** 2  # Area of square surface
    l_o = l_e * cos_theta_e * rho / np.pi  # * cos_theta_o
    expected = {  # Special expected values for some cases
        "default": l_o * 2.0 / ek.pi,
        "target_square_large": l_o * 0.25,
    }
    expected_value = expected.get(sensor_setup, l_o)

    rtol = {  # Special tolerance values for some cases
        "target_square_large": 1e-2,
    }
    rtol_value = rtol.get(sensor_setup, 5e-3)

    assert np.allclose(result, expected_value, rtol=rtol_value)


def test_checkerboard(variant_scalar_rgb):
    """
    Very basic render test with checkboard texture and square target.
    """
    from mitsuba.core import ScalarTransform4f
    from mitsuba.core.xml import load_dict

    l_o = 1.0
    rho0 = 0.5
    rho1 = 1.0

    # Scene setup
    scene_dict = {
        "type": "scene",
        "shape": {
            "type": "rectangle",
            "bsdf": {
                "type": "diffuse",
                "reflectance": {
                    "type": "checkerboard",
                    "color0": rho0,
                    "color1": rho1,
                    "to_uv": ScalarTransform4f.scale(2),
                },
            },
        },
        "emitter": {"type": "directional", "direction": [0, 0, -1], "irradiance": 1.0},
        "sensor0": {
            "type": "distant",
            "direction": [0, 0, 1],
            "ray_target": {"type": "rectangle"},
            "sampler": {
                "type": "independent",
                "sample_count": 50000,
            },
            "film": {
                "type": "hdrfilm",
                "height": 2,
                "width": 2,
                "pixel_format": "luminance",
                "component_format": "float32",
                "rfilter": {"type": "box"},
            },
        },
        # "sensor1": {  # In case one would like to check what the scene looks like
        #     "type": "perspective",
        #     "to_world": ScalarTransform4f.look_at(origin=[0, 0, 5], target=[0, 0, 0], up=[0, 1, 0]),
        #     "sampler": {
        #         "type": "independent",
        #         "sample_count": 10000,
        #     },
        #     "film": {
        #         "type": "hdrfilm",
        #         "height": 256,
        #         "width": 256,
        #         "pixel_format": "luminance",
        #         "component_format": "float32",
        #         "rfilter": {"type": "box"},
        #     },
        # },
        "integrator": {"type": "path"},
    }

    scene = load_dict(scene_dict)

    sensor = scene.sensors()[0]
    scene.integrator().render(scene, sensor)

    data = np.array(sensor.film().bitmap())

    expected = l_o * 0.5 * (rho0 + rho1) / ek.pi
    assert np.allclose(data, expected, atol=1e-3)
