#include <mitsuba/render/sensor.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/core/bbox.h>
#include <mitsuba/core/fresolver.h>

// Alembic Includes
#include <Alembic/Abc/All.h>
#include <Alembic/AbcGeom/All.h>
#include <Alembic/AbcCoreFactory/All.h>
#include <Alembic/AbcCoreOgawa/All.h>

using namespace Alembic::AbcGeom;
using namespace Alembic::AbcCoreFactory;
using namespace Alembic::AbcCoreAbstract;

NAMESPACE_BEGIN(mitsuba)

/**!

.. _sensor-perspective_abc:

Alembic loader for perspective pinhole camera (:monosp:`perspective_abc`)
--------------------------------------------------

.. pluginparameters::

 * - filename
   - |string|
   - Filename of the Alembic file to load 
 * - shape_name
   - |string|
   - Alembic file may contain several separate cameras or other objects. This optional parameter
     specifies name of camera to load.
 * - to_world
   - |transform|
   - Specifies an optional camera-to-world transformation.
     (Default: none (i.e. camera space = world space))
 * - convert_to_z_up
   - |bool|
   - Optionally convert camera so up vector is +Z (Mitsuba default). Useful if camera was exported
   - from software that is Y-up. For example, Maya or Houdini. (Default: |false|)

This plugin brings support for reading cameras from Alembic file.
It reads position and orientation data as well as field of view and near/far clipping planes.
Other camera parameters are identical to standard perspective camera plugin.

.. code-block:: xml

    <sensor type="perspective_abc">
        <string name="filename" value="path/to/abc/city.abc"/>
		<string name="shape_name" value="hero_camera"/>
    </sensor>

 */

template <typename Float, typename Spectrum>
class AlembicPerspectiveCamera final : public ProjectiveCamera<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(ProjectiveCamera, m_world_transform, m_needs_sample_3,
                    m_film, m_sampler, m_resolution, m_shutter_open,
                    m_shutter_open_time, m_near_clip, m_far_clip)
    MTS_IMPORT_TYPES()

    // =============================================================
    //! @{ \name Constructors
    // =============================================================

    AlembicPerspectiveCamera(const Properties &props) : Base(props) {
        auto fs = Thread::thread()->file_resolver();
        fs::path file_path = fs->resolve(props.string("filename"));
        m_name = file_path.filename().string();

        auto fail = [&](const std::string &descr) {
            Throw("Error while loading camera from Alembic file \"%s\": %s.", m_name, descr);
        };

        Log(Debug, "Loading camera from \"%s\" ..", m_name);
        if (!fs::exists(file_path))
            fail("file not found");

        m_shape_name = props.string("shape_name", "");
        const bool convert_to_z_up = props.bool_("convert_to_z_up", false);
        m_use_shape_name = false;

        if (!m_shape_name.empty())
            m_use_shape_name = true;

        IFactory factory;
        IFactory::CoreType core_type;
        IArchive archive = factory.getArchive(file_path.string(), core_type);
        IObject archive_top = archive.getTop();

        index_t sample_index = 0;
        ScalarTransform4f alembic_transform;

        ICamera camera;
        find_camera_recursively(archive_top, sample_index, alembic_transform, camera);

        // fail if camera was not found in file
        if (!camera.valid()){
            std::string error_msg = "did not found camera in file";
            if (m_use_shape_name)
                error_msg += tfm::format(". Got shape_name \"%s\". "
                                         "Please check if this camera was exported into "
                                         "Alembic file and name is correct" , m_shape_name);
            fail(error_msg);
        }

        CameraSample camera_sample = camera.getSchema().getValue(
                                                ISampleSelector(m_alembic_time) );

        ScalarVector2i size = m_film->size();
        m_x_fov = camera_sample.getFieldOfView();

        //support crops
        parse_fov(props, size.x() / (float) size.y());

        // convert y up to z up if requested
        if (convert_to_z_up){
            ScalarTransform4f rotate_to_z_up;
            rotate_to_z_up.matrix = {-1, 0, 0, 0,
                                  0, 1, 0, 0,
                                  0, 0, -1, 0,
                                  0, 0, 0, 1
                                };
            m_obj_xform = m_obj_xform * rotate_to_z_up;
        }
        
        m_near_clip = camera_sample.getNearClippingPlane();
        m_far_clip = camera_sample.getFarClippingPlane();

        if (m_world_transform->has_scale())
            Throw("Scale factors in the camera-to-world transformation are not allowed!");

        update_camera_transforms();
    }

    void update_camera_transforms() {
        m_camera_to_sample = perspective_projection(
            m_film->size(), m_film->crop_size(), m_film->crop_offset(),
            m_x_fov, m_near_clip, m_far_clip);

        m_sample_to_camera = m_camera_to_sample.inverse();

        // Position differentials on the near plane
        m_dx = m_sample_to_camera * ScalarPoint3f(1.f / m_resolution.x(), 0.f, 0.f) -
               m_sample_to_camera * ScalarPoint3f(0.f);
        m_dy = m_sample_to_camera * ScalarPoint3f(0.f, 1.f / m_resolution.y(), 0.f)
             - m_sample_to_camera * ScalarPoint3f(0.f);

        /* Precompute some data for importance(). Please
           look at that function for further details. */
        ScalarPoint3f pmin(m_sample_to_camera * ScalarPoint3f(0.f, 0.f, 0.f)),
                      pmax(m_sample_to_camera * ScalarPoint3f(1.f, 1.f, 0.f));

        m_image_rect.reset();
        m_image_rect.expand(ScalarPoint2f(pmin.x(), pmin.y()) / pmin.z());
        m_image_rect.expand(ScalarPoint2f(pmax.x(), pmax.y()) / pmax.z());
        m_normalization = 1.f / m_image_rect.volume();
        m_needs_sample_3 = false;
    }

    //! @}
    // =============================================================

    // =============================================================
    //! @{ \name Sampling methods (Sensor interface)
    // =============================================================

    std::pair<Ray3f, Spectrum> sample_ray(Float time, Float wavelength_sample,
                                          const Point2f &position_sample,
                                          const Point2f & /*aperture_sample*/,
                                          Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);

        auto [wavelengths, wav_weight] = sample_wavelength<Float, Spectrum>(wavelength_sample);
        Ray3f ray;
        ray.time = time;
        ray.wavelengths = wavelengths;

        // Compute the sample position on the near plane (local camera space).
        Point3f near_p = m_sample_to_camera *
                         Point3f(position_sample.x(), position_sample.y(), 0.f);

        // Convert into a normalized ray direction; adjust the ray interval accordingly.
        Vector3f d = normalize(Vector3f(near_p));

        Float inv_z = rcp(d.z());
        ray.mint = m_near_clip * inv_z;
        ray.maxt = m_far_clip * inv_z;

        auto trafo = m_obj_xform * m_world_transform->eval(ray.time, active);
        // auto trafo = m_obj_xform;

        ray.o = trafo.translation();
        ray.d = trafo * d;
        ray.update();

        return std::make_pair(ray, wav_weight);
    }

    std::pair<RayDifferential3f, Spectrum>
    sample_ray_differential(Float time, Float wavelength_sample, const Point2f &position_sample,
                            const Point2f & /*aperture_sample*/, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);

        auto [wavelengths, wav_weight] = sample_wavelength<Float, Spectrum>(wavelength_sample);
        RayDifferential3f ray;
        ray.time = time;
        ray.wavelengths = wavelengths;

        // Compute the sample position on the near plane (local camera space).
        Point3f near_p = m_sample_to_camera *
                         Point3f(position_sample.x(), position_sample.y(), 0.f);

        // Convert into a normalized ray direction; adjust the ray interval accordingly.
        Vector3f d = normalize(Vector3f(near_p));
        Float inv_z = rcp(d.z());
        ray.mint = m_near_clip * inv_z;
        ray.maxt = m_far_clip * inv_z;

        auto trafo = m_obj_xform * m_world_transform->eval(ray.time, active);
        // auto trafo = m_obj_xform;
        ray.o = trafo.transform_affine(Point3f(0.f));
        ray.d = trafo * d;
        ray.update();

        ray.o_x = ray.o_y = ray.o;

        ray.d_x = trafo * normalize(Vector3f(near_p) + m_dx);
        ray.d_y = trafo * normalize(Vector3f(near_p) + m_dy);
        ray.has_differentials = true;

        return std::make_pair(ray, wav_weight);
    }

    ScalarBoundingBox3f bbox() const override {
        return m_world_transform->translation_bounds();
    }

    /**
     * \brief Compute the directional sensor response function of the camera
     * multiplied with the cosine foreshortening factor associated with the
     * image plane
     *
     * \param d
     *     A normalized direction vector from the aperture position to the
     *     reference point in question (all in local camera space)
     */
    Float importance(const Vector3f &d) const {
        /* How is this derived? Imagine a hypothetical image plane at a
           distance of d=1 away from the pinhole in camera space.

           Then the visible rectangular portion of the plane has the area

              A = (2 * tan(0.5 * xfov in radians))^2 / aspect

           Since we allow crop regions, the actual visible area is
           potentially reduced:

              A' = A * (cropX / filmX) * (cropY / filmY)

           Perspective transformations of such aligned rectangles produce
           an equivalent scaled (but otherwise undistorted) rectangle
           in screen space. This means that a strategy, which uniformly
           generates samples in screen space has an associated area
           density of 1/A' on this rectangle.

           To compute the solid angle density of a sampled point P on
           the rectangle, we can apply the usual measure conversion term:

              d_omega = 1/A' * distance(P, origin)^2 / cos(theta)

           where theta is the angle that the unit direction vector from
           the origin to P makes with the rectangle. Since

              distance(P, origin)^2 = Px^2 + Py^2 + 1

           and

              cos(theta) = 1/sqrt(Px^2 + Py^2 + 1),

           we have

              d_omega = 1 / (A' * cos^3(theta))
        */

        Float ct = Frame3f::cos_theta(d), inv_ct = rcp(ct);

        // Compute the position on the plane at distance 1
        Point2f p(d.x() * inv_ct, d.y() * inv_ct);

        /* Check if the point lies to the front and inside the
           chosen crop rectangle */
        Mask valid = ct > 0 && m_image_rect.contains(p);

        return select(valid, m_normalization * inv_ct * inv_ct * inv_ct, 0.f);
    }

    //! @}
    // =============================================================

    void traverse(TraversalCallback *callback) override {
        Base::traverse(callback);
        // TODO x_fov
    }

    void parameters_changed(const std::vector<std::string> &keys) override {
        Base::parameters_changed(keys);
        // TODO
    }

    std::string to_string() const override {
        using string::indent;

        std::ostringstream oss;
        oss << "AlembicPerspectiveCamera[" << std::endl
            << "  x_fov = " << m_x_fov << "," << std::endl
            << "  near_clip = " << m_near_clip << "," << std::endl
            << "  far_clip = " << m_far_clip << "," << std::endl
            << "  film = " << indent(m_film) << "," << std::endl
            << "  sampler = " << indent(m_sampler) << "," << std::endl
            << "  resolution = " << m_resolution << "," << std::endl
            << "  shutter_open = " << m_shutter_open << "," << std::endl
            << "  shutter_open_time = " << m_shutter_open_time << "," << std::endl
            << "  world_transform = " << indent(m_world_transform) << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    ScalarTransform4f m_camera_to_sample;
    ScalarTransform4f m_sample_to_camera;
    ScalarBoundingBox2f m_image_rect;
    ScalarFloat m_normalization;
    ScalarFloat m_x_fov;
    ScalarVector3f m_dx, m_dy;

    // Alembic camera variables
    std::string m_alembic_name;
    double m_alembic_time = 0;
    std::string m_name;
    Transform4f m_obj_xform;
    std::string m_shape_name;
    bool m_use_shape_name;

    void find_camera_recursively(const IObject& obj,
        index_t sample_index,
        ScalarTransform4f &t, ICamera& camera )
    {
        for (size_t i = 0; i < obj.getNumChildren(); i++)
        {
            ScalarTransform4f child_transform = t;
            IObject child = obj.getChild(i);

            if ( IXform::matches(child.getMetaData()) ){
                IXform xform_obj = IXform(child, kWrapExisting);
                IXformSchema &xs = xform_obj.getSchema();
                XformSample xform_sample = xs.getValue(ISampleSelector(m_alembic_time));

                ScalarTransform4f obj_xform = read_abc_xform(child);
                if (xform_sample.getInheritsXforms()){
                    // camera can be constrained to another object
                    child_transform = child_transform * obj_xform;
                }
                else{
                    child_transform = obj_xform;
                }
            }
            else if ( ICamera::matches(child.getMetaData()) ){
                bool add_camera = true;
                if (m_use_shape_name){
                    // check if shape name is in alembic's camera name to avoid
                    // typing full names.
                    // if camera name from Xml file is not in camera name, skip camera
                    if (child.getFullName().find(m_shape_name) == std::string::npos)
                        add_camera =false;
                }

                if (add_camera){
                    camera = ICamera(child, kWrapExisting);
                    m_obj_xform = child_transform;
                }
                continue;
            }
            // continue traversing alembic tree in case there are more cameras
            find_camera_recursively(child, sample_index, child_transform, camera);
        }
    }

    ScalarTransform4f read_abc_xform(const IObject& obj)
    {
        ScalarTransform4f obj_xform;
        if (IXform::matches(obj.getMetaData()) ){
            IXform xform_obj = IXform(obj, kWrapExisting);
            IXformSchema &xs = xform_obj.getSchema();
            XformSample xform_sample = xs.getValue(ISampleSelector(m_alembic_time));

            if (xform_sample.getInheritsXforms()){
                M44d m = xform_sample.getMatrix();
                for (size_t i = 0; i < 4; ++i){
                    for (size_t j = 0; j < 4; ++j){
                        obj_xform.matrix[i][j] = m[i][j];
                    }
                }
            }
        }
        return obj_xform;
    }
};

MTS_IMPLEMENT_CLASS_VARIANT(AlembicPerspectiveCamera, ProjectiveCamera)
MTS_EXPORT_PLUGIN(AlembicPerspectiveCamera, "Alembic Perspective Camera");
NAMESPACE_END(mitsuba)
