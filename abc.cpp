#include <mitsuba/render/mesh.h>
#include <mitsuba/core/fstream.h>
#include <mitsuba/core/mstream.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/util.h>
#include <mitsuba/core/timer.h>
#include <unordered_map>
#include <fstream>

// Alembic Includes
#include <Alembic/Abc/All.h>
#include <Alembic/AbcGeom/All.h>
#include <Alembic/AbcCoreFactory/All.h>
#include <Alembic/AbcCoreOgawa/All.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _shape-abc:

ABC (Alembic) mesh loader (:monosp:`abc`)
----------------------------------------------------------

.. pluginparameters::

 * - filename
   - |string|
   - Filename of the Alembic file to load 
 * - shape_name
   - |string|
   - Alembic file may contain several separate meshes. This optional parameter
     specifies name of mesh to load. (Default: not specified, i.e. load all objects)
 * - shape_index
   - |int|
   - Alembic file may contain several separate meshes. This optional parameter
     specifies which mesh should be loaded. (Default: 0, i.e. the first one) If both 
     shape_name and shape_index are specified, shape_name takes precedence.
 * - m_flip_normals
   - |bool|
   - Will flip normals on the mesh. Some programs output face indexes in clockwise order, some in counter-clockwise. 
     This parameter optionally fixes handedness. (Default: |true|)
 * - load_visible
   - |bool|
   - Alembic allows for objects or specific faces of an object to be flagged as either visible or hidden. 
     This is useful because it allows objects or faces to appear or disaper over time, by animating the visiblity property. 
     Setting this to |true| will load only objects flagged as visible. (Default: |false|)
 * - face_normals
   - |bool|
   - When set to |true|, any existing or computed vertex normals are
     discarded and *face normals* will instead be used during rendering.
     This gives the rendered object a faceted appearance. (Default: |false|)
 * - flip_tex_coords
   - |bool|
   - Treat the vertical component of the texture as inverted? (Default: |true|)
 * - to_world
   - |Transform4f|
   - Specifies an optional linear object-to-world Transform4fation.
     (Default: none, i.e. object space = world space)

This plugin implements loader for Alembic file format (www.alembic.io). The
current plugin implementation supports arbitrary meshes with optional UV
coordinates, vertex normals and other custom vertex or face attributes.

 */


using namespace std;
using namespace Alembic::AbcGeom;
using namespace Alembic::AbcCoreFactory;
using namespace Alembic::AbcCoreAbstract;

template <typename Float, typename Spectrum>
class AlembicMesh final : public Mesh<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Mesh, m_bsdf, m_name, m_bbox, m_to_world, m_vertex_count, m_face_count,
                    m_vertex_positions_buf, m_vertex_normals_buf, m_vertex_texcoords_buf,
                    m_faces_buf, m_disable_vertex_normals, has_vertex_normals,
                    has_vertex_texcoords, recompute_vertex_normals, set_children)
    MTS_IMPORT_TYPES(BSDF, Shape)

    // Mesh is always stored in single precision
    using InputFloat = float;
    using InputPoint3f  = Point<InputFloat, 3>;
    using InputVector2f = Vector<InputFloat, 2>;
    using InputVector3f = Vector<InputFloat, 3>;
    using InputNormal3f = Normal<InputFloat, 3>;
    using FloatStorage = DynamicBuffer<replace_scalar_t<Float, InputFloat>>;

    using typename Base::ScalarSize;
    using typename Base::ScalarIndex;
    using typename Base::MeshAttributeType;
    using ScalarIndex3 = std::array<ScalarIndex, 3>;

    struct AlembicPolymesh
    {
        std::string id;
        IPolyMeshSchema schema;
        ScalarTransform4f xform;
    };

    struct VertexBinding {
        ScalarIndex3 key {{ 0, 0, 0 }};
        ScalarIndex value { 0 };
        VertexBinding *next { nullptr };
    };

    bool m_flip_normals = true;
    bool m_load_visible = false;

    // Alembics can store animation data
    // For now use values at 0 time until Mitsuba supports animaiton
    double m_alembic_time = 0;

    AlembicMesh(const Properties &props) : Base(props) {

        auto fs = Thread::thread()->file_resolver();
        fs::path file_path = fs->resolve(props.string("filename"));
        m_name = file_path.filename().string();

        // Some DCC output face indices in clockwise order, some in counter-clockwise. 
        // It looks like Mitsuba expects opposite order compared to most common DCC's
        // Optionally fix handedness to avoid black renders.
        m_flip_normals = props.bool_("flip_normals", true);
        m_load_visible = props.bool_("load_visible", false);
        const long shape_index = props.int_("shape_index", -1);
        const bool flip_tex_coords = props.bool_("flip_tex_coords", true);
        m_shape_name = props.string("shape_name", "");
        m_use_shape_name = false;
        bool use_shape_index = false;

        if (!m_shape_name.empty())
            m_use_shape_name = true;

        // Use shape_index if it is provided and shape_name is not used
        if (shape_index != -1 && !m_use_shape_name)
            use_shape_index = true;

        auto fail = [&](const std::string &descr) {
            Throw("Error while loading Alembic file \"%s\": %s.", m_name, descr);
        };
        auto detailed_fail = [&](const std::string &descr, std::string m_mesh_name="") {
            Throw("Error while loading Alembic file \"%s\" at object \"%s\". %s.", m_name, m_mesh_name, descr);
        };

        Log(Debug, "Loading mesh from \"%s\" ..", m_name);
        if (!fs::exists(file_path))
            fail("file not found");

        IFactory factory;
        IFactory::CoreType core_type;
        IArchive archive = factory.getArchive(file_path.string(), core_type);
        IObject archive_top = archive.getTop();

        index_t sample_index = 0;
        std::vector<AlembicPolymesh> alembic_polymeshes;
        ScalarTransform4f alembic_transform;

        // create one mesh specified by shape_index
        if (use_shape_index && (shape_index > (uint) archive_top.getNumChildren() - 1) )
            fail(tfm::format("Unable to load mesh. Shape index \"%i\" is "
                                "out of range 0..%i",
                                shape_index, archive_top.getNumChildren() - 1));

        if (use_shape_index)
            archive_top = archive_top.getChild(shape_index);
         
        ScalarTransform4f obj_xform = read_abc_xform(archive_top);
        find_children_recursively(archive_top, sample_index, alembic_transform, alembic_polymeshes);

        if (alembic_polymeshes.empty()){
            std::string error_msg = "found no polygonal meshes in file";
            if (m_use_shape_name)
                error_msg += tfm::format(". Got shape_name \"%s\" "
                                         "please check if this shape was exported into Alembic file" , m_shape_name);
            if (use_shape_index)
                error_msg += tfm::format(". Got shape_index \"%i\" "
                                         "could not get polymesh. Object at \"%i\" is \"%s\"" , 
                                         shape_index, shape_index, archive_top.getFullName());
            fail(error_msg);
        }

        for (size_t each_alembic = 0; each_alembic < alembic_polymeshes.size(); each_alembic++){
            m_mesh_name = alembic_polymeshes[each_alembic].id;

            IPolyMeshSchema poly_schema = alembic_polymeshes[each_alembic].schema;
            IPolyMeshSchema::Sample sample = poly_schema.getValue(sample_index);
            IN3fGeomParam normals_param = poly_schema.getNormalsParam();   
            IV2fGeomParam uvs_param = poly_schema.getUVsParam();         
            ICompoundProperty arbitrary_properties = poly_schema.getArbGeomParams();
            alembic_polymeshes[each_alembic].xform = obj_xform * alembic_polymeshes[each_alembic].xform;

            const bool polymesh_has_uvs = uvs_param.valid() && uvs_param.getValueProperty().valid();
            const bool polymesh_has_vertex_normals = normals_param.valid() && normals_param.getValueProperty().valid();

            size_t num_polygons=sample.getFaceCounts()->size();
            P3fArraySamplePtr positions = sample.getPositions();
            
            std::vector<InputVector3f> vertices;
            std::vector<InputNormal3f> normals;
            std::vector<InputVector2f> texcoords;
            std::vector<VertexBinding> vertex_map;

            for (size_t i = 0, e = positions->size(); i < e; ++i){
                InputPoint3f p = alembic_polymeshes[each_alembic].xform.transform_affine(
                                InputPoint3f( (*positions)[i][0], (*positions)[i][1], (*positions)[i][2] ) );
                p = m_to_world.transform_affine(p);
                if (unlikely(!all(enoki::isfinite(p))))
                    detailed_fail("Alembic mesh contains invalid vertex position data", m_mesh_name);
                m_bbox.expand(p);
                vertices.push_back(p);
            }

            Int32ArraySamplePtr alembic_face_indices = sample.getFaceIndices();
            std::vector<ScalarIndex3> indices;
            indices.reserve( alembic_face_indices->size() );

            Int32ArraySamplePtr vertices_per_face = sample.getFaceCounts();
            size_t mesh_face_index=0;
            ScalarIndex vertex_ctr = 0;

            if (polymesh_has_uvs)
            {
                make_attr<IV2fGeomParam, V2fArraySamplePtr, InputVector2f>(
                    uvs_param, num_polygons, vertices_per_face, texcoords);
                if (flip_tex_coords){
                    for (size_t i = 0; i < texcoords.size(); i++)
                    {
                        InputVector2f uv = texcoords[i];
                        uv.y() = 1.f - uv.y();
                        texcoords[i] = uv;
                    }
                }
            }

            if (!m_disable_vertex_normals && polymesh_has_vertex_normals)
            {
                make_attr<IN3fGeomParam, N3fArraySamplePtr, InputNormal3f>(
                    normals_param, num_polygons, vertices_per_face, normals);
                
                for (size_t i = 0; i < normals.size(); i++)
                {
                    InputNormal3f n = normals[i];
                    n = normalize(m_to_world.transform_affine(n));
                    if ( unlikely(!all(enoki::isfinite(n))) ){
                        Log(Warn,
                            "Invalid vertex normal: %s at vertex %s of %s.", n, i, m_mesh_name);
                    }
                    normals[i] = n;
                }
            }

            size_t uv_ctr=0;
            for (size_t each_face=0; each_face<num_polygons; each_face++){
                ScalarIndex3 key {{ (ScalarIndex) 0, (ScalarIndex) 0, (ScalarIndex) 0 }};
                ScalarIndex3 tri;

                size_t vertex_count_in_face = (*vertices_per_face)[each_face];
                // Reverse face indices for current face aka "flip normals"
                std::vector<ScalarIndex>current_face_indices;
                current_face_indices.reserve( vertex_count_in_face );
                std::vector<InputVector2f> uvs_per_face;
                for (size_t vertex_index = 0; vertex_index<vertex_count_in_face; vertex_index++){
                    current_face_indices.push_back((*alembic_face_indices)[mesh_face_index]);
                    mesh_face_index++;
                }

                // flip the order, do not modify first index, so 0, 1, 2, 3 becomes 0, 3, 2, 1
                // to reverse handedness
                if (m_flip_normals){
                    std::reverse(std::begin(current_face_indices)+1, std::end(current_face_indices));
                }
                
                for (size_t vertex_index=0; vertex_index<vertex_count_in_face;){
                    key[0] = current_face_indices[vertex_index];
                    key[1] = uv_ctr+1;
                    size_t map_index = key[0];
                    
                    if (unlikely(vertex_map.size() < vertices.size()))
                                vertex_map.resize(vertices.size());
                    
                    // Hash table lookup
                    VertexBinding *entry = &vertex_map[map_index];
                    
                    while (entry->key != key && entry->next != nullptr)
                        entry = entry->next;

                    ScalarIndex id;
                    if (entry->key == key) {
                        // Hit
                        id = entry->value;
                    } else {
                        // Miss
                        if (entry->key != ScalarIndex3{{0, 0, 0}}) {
                            entry->next = new VertexBinding();
                            entry = entry->next;
                        }
                        entry->key = key;
                        id = entry->value = vertex_ctr++;
                    }

                    if (vertex_index < 3) {
                        tri[vertex_index] = id;
                    } else {
                        tri[1] = tri[2];
                        tri[2] = id;
                    }
                    vertex_index++;
                    if (vertex_index >= 3){
                        indices.push_back(tri);
                    }
                    uv_ctr++;
                }
            } 

            m_vertex_count = vertex_ctr;
            m_face_count = (ScalarSize)indices.size();
            m_vertex_positions_buf = empty<FloatStorage>(m_vertex_count * 3);
            m_faces_buf = DynamicBuffer<UInt32>::copy(indices.data(), m_face_count * 3);


            if (!m_disable_vertex_normals){
                m_vertex_normals_buf = empty<FloatStorage>(m_vertex_count * 3);   
            }

            if (!texcoords.empty()){
                m_vertex_texcoords_buf = empty<FloatStorage>(m_vertex_count*2);
            }
            
            for (const auto& v_ : vertex_map) {
                const VertexBinding *v = &v_;

                while (v && v->key != ScalarIndex3{{0, 0, 0}}) {
                    InputFloat* position_ptr   = m_vertex_positions_buf.data() + v->value * 3;
                    InputFloat* normal_ptr   = m_vertex_normals_buf.data() + v->value * 3;
                    InputFloat* texcoord_ptr = m_vertex_texcoords_buf.data() + v->value * 2;
                    auto key = v->key;

                    store_unaligned(position_ptr, vertices[key[0]]);

                    if (key[1] && polymesh_has_uvs)
                        store_unaligned(texcoord_ptr, texcoords[key[1] - 1]);

                    if (!m_disable_vertex_normals && polymesh_has_vertex_normals && key[1])
                        store_unaligned(normal_ptr, normals[key[1] - 1]);

                    v = v->next;
                }
            }
            
            // Mesh constructor in mesh.cpp already has managed(), cuda_sync()
            // and set_children() calls, so we don't need to call it here
            m_mesh = new Mesh<Float, Spectrum>(m_mesh_name, m_vertex_count, m_face_count, 
                                    props, polymesh_has_vertex_normals, polymesh_has_uvs);

            m_mesh->vertex_positions_buffer() = m_vertex_positions_buf;
            m_mesh->faces_buffer() = m_faces_buf;
            m_mesh->vertex_normals_buffer() = m_vertex_normals_buf;
            m_mesh->vertex_texcoords_buffer() = m_vertex_texcoords_buf;

            m_mesh->recompute_bbox();
            if (!m_disable_vertex_normals && normals.empty()) {
                Timer timer2;
                m_mesh->recompute_vertex_normals();
                Log(Warn, "\"%s\": computed vertex normals for %s (took %s)", m_name, m_mesh_name,
                    util::time_string(timer2.value()));
            }

            if (arbitrary_properties.valid())
                add_arbitrary_geom_params(arbitrary_properties);    

            m_mesh_objects.push_back(m_mesh);
        }
    }

    std::vector<ref<Object>> expand() const override {       
        return m_mesh_objects;
    }

protected:
    std::string m_mesh_name;

private:
    ref<Mesh<Float, Spectrum>>  m_mesh;
    std::vector<ref<Object>> m_mesh_objects;
    std::string m_shape_name;
    bool m_use_shape_name;

    std::unordered_map<std::string, std::string> known_attr = {
        {"Cd","color"},
        {"color","color"},
        {"transparency","alpha"},
        {"v","velocity"}
    };

    void add_arbitrary_geom_params(ICompoundProperty parent)
    {
        for (size_t i = 0; i < parent.getNumProperties(); ++i){
            const PropertyHeader &property_header = parent.getPropertyHeader(i);

            if (IFloatGeomParam::matches(property_header)){
                process_arbitrary_geom_param<IFloatGeomParam, Float>(
                        parent, property_header);
                        
            }
            else if (IHalfGeomParam::matches(property_header)){
                process_arbitrary_geom_param<IHalfGeomParam, Float >(
                        parent, property_header);
                        
            }
            else if (IDoubleGeomParam::matches(property_header)){
                process_arbitrary_geom_param<IDoubleGeomParam, double>(
                        parent, property_header);
                        
            }
            else if (ICharGeomParam::matches(property_header)){
                process_arbitrary_geom_param<ICharGeomParam, Int8>(
                        parent, property_header);
                        
            }
            else if (IInt16GeomParam::matches(property_header)){
                process_arbitrary_geom_param<IInt16GeomParam, int16_t>(
                        parent, property_header);
            }
            else if (IInt32GeomParam::matches(property_header)){
                process_arbitrary_geom_param<IInt32GeomParam, int32_t>(
                        parent, property_header);
            }
            else if (IInt64GeomParam::matches(property_header)){
                process_arbitrary_geom_param<IInt64GeomParam, Int64>(
                        parent, property_header);
            }
            else if (IStringGeomParam::matches(property_header)){
                // shop_material path
                process_arbitrary_geom_param<IStringGeomParam, std::string>(
                        parent, property_header);
            }
            else if (IV2fGeomParam::matches(property_header)){
                process_arbitrary_geom_param<IV2fGeomParam, Float>(
                        parent, property_header);
            }
            else if (IV2dGeomParam::matches(property_header)){
                process_arbitrary_geom_param<IV2dGeomParam, double>(
                        parent, property_header);
            }
            else if (IV3fGeomParam::matches(property_header)){
                process_arbitrary_geom_param<IV3fGeomParam, Float>(
                        parent, property_header);
            }
            else if (IV3dGeomParam::matches(property_header)){
                process_arbitrary_geom_param<IV3dGeomParam, double>(
                        parent, property_header);
            }
            else if (IN3fGeomParam::matches(property_header)){
                process_arbitrary_geom_param<IN3fGeomParam, Float>(
                        parent, property_header);
            }
            else if (IN3dGeomParam::matches(property_header)){
                process_arbitrary_geom_param<IN3dGeomParam, double>(
                        parent, property_header);
            }
            else if (IC3hGeomParam::matches(property_header)){
                process_arbitrary_geom_param<IC3hGeomParam, Float>(
                        parent, property_header);
            }
            else if (IC3fGeomParam::matches(property_header)){
                // e.g. vertex color attribute
                process_arbitrary_geom_param<IC3fGeomParam, Float>(
                        parent, property_header);
            }
            else if (IC3cGeomParam::matches(property_header)){
                process_arbitrary_geom_param<IC3cGeomParam, uint>(
                        parent, property_header);
            }
            else if (IP3fGeomParam::matches(property_header)){
                process_arbitrary_geom_param<IP3fGeomParam, Float>(
                        parent, property_header);
            }
            else if (IP3dGeomParam::matches(property_header)){
                process_arbitrary_geom_param<IP3dGeomParam, double>(
                        parent, property_header);
            }
            else if (IBoolGeomParam::matches(property_header)){
                process_arbitrary_geom_param<IBoolGeomParam, bool>(
                        parent, property_header);
            }
        }
    }

    template <typename geom_param_t, typename pod_t>
    void process_arbitrary_geom_param(
            ICompoundProperty parent,
            const PropertyHeader & property_header )
    {
        std::string attr_name = property_header.getName();
        geom_param_t param(parent, property_header.getName());
        ISampleSelector sample_selector( m_alembic_time );
        typename geom_param_t::sample_type param_sample;
        param.getExpanded(param_sample, sample_selector);

        // size, e.g. 3 for color; 1 for shop_materialpath
        size_t extent = geom_param_t::prop_type::traits_type::dataType().getExtent();
        const pod_t *values = reinterpret_cast<const pod_t *>(
                param_sample.getVals()->get());

        switch (param_sample.getScope())
        {
            case kVaryingScope:
            case kVertexScope:
                //Point attribute
                if ( !(known_attr.find(attr_name) == known_attr.end()) )
                    attr_name = "vertex_"+known_attr[attr_name];
                m_mesh->add_attribute(attr_name, extent, 
                                    FloatStorage::copy(values, m_vertex_count * extent));
                break;
            case kFacevaryingScope:{
                // Vertex attribute
                if ( !(known_attr.find(attr_name) == known_attr.end()) )
                    attr_name = "vertex_"+known_attr[attr_name];

                bool is_vertex_attr = attr_name.find("vertex_") == 0;
                if(is_vertex_attr){
                    m_mesh->add_attribute(attr_name, extent, 
                                    FloatStorage::copy(values, m_vertex_count * extent));
                }
                else{
                    Log(Info, "Skipping vertex attribute \"%s\" on \"%s\" as it does not start with \"vertex_\"", attr_name, m_mesh_name);
                }
                break;
            }
            case kConstantScope:
            case kUnknownScope:
            case kUniformScope:{
                // Known attribute
                if ( !(known_attr.find(attr_name) == known_attr.end()) )
                    attr_name = "face_"+known_attr[attr_name];

                bool is_face_attr = attr_name.find("face_") == 0;
                if(is_face_attr){
                    m_mesh->add_attribute(attr_name, extent, 
                                    FloatStorage::copy(values, m_face_count * extent));
                }
                else{
                    Log(Info, "Skipping face attribute \"%s\" on \"%s\" as it does not start with \"face_\"", attr_name, m_mesh_name);
                }
            }
        }
    }
    
    template <typename geom_param_t, typename ptr_t, typename mitsuba_t>
    void make_attr(geom_param_t param,
                size_t num_polygons, 
                Int32ArraySamplePtr &vertices_per_face, 
                std::vector<mitsuba_t> &data_array )
    {
        typename geom_param_t::sample_type param_sample;
        param.getExpanded(param_sample, ISampleSelector( m_alembic_time )); //abc time, for now take value at 0 time
        size_t extent = geom_param_t::prop_type::traits_type::dataType().getExtent();
        ptr_t param_sample_values = param_sample.getVals();

        size_t data_index=0;
        for (size_t each_face=0; each_face<num_polygons; each_face++){
            // This is to properly handle the polygons with many vertices, not only triangles.
            size_t vertex_count_in_face = (*vertices_per_face)[each_face];
            std::vector<mitsuba_t> data_per_face;

            for (size_t vertex_index=0; vertex_index<vertex_count_in_face; vertex_index++, data_index++){
                mitsuba_t n;
                for (size_t j = 0; j < extent; j++){
                    n[j] = (*param_sample_values)[data_index][j];
                }
                data_per_face.push_back(n);
            }
            if (m_flip_normals)
                std::reverse(std::begin(data_per_face)+1, std::end(data_per_face));

            for (size_t j=0; j<data_per_face.size(); j++){
                data_array.push_back( data_per_face[j] );
            }
        }
    }

    void find_children_recursively(const IObject& obj,
        index_t sample_index,
        ScalarTransform4f &t, std::vector<AlembicPolymesh>& polymeshes)
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
                    child_transform = child_transform * obj_xform;
                }
                else{
                    child_transform = obj_xform;
                }
            }
            else if (IPolyMesh::matches(child.getMetaData())){
                bool mesh_visible = !m_load_visible || GetVisibilityProperty(child) == (bool)kVisibilityVisible;
                bool add_mesh = true;
                IPolyMesh poly_obj(child, kWrapExisting);
                AlembicPolymesh polymesh;
                polymesh.schema = poly_obj.getSchema();
                polymesh.id = child.getFullName();
                polymesh.xform = t;

                if (m_use_shape_name){
                    // check if shape name is in alembic's mesh name to avoid
                    // typing full names. E.g. "sphere" instead of /sphere_object1/sphere1
                    // if shape name from Xml file is not in mesh name, skip mesh
                    if (child.getFullName().find(m_shape_name) == std::string::npos)
                        add_mesh =false;
                }

                if (mesh_visible && add_mesh)
                    polymeshes.push_back(polymesh);

                continue;
            }
            // continue traversing alembic tree for more child objects
            find_children_recursively(child, sample_index, child_transform, polymeshes);
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

    MTS_DECLARE_CLASS()
};

MTS_IMPLEMENT_CLASS_VARIANT(AlembicMesh, Mesh)
MTS_EXPORT_PLUGIN(AlembicMesh, "Alembic Mesh")
NAMESPACE_END(mitsuba)
