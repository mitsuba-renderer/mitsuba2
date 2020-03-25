set(MTS_PLUGIN_PREFIX "integrators")

add_plugin(depth   depth.cpp)
add_plugin(direct  direct.cpp)
add_plugin(path    path.cpp)
add_plugin(aov     aov.cpp)
add_plugin(stokes  stokes.cpp)
add_plugin(moment  moment.cpp)
add_plugin(volpath volpath.cpp)
add_plugin(volpathsimple volpath-simple.cpp)
add_plugin(pathreparam   path-reparam-old.cpp)
add_plugin(pathreparamnew   path-reparam.cpp)

# Register the test directory
add_tests(${CMAKE_CURRENT_SOURCE_DIR}/tests)
