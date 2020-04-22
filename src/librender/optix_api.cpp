#if defined(_WIN32)
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

#include "optix_api.h"
#include <mitsuba/core/logger.h>

#if defined(MTS_DYNAMIC_OPTIX)
// Driver API
const char* (*optixGetErrorName)(OptixResult) = nullptr;
const char* (*optixGetErrorString)(OptixResult) = nullptr;
OptixResult (*optixDeviceContextCreate)(CUcontext, const OptixDeviceContextOptions*, OptixDeviceContext*) = nullptr;
OptixResult (*optixDeviceContextDestroy)(OptixDeviceContext) = nullptr;
OptixResult (*optixModuleCreateFromPTX)(OptixDeviceContext, const OptixModuleCompileOptions*, const OptixPipelineCompileOptions*, const char*, size_t, char*, size_t*, OptixModule*) = nullptr;
OptixResult (*optixModuleDestroy)(OptixModule) = nullptr;
OptixResult (*optixProgramGroupCreate)(OptixDeviceContext, const OptixProgramGroupDesc*, unsigned int, const OptixProgramGroupOptions*, char*, size_t*, OptixProgramGroup*) = nullptr;
OptixResult (*optixProgramGroupDestroy)(OptixProgramGroup) = nullptr;
OptixResult (*optixPipelineCreate)(OptixDeviceContext, const OptixPipelineCompileOptions*, const OptixPipelineLinkOptions*, const OptixProgramGroup*, unsigned int, char*, size_t*, OptixPipeline*) = nullptr;
OptixResult (*optixPipelineDestroy)(OptixPipeline) = nullptr;
OptixResult (*optixAccelComputeMemoryUsage)(OptixDeviceContext, const OptixAccelBuildOptions*, const OptixBuildInput*, unsigned int, OptixAccelBufferSizes*) = nullptr;
OptixResult (*optixAccelBuild)(OptixDeviceContext, CUstream, const OptixAccelBuildOptions*, const OptixBuildInput*,unsigned int, CUdeviceptr, size_t, CUdeviceptr, size_t, OptixTraversableHandle*, const OptixAccelEmitDesc*, unsigned int) = nullptr;
OptixResult (*optixAccelCompact)(OptixDeviceContext, CUstream,  OptixTraversableHandle, CUdeviceptr, size_t, OptixTraversableHandle*) = nullptr;
OptixResult (*optixSbtRecordPackHeader)(OptixProgramGroup, void*) = nullptr;
OptixResult (*optixLaunch)(OptixPipeline, CUstream, CUdeviceptr, size_t, const OptixShaderBindingTable*, unsigned int, unsigned int, unsigned int) = nullptr;
OptixResult (*optixQueryFunctionTable)(int, unsigned int, OptixQueryFunctionTableOptions*, const void**, void*, size_t);
#endif

#if defined(MTS_DYNAMIC_OPTIX)
static void *optix_handle = nullptr;
#endif

static bool optix_init_attempted = false;
static bool optix_init_success = false;

NAMESPACE_BEGIN(mitsuba)

bool optix_init() {
    if (optix_init_attempted)
        return optix_init_success;
    optix_init_attempted = true;

#if defined(MTS_DYNAMIC_OPTIX)
    optix_handle = nullptr;

# if defined(_WIN32)
#    define dlsym(ptr, name) GetProcAddress((HMODULE) ptr, name)
    const char* optix_fname = "nvoptix.dll";
# elif defined(__linux__)
    const char *optix_fname  = "libnvoptix.so.1";
# else
    const char *optix_fname  = "libnvoptix.dylib";
# endif

#  if !defined(_WIN32)
    // Don't dlopen libnvoptix.so if it was loaded by another library
    if (dlsym(RTLD_NEXT, "optixLaunch"))
        optix_handle = RTLD_NEXT;

    if (!optix_handle)
        optix_handle = dlopen(optix_fname, RTLD_NOW);
#  else
    char system_path[MAX_PATH];
    if (GetSystemDirectoryA(system_path, MAX_PATH) != 0) {
        strcat(system_path, "\\");
        strcat(system_path, optix_fname);
        optix_handle = LoadLibraryA(system_path);
    }
#  endif

    if (!optix_handle) {
        Log(LogLevel::Error,
            "optix_init(): %s could not be loaded.", optix_fname);
        return false;
    }

    // Load optixQueryFunctionTable from library
    optixQueryFunctionTable = decltype(optixQueryFunctionTable)(
        dlsym(optix_handle, "optixQueryFunctionTable"));

    if (!optixQueryFunctionTable) {
        Log(LogLevel::Error,
            "optix_init(): could not find symbol optixQueryFunctionTable");
        return false;
    }

    void *function_table[36];
    optixQueryFunctionTable(OPTIX_ABI_VERSION, 0, 0, 0, &function_table, sizeof(function_table));

    #define LOAD(name, index) name = (decltype(name)) function_table[index];

    LOAD(optixGetErrorName, 0);
    LOAD(optixGetErrorString, 1);
    LOAD(optixDeviceContextCreate, 2);
    LOAD(optixDeviceContextDestroy, 3);
    LOAD(optixModuleCreateFromPTX, 12);
    LOAD(optixModuleDestroy, 13);
    LOAD(optixProgramGroupCreate, 14);
    LOAD(optixProgramGroupDestroy, 15);
    LOAD(optixPipelineCreate, 17);
    LOAD(optixPipelineDestroy, 18);
    LOAD(optixAccelComputeMemoryUsage, 20);
    LOAD(optixAccelBuild, 21);
    LOAD(optixAccelCompact, 25);
    LOAD(optixSbtRecordPackHeader, 27);
    LOAD(optixLaunch, 28);
#else
    rt_check(optixInit());
#endif
    optix_init_success = true;
    return true;
}

void optix_shutdown() {
#if defined(MTS_DYNAMIC_OPTIX)
    if (!optix_init_success)
        return;

    #define Z(x) x = nullptr
    Z(optixGetErrorName); Z(optixGetErrorString); Z(optixDeviceContextCreate);
    Z(optixDeviceContextDestroy); Z(optixModuleCreateFromPTX); Z(optixModuleDestroy);
    Z(optixProgramGroupCreate); Z(optixProgramGroupDestroy);
    Z(optixPipelineCreate); Z(optixPipelineDestroy); Z(optixAccelComputeMemoryUsage);
    Z(optixAccelBuild); Z(optixAccelCompact); Z(optixSbtRecordPackHeader);
    Z(optixLaunch); Z(optixQueryFunctionTable);
    #undef Z

#if !defined(_WIN32)
    if (optix_handle != RTLD_NEXT)
        dlclose(optix_handle);
#else
    FreeLibrary((HMODULE) optix_handle);
#endif

    optix_handle = nullptr;
#endif

    optix_init_success = false;
    optix_init_attempted = false;
}

void __rt_check(OptixResult errval, const char *file, const int line) {
    if (errval != OPTIX_SUCCESS) {
        const char *message = optixGetErrorString(errval);
        if (errval == 1546)
            message = "Failed to load OptiX library! Very likely, your NVIDIA graphics "
                "driver is too old and not compatible with the version of OptiX that is "
                "being used. In particular, OptiX 6.5 requires driver revision R435.80 or newer.";
        fprintf(stderr, "rt_check(): OptiX API error = %04d (%s) in %s:%i.\n",
                (int) errval, message, file, line);
        exit(EXIT_FAILURE);
    }
}

void __rt_check_log(OptixResult errval, const char *file, const int line) {
    if (errval != OPTIX_SUCCESS) {
        const char *message = optixGetErrorString(errval);
        fprintf(stderr, "rt_check(): OptiX API error = %04d (%s) in %s:%i.\n",
                (int) errval, message, file, line);
        fprintf(stderr, "\tLog: %s%s", optix_log_buffer,
                optix_log_buffer_size > sizeof(optix_log_buffer) ? "<TRUNCATED>" : "");
        exit(EXIT_FAILURE);
    }
}

NAMESPACE_END(mitsuba)