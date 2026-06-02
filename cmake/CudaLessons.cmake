# CudaLessons.cmake — 注册 src/ 与 benchmarks/ 下的 .cu 可执行目标

set(CUDAPRO_CUBLAS_SOURCES
  lesson11_cublas.cu
  lesson15_cublas_gemm.cu
  lesson16_tensor_gemm.cu
)

set(CUDAPRO_CUTLASS_SOURCES
  lesson17_qkv_rmsnorm.cu
)

function(cudapro_target_name src out_var)
  get_filename_component(_exe "${src}" NAME_WE)
  set("${out_var}" "${_exe}" PARENT_SCOPE)
endfunction()

function(cudapro_add_cuda_executable src_file)
  get_filename_component(_src_name "${src_file}" NAME)
  cudapro_target_name("${_src_name}" _exe)

  if(TARGET "${_exe}")
    return()
  endif()

  add_executable("${_exe}" "${src_file}")
  set_target_properties("${_exe}" PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )
  target_include_directories("${_exe}" PRIVATE
    "${CUDAPRO_INCLUDE_DIR}"
    "${CUDAPRO_SOURCE_DIR}"
  )
  target_link_libraries("${_exe}" PRIVATE CUDA::cudart)

  if(_src_name IN_LIST CUDAPRO_CUBLAS_SOURCES)
    target_link_libraries("${_exe}" PRIVATE CUDA::cublas)
  endif()

  if(_src_name IN_LIST CUDAPRO_CUTLASS_SOURCES)
    if(NOT EXISTS "${CUTLASS_DIR}/include/cutlass/cutlass.h")
      message(WARNING "SKIP ${_exe}: CUTLASS not found at ${CUTLASS_DIR}")
      return()
    endif()
    target_include_directories("${_exe}" PRIVATE
      "${CUTLASS_DIR}/include"
      "${CUTLASS_DIR}/tools/util/include"
    )
  endif()

  set_property(GLOBAL APPEND PROPERTY CUDAPRO_CUDA_TARGETS "${_exe}")
endfunction()

function(cudapro_register_cuda_sources source_dir)
  file(GLOB _sources CONFIGURE_DEPENDS "${source_dir}/*.cu")
  foreach(_src ${_sources})
    cudapro_add_cuda_executable("${_src}")
  endforeach()
endfunction()

function(cudapro_add_all_cuda_target)
  get_property(_targets GLOBAL PROPERTY CUDAPRO_CUDA_TARGETS)
  if(_targets)
    add_custom_target(all_cuda DEPENDS ${_targets})
  else()
    add_custom_target(all_cuda)
    message(WARNING "No CUDA lesson targets were registered.")
  endif()
endfunction()
