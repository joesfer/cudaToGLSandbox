cmake_minimum_required(VERSION 3.10)

project(CudaGLSandbox VERSION 0.1.0 LANGUAGES CXX CUDA)

# Add the additional module paths

list(INSERT CMAKE_MODULE_PATH 0 "${PROJECT_SOURCE_DIR}/cmake")

if(EXISTS "$ENV{CMAKE_MODULE_PATH}")
	list(INSERT CMAKE_MODULE_PATH 1 "$ENV{CMAKE_MODULE_PATH}")
endif()

# Set the install prefix

string(APPEND CMAKE_INSTALL_PREFIX "/" "${PROJECT_NAME}-${PROJECT_VERSION}")

# Set the options for the build type

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
	set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}/dist" CACHE PATH
		"Install path prefix." FORCE)
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE RelWidthDebInfo CACHE STRING
		"Options are Debug, Release, MinSizeRel or RelWithDebInfo." FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release
		MinSizeRel RelWithDebInfo)
endif()

# Create a default setup for targets

function(target_default_setup TARGET)
	target_compile_options(${TARGET} PRIVATE
		# Enable for GCC only, not NVCC
		$<$<COMPILE_LANGUAGE:CXX>:-Wall>
		)
	target_compile_features(${TARGET} PRIVATE cxx_std_17)

	set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)

	# Allow device functions to be in separate compilation units
	set_target_properties(${TARGET} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

	# Cuda C++ standard
	set_target_properties(${TARGET} PROPERTIES CUDA_STANDARD 11)
endfunction()

# Configure default include directories

set(PRIVATE_INCLUDE_DIRS
	${PROJECT_SOURCE_DIR}/src
	${PROJECT_BINARY_DIR}/src
	)

function(target_default_includes TARGET)
	target_include_directories(${TARGET} PUBLIC
		$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
		$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
		$<INSTALL_INTERFACE:include>
		)

	target_include_directories(${TARGET} PRIVATE
		${PRIVATE_INCLUDE_DIRS}
		)
endfunction()

# Add subdirectories
add_subdirectory(src)
