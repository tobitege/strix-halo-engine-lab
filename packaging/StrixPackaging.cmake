include_guard(GLOBAL)

# Install rules and CPack packages for every build mode (host, HIP, CUDA Qwen, CUDA DeepSeek).
# The root CMakeLists.txt includes this file deferred (cmake_language(DEFER)), so every target exists here, also in
# the CUDA modes that return() early. The lists below name user-facing executables only: tests, probes, benches,
# lab/, artifacts/, weights, mmproj files and captures are never installed. A listed target that this build mode
# does not create is skipped. Release flow and support matrix: docs/install.md.
include(GNUInstallDirs)
include("${CMAKE_CURRENT_LIST_DIR}/StrixSupportMatrix.cmake")

get_filename_component(_strix_pkg_src "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(_strix_pkg_dir "${CMAKE_CURRENT_LIST_DIR}")
set(_strix_pkg_out "${CMAKE_BINARY_DIR}/packaging")

option(STRIX_PACKAGING "Release packaging build: fail when a required executable of this build mode is missing" OFF)
option(STRIX_PACKAGE_REQUIRE_TAG "Fail unless HEAD carries the tag v${PROJECT_VERSION}" OFF)
set(STRIX_PACKAGE_INSTALL_RPATH "" CACHE STRING "Install RPATH of backend executables (e.g. /opt/rocm/lib); empty: none")

# ---- backend of this tree -------------------------------------------------------------------------------------
if(STRIX_CUDA_QWEN_ONLY)
    set(STRIX_PACKAGE_BACKEND cuda)
    set(_strix_pkg_flavor cuda-qwen)
    set(_strix_pkg_required strix-cuda-qwen-serve)
elseif(STRIX_CUDA_DS41_ONLY)
    set(STRIX_PACKAGE_BACKEND cuda)
    set(_strix_pkg_flavor cuda-ds41)
    set(_strix_pkg_required strix-cuda-ds41-replica)
elseif(STRIX_ENABLE_HIP_RUNTIME)
    set(STRIX_PACKAGE_BACKEND hip)
    set(_strix_pkg_flavor hip)
    set(_strix_pkg_required strix-hip-decode-token)
else()
    set(STRIX_PACKAGE_BACKEND host)
    set(_strix_pkg_flavor host)
    set(_strix_pkg_required strix-serve strix-skeleton-serve strix-inspect-gguf)
endif()

set(STRIX_PACKAGE_CORE_TARGETS strix-serve strix-skeleton-serve strix-memory-plan strix-model-describe
    strix-qwen-chain-plan strix-deepseek41-expert-server)
set(STRIX_PACKAGE_TOOLS_TARGETS strix-inspect-gguf strix-tokenize strix-model-manifest strix-deepseek41-inspect
    strix-deepseek41-model-describe strix-ds41-prepare-load strix-gemma4-tokenize)
set(STRIX_PACKAGE_HIP_TARGETS strix-hip-decode-token strix-deepseek41-rank strix-deepseek41-serve
    strix-deepseek41-prefill strix-deepseek41-prefix strix-deepseek41-batch strix-deepseek41-pipeline-front
    strix-deepseek41-dspark-rank strix-ds41-prefill-pair strix-minimax-rank strix-minimax-serve strix-glm5-rank
    strix-glm5-serve strix-gemma4-decode strix-gemma4-serve)
set(STRIX_PACKAGE_CUDA_TARGETS strix-cuda-qwen-serve strix-cuda-qwen-prefill-service strix-cuda-ds41-replica
    strix-cuda-ds41-decode strix-cuda-ds41-prefill strix-cuda-ds41-prefill-service strix-cuda-ds41-pipeline-front
    strix-cuda-ds41-multi-front)

if(STRIX_PACKAGING)
    foreach(_t IN LISTS _strix_pkg_required)
        if(NOT TARGET ${_t})
            message(FATAL_ERROR "STRIX_PACKAGING: ${_t} is required in the ${_strix_pkg_flavor} build mode")
        endif()
    endforeach()
endif()

# ---- executables ----------------------------------------------------------------------------------------------
set(_strix_pkg_installed "")
function(_strix_pkg_install_executables component)
    foreach(_t IN LISTS ARGN)
        if(NOT TARGET ${_t})
            continue()
        endif()
        get_target_property(_type ${_t} TYPE)
        if(NOT _type STREQUAL "EXECUTABLE")
            continue()
        endif()
        if(NOT STRIX_PACKAGE_INSTALL_RPATH STREQUAL "" AND component MATCHES "^(hip|cuda)$")
            set_target_properties(${_t} PROPERTIES INSTALL_RPATH "${STRIX_PACKAGE_INSTALL_RPATH}")
        endif()
        install(TARGETS ${_t} RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT ${component})
        list(APPEND _strix_pkg_installed "${component}:${_t}")
    endforeach()
    set(_strix_pkg_installed "${_strix_pkg_installed}" PARENT_SCOPE)
endfunction()
_strix_pkg_install_executables(core ${STRIX_PACKAGE_CORE_TARGETS})
_strix_pkg_install_executables(tools ${STRIX_PACKAGE_TOOLS_TARGETS})
_strix_pkg_install_executables(hip ${STRIX_PACKAGE_HIP_TARGETS})
_strix_pkg_install_executables(cuda ${STRIX_PACKAGE_CUDA_TARGETS})

# ---- components packaged by this tree -------------------------------------------------------------------------
# A host tree ships core + tools + docs; a backend tree ships only its backend package (Linux), so one host build
# serves every backend. On Windows the CUDA build is one self-contained archive / installer.
if(WIN32)
    set(_strix_pkg_components cuda core tools docs)
elseif(STRIX_PACKAGE_BACKEND STREQUAL "host")
    set(_strix_pkg_components core tools docs)
else()
    set(_strix_pkg_components ${STRIX_PACKAGE_BACKEND})
endif()

# ---- notices with every package -------------------------------------------------------------------------------
# Linux: one copy per package (each can be installed alone); Windows: the single archive carries one copy.
if(WIN32)
    set(_strix_pkg_notice_components cuda)
else()
    set(_strix_pkg_notice_components core tools docs hip cuda)
endif()
foreach(_c IN LISTS _strix_pkg_notice_components)
    install(FILES "${_strix_pkg_src}/LICENSE" "${_strix_pkg_dir}/THIRD-PARTY.md"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/strix-${_c}" COMPONENT ${_c})
    install(DIRECTORY "${_strix_pkg_src}/LICENSES/" DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/strix-${_c}/LICENSES"
        COMPONENT ${_c})
endforeach()

# ---- data and docs --------------------------------------------------------------------------------------------
install(DIRECTORY "${_strix_pkg_src}/models/" DESTINATION "${CMAKE_INSTALL_DATADIR}/strix/models" COMPONENT core
    FILES_MATCHING PATTERN "*.toml")
install(DIRECTORY "${_strix_pkg_src}/examples/skeleton_engine/models/"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/strix/examples/skeleton_engine/models" COMPONENT core
    FILES_MATCHING PATTERN "*.toml")
if(WIN32)
    install(FILES "${_strix_pkg_src}/tools/bitcalc_cmp.py" DESTINATION "${CMAKE_INSTALL_DATADIR}/strix/tools"
        COMPONENT tools)
else()
    install(PROGRAMS "${_strix_pkg_src}/tools/bitcalc_cmp.py" DESTINATION "${CMAKE_INSTALL_BINDIR}"
        RENAME strix-bitcalc-cmp COMPONENT tools)
endif()
install(FILES "${_strix_pkg_src}/README.md" "${_strix_pkg_src}/README.es.md"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/strix-docs" COMPONENT docs)
# User documentation only: the coordination log and decision records stay in the repository.
install(DIRECTORY "${_strix_pkg_src}/docs/" DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/strix-docs/docs" COMPONENT docs
    FILES_MATCHING PATTERN "*.md" PATTERN "*.svg" PATTERN "*.png" PATTERN "*.json"
    PATTERN "coord" EXCLUDE PATTERN "decisions" EXCLUDE)
# Post-install smoke test, one copy per package so each package can be checked alone.
if(WIN32)
    # Launcher and smoke share the installed backend's actual CLI contracts.
    if(STRIX_CUDA_DS41_ONLY)
        set(_strix_server "strix-cuda-ds41-replica")
        set(_strix_inspect "--print-config")
        set(_strix_health false)
    else()
        set(_strix_server "strix-cuda-qwen-serve")
        set(_strix_inspect "--print-effective-config")
        set(_strix_health true)
    endif()
    file(WRITE "${_strix_pkg_out}/server.json"
        "{\"binary\":\"${_strix_server}\",\"inspect\":\"${_strix_inspect}\",\"health\":${_strix_health}}\n")
    install(FILES "${_strix_pkg_out}/server.json" DESTINATION "${CMAKE_INSTALL_DATADIR}/strix/support" COMPONENT cuda)
    install(FILES "${_strix_pkg_dir}/windows/smoke.ps1" DESTINATION "." COMPONENT cuda)
else()
    foreach(_c IN ITEMS core tools hip cuda)
        install(PROGRAMS "${_strix_pkg_dir}/smoke.sh" DESTINATION "${CMAKE_INSTALL_DATADIR}/strix/support"
            RENAME "smoke-${_c}.sh" COMPONENT ${_c})
    endforeach()
endif()

# Expected executable sets come from the same target selection as the install rules.
set(_strix_pkg_without_help strix-inspect-gguf strix-model-describe strix-model-manifest)
foreach(_c IN ITEMS core tools hip cuda)
    set(_expected "")
    set(_without_help "")
    foreach(_entry IN LISTS _strix_pkg_installed)
        if(_entry MATCHES "^${_c}:(.+)$")
            string(APPEND _expected "${CMAKE_MATCH_1}${CMAKE_EXECUTABLE_SUFFIX}\n")
            if(CMAKE_MATCH_1 IN_LIST _strix_pkg_without_help)
                string(APPEND _without_help "${CMAKE_MATCH_1}${CMAKE_EXECUTABLE_SUFFIX}\n")
            endif()
        endif()
    endforeach()
    if(_c STREQUAL "tools" AND NOT WIN32)
        string(APPEND _expected "strix-bitcalc-cmp\n")
    endif()
    if(NOT _expected STREQUAL "")
        file(WRITE "${_strix_pkg_out}/without-help-${_c}.txt" "${_without_help}")
        install(FILES "${_strix_pkg_out}/without-help-${_c}.txt"
            DESTINATION "${CMAKE_INSTALL_DATADIR}/strix/support" COMPONENT ${_c})
        file(WRITE "${_strix_pkg_out}/executables-${_c}.txt" "${_expected}")
        install(FILES "${_strix_pkg_out}/executables-${_c}.txt"
            DESTINATION "${CMAKE_INSTALL_DATADIR}/strix/support" COMPONENT ${_c})
    endif()
endforeach()

# ---- build receipt (backend, archs and their qualification) ---------------------------------------------------
find_package(Git QUIET)
set(_strix_pkg_sha "${STRIX_SOURCE_GIT_SHA}")
set(_strix_pkg_tag "")
if(GIT_FOUND AND EXISTS "${_strix_pkg_src}/.git")
    execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD WORKING_DIRECTORY "${_strix_pkg_src}"
        OUTPUT_VARIABLE _strix_pkg_sha OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    execute_process(COMMAND "${GIT_EXECUTABLE}" tag --points-at HEAD --list "v*" WORKING_DIRECTORY "${_strix_pkg_src}"
        OUTPUT_VARIABLE _strix_pkg_tag OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
endif()
string(SUBSTRING "${_strix_pkg_sha}0000000000000" 0 12 _strix_pkg_short)
if(_strix_pkg_tag STREQUAL "v${PROJECT_VERSION}" OR "v${PROJECT_VERSION}" IN_LIST _strix_pkg_tag)
    set(_strix_pkg_release_default 1)
else()
    if(STRIX_PACKAGE_REQUIRE_TAG)
        message(FATAL_ERROR "STRIX_PACKAGE_REQUIRE_TAG: HEAD is not tagged v${PROJECT_VERSION} (tags: '${_strix_pkg_tag}')")
    endif()
    # Untagged builds sort before the tagged release: 0.1.0-0.git<sha> < 0.1.0-1.
    set(_strix_pkg_release_default "0.git${_strix_pkg_short}")
endif()
# Recomputed at every configure (a new commit or tag changes it); a non-empty cache value overrides it.
set(STRIX_PACKAGE_RELEASE_OVERRIDE "" CACHE STRING "Package release number (DEB/RPM); empty: from the git tag")
if(STRIX_PACKAGE_RELEASE_OVERRIDE STREQUAL "")
    set(STRIX_PACKAGE_RELEASE "${_strix_pkg_release_default}")
else()
    set(STRIX_PACKAGE_RELEASE "${STRIX_PACKAGE_RELEASE_OVERRIDE}")
endif()

set(_strix_pkg_receipt "format: strix-package-build/1\nversion: ${PROJECT_VERSION}-${STRIX_PACKAGE_RELEASE}\n")
string(APPEND _strix_pkg_receipt "revision: ${_strix_pkg_sha}\nbuild-mode: ${_strix_pkg_flavor}\n")
if(STRIX_PACKAGE_BACKEND STREQUAL "hip")
    strix_support_lines(hip _strix_pkg_arch_lines ${STRIX_HIP_ARCHS})
elseif(STRIX_PACKAGE_BACKEND STREQUAL "cuda")
    strix_support_lines(cuda _strix_pkg_arch_lines ${CMAKE_CUDA_ARCHITECTURES})
else()
    set(_strix_pkg_arch_lines "host: no GPU code")
endif()
string(REPLACE ";" ", " _strix_pkg_arch_summary "${_strix_pkg_arch_lines}")
string(REPLACE ";" "\n  " _strix_pkg_arch_block "${_strix_pkg_arch_lines}")
string(APPEND _strix_pkg_receipt "archs:\n  ${_strix_pkg_arch_block}\n")
string(APPEND _strix_pkg_receipt "qualified: tuning and numerics measured on that hardware; unqualified: compiled only\n")
string(REPLACE ";" "\n  " _strix_pkg_exe_block "${_strix_pkg_installed}")
string(APPEND _strix_pkg_receipt "executables:\n  ${_strix_pkg_exe_block}\n")
file(WRITE "${_strix_pkg_out}/build-${_strix_pkg_flavor}.txt" "${_strix_pkg_receipt}")
foreach(_c IN LISTS _strix_pkg_notice_components)
    if(NOT _c IN_LIST _strix_pkg_components)
        continue()
    endif()
    install(FILES "${_strix_pkg_out}/build-${_strix_pkg_flavor}.txt"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/strix/support" RENAME "build-${_c}.txt" COMPONENT ${_c})
endforeach()

# ---- serve configuration and service --------------------------------------------------------------------------
if(WIN32)
    # The MSVC C++ runtime (msvcp140, vcruntime140*) next to the executables, from the local Visual Studio
    # redistributable folder: the ZIP then needs only the NVIDIA driver (cudart is linked statically).
    set(CMAKE_INSTALL_SYSTEM_RUNTIME_DESTINATION "${CMAKE_INSTALL_BINDIR}")
    set(CMAKE_INSTALL_SYSTEM_RUNTIME_COMPONENT cuda)
    set(CMAKE_INSTALL_UCRT_LIBRARIES OFF)
    include(InstallRequiredSystemLibraries)
    install(FILES "${_strix_pkg_dir}/serve.toml" "${_strix_pkg_dir}/windows/serve.args"
        "${_strix_pkg_dir}/windows/devices.example.toml" DESTINATION "${CMAKE_INSTALL_SYSCONFDIR}/strix" COMPONENT cuda)
    install(PROGRAMS "${_strix_pkg_dir}/windows/install.ps1" "${_strix_pkg_dir}/windows/uninstall.ps1"
        "${_strix_pkg_dir}/windows/strix-start.ps1" DESTINATION "." COMPONENT cuda)
else()
    set(STRIX_PKG_BINDIR "${CMAKE_INSTALL_FULL_BINDIR}")
    set(STRIX_PKG_SYSCONFDIR "${CMAKE_INSTALL_FULL_SYSCONFDIR}/strix")
    configure_file("${_strix_pkg_dir}/linux/strix.service.in" "${_strix_pkg_out}/strix.service" @ONLY)
    # Absolute /etc/strix when the prefix is /usr (GNUInstallDirs), <prefix>/etc/strix otherwise.
    install(FILES "${_strix_pkg_dir}/serve.toml" "${_strix_pkg_dir}/linux/strix.env"
        DESTINATION "${CMAKE_INSTALL_FULL_SYSCONFDIR}/strix" COMPONENT hip)
    install(FILES "${_strix_pkg_out}/strix.service" DESTINATION "lib/systemd/system" COMPONENT hip)
    install(FILES "${_strix_pkg_dir}/linux/sysusers-strix.conf" DESTINATION "lib/sysusers.d" RENAME strix.conf
        COMPONENT hip)
endif()

# ---- host test: the support matrix agrees with the arch capability layer ----------------------------------------
if(BUILD_TESTING AND NOT STRIX_PACKAGE_BACKEND STREQUAL "cuda")
    add_executable(strix-package-support-tests "${_strix_pkg_src}/tests/package_support_tests.cpp")
    target_include_directories(strix-package-support-tests PRIVATE "${_strix_pkg_src}/include")
    target_compile_features(strix-package-support-tests PRIVATE cxx_std_20)
    string(REPLACE ";" "," _known "${STRIX_HIP_KNOWN_ARCHS}")
    string(REPLACE ";" "," _qual "${STRIX_HIP_QUALIFIED_ARCHS}")
    target_compile_definitions(strix-package-support-tests PRIVATE
        "STRIX_PKG_HIP_KNOWN=\"${_known}\"" "STRIX_PKG_HIP_QUALIFIED=\"${_qual}\"")
    add_test(NAME strix-package-support COMMAND strix-package-support-tests)
    set_tests_properties(strix-package-support PROPERTIES LABELS "cpu;packaging" TIMEOUT 30)
endif()

if(NOT PROJECT_IS_TOP_LEVEL)
    return()
endif()

# ---- CPack ----------------------------------------------------------------------------------------------------
set(CPACK_PACKAGE_NAME "strix")
set(CPACK_PACKAGE_VENDOR "strix-halo-engine-lab")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_CONTACT "strix-halo-engine-lab maintainers <19479239+hyukudan@users.noreply.github.com>")
set(CPACK_PACKAGE_HOMEPAGE_URL "https://github.com/hyukudan/strix-halo-engine-lab")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Strix inference engine: exact local LLM serving on AMD HIP and NVIDIA CUDA")
set(CPACK_RESOURCE_FILE_LICENSE "${_strix_pkg_src}/LICENSE")
set(CPACK_PACKAGE_CHECKSUM SHA256)
set(CPACK_COMPONENTS_ALL ${_strix_pkg_components})
set(CPACK_THREADS 0)

set(CPACK_COMPONENT_CORE_DISPLAY_NAME "Strix core")
set(CPACK_COMPONENT_CORE_DESCRIPTION "Shared serve shell (strix-serve), skeleton health server, memory planner and model registry.")
set(CPACK_COMPONENT_TOOLS_DISPLAY_NAME "Strix tools")
set(CPACK_COMPONENT_TOOLS_DESCRIPTION "GGUF inspection, tokenizer, model manifest and capture comparison tools.")
set(CPACK_COMPONENT_DOCS_DISPLAY_NAME "Strix documentation")
set(CPACK_COMPONENT_DOCS_DESCRIPTION "User documentation (English and Spanish) and license notices.")
set(CPACK_COMPONENT_HIP_DISPLAY_NAME "Strix HIP backend")
set(CPACK_COMPONENT_HIP_DESCRIPTION "AMD HIP/ROCm engines and servers. Targets: ${_strix_pkg_arch_summary}.")
set(CPACK_COMPONENT_CUDA_DISPLAY_NAME "Strix CUDA backend")
set(CPACK_COMPONENT_CUDA_DESCRIPTION "NVIDIA CUDA engines and servers (${_strix_pkg_flavor}). Targets: ${_strix_pkg_arch_summary}.")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    set(_strix_pkg_arch x86_64)
else()
    string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _strix_pkg_arch)
endif()

if(WIN32)
    set(_strix_pkg_generators ZIP)
    find_program(STRIX_MAKENSIS makensis)
    find_program(STRIX_WIX_CANDLE candle)
    if(STRIX_MAKENSIS)
        list(APPEND _strix_pkg_generators NSIS)
    elseif(STRIX_WIX_CANDLE)
        list(APPEND _strix_pkg_generators WIX)
    endif()
    set(CPACK_PACKAGE_FILE_NAME "strix-${_strix_pkg_flavor}-${PROJECT_VERSION}-windows-${_strix_pkg_arch}")
    set(CPACK_PACKAGE_INSTALL_DIRECTORY "Strix")
    set(CPACK_ARCHIVE_COMPONENT_INSTALL OFF)
    set(CPACK_NSIS_DISPLAY_NAME "Strix ${PROJECT_VERSION} (${_strix_pkg_flavor})")
    set(CPACK_NSIS_MODIFY_PATH ON)
    set(CPACK_WIX_UPGRADE_GUID "6F3C2A9E-4B1D-4E7A-9C55-2D8B71A0E413")
else()
    set(_strix_pkg_generators TGZ)
    find_program(STRIX_DPKG_DEB dpkg-deb)
    find_program(STRIX_RPMBUILD rpmbuild)
    if(STRIX_DPKG_DEB)
        list(APPEND _strix_pkg_generators DEB)
    endif()
    if(STRIX_RPMBUILD)
        list(APPEND _strix_pkg_generators RPM)
    endif()
    # Stage through DESTDIR so /etc/strix stays absolute in every generator (tar.gz mirrors the root layout).
    set(CPACK_SET_DESTDIR ON)
    set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
    set(CPACK_PROJECT_CONFIG_FILE "${_strix_pkg_dir}/CPackProjectConfig.cmake")
    set(CPACK_STRIP_FILES ON)
    set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
    foreach(_c IN ITEMS core tools docs hip cuda)
        string(TOUPPER "${_c}" _C)
        set(CPACK_ARCHIVE_${_C}_FILE_NAME "strix-${_c}-${PROJECT_VERSION}-${STRIX_PACKAGE_RELEASE}-linux-${_strix_pkg_arch}")
    endforeach()

    # DEB
    set(CPACK_DEB_COMPONENT_INSTALL ON)
    set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)
    set(CPACK_DEBIAN_PACKAGE_RELEASE "${STRIX_PACKAGE_RELEASE}")
    set(CPACK_DEBIAN_PACKAGE_SECTION "science")
    set(CPACK_DEBIAN_PACKAGE_CONTROL_STRICT_PERMISSION TRUE)
    find_program(STRIX_DPKG_SHLIBDEPS dpkg-shlibdeps)
    # Keep the toolchain's versioned ABI requirements, including the actual HIP SONAME provider.
    # The tested HIP range supplements the scan; it must never replace it.
    set(STRIX_PACKAGE_DEB_HIP_DEPENDS "libamdhip64-7 (>= 7.1.0) | hip-runtime-amd (>= 7.1.0)" CACHE STRING
        "DEB Depends of strix-hip on the ROCm runtime (tested range)")
    set(_strix_deb_cxx_explicit "libc6, libstdc++6, libgcc-s1, ")
    if(STRIX_DPKG_SHLIBDEPS AND EXISTS "/etc/debian_version")
        set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
        set(_strix_deb_cxx "")
    else()
        # Not a Debian build host (no shlibs database): declare the C++ runtime explicitly.
        set(_strix_deb_cxx "${_strix_deb_cxx_explicit}")
    endif()
    set(CPACK_DEBIAN_CORE_PACKAGE_DEPENDS "${_strix_deb_cxx}")
    string(REGEX REPLACE ", $" "" CPACK_DEBIAN_CORE_PACKAGE_DEPENDS "${CPACK_DEBIAN_CORE_PACKAGE_DEPENDS}")
    set(CPACK_DEBIAN_TOOLS_PACKAGE_DEPENDS "${_strix_deb_cxx}python3, python3-numpy")
    if(STRIX_PACKAGE_BACKEND STREQUAL "hip" AND STRIX_DPKG_DEB AND
       (NOT STRIX_DPKG_SHLIBDEPS OR NOT EXISTS "/etc/debian_version"))
        message(FATAL_ERROR "HIP DEBs require dpkg-shlibdeps and a Debian shlibs database")
    endif()
    set(CPACK_DEBIAN_HIP_PACKAGE_SHLIBDEPS ON)
    set(CPACK_DEBIAN_HIP_PACKAGE_DEPENDS "${STRIX_PACKAGE_DEB_HIP_DEPENDS}")
    set(CPACK_DEBIAN_CUDA_PACKAGE_DEPENDS "${CPACK_DEBIAN_CORE_PACKAGE_DEPENDS}")
    set(CPACK_DEBIAN_HIP_PACKAGE_SUGGESTS "strix-core, strix-tools, strix-docs")
    set(CPACK_DEBIAN_DOCS_PACKAGE_ARCHITECTURE all)
    set(CPACK_DEBIAN_DOCS_PACKAGE_SHLIBDEPS OFF)

    # RPM: library requirements come from rpmbuild's automatic soname scan (libamdhip64.so.N on any ROCm distro).
    set(CPACK_RPM_COMPONENT_INSTALL ON)
    set(CPACK_RPM_FILE_NAME RPM-DEFAULT)
    set(CPACK_RPM_PACKAGE_RELEASE "${STRIX_PACKAGE_RELEASE}")
    set(CPACK_RPM_PACKAGE_LICENSE "Apache-2.0")
    set(CPACK_RPM_PACKAGE_GROUP "Applications/Engineering")
    set(CPACK_RPM_PACKAGE_RELOCATABLE OFF)
    set(CPACK_RPM_TOOLS_PACKAGE_REQUIRES "python3, python3-numpy")
    set(CPACK_RPM_DOCS_PACKAGE_ARCHITECTURE noarch)
    set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION /usr/lib/systemd /usr/lib/systemd/system /usr/lib/sysusers.d
        /usr/share/doc)
    set(CPACK_RPM_HIP_USER_FILELIST "%config(noreplace) ${CMAKE_INSTALL_FULL_SYSCONFDIR}/strix/serve.toml"
        "%config(noreplace) ${CMAKE_INSTALL_FULL_SYSCONFDIR}/strix/strix.env")

    # Maintainer scripts (one POSIX script per hook for DEB and RPM): service user and daemon-reload only; the
    # service is never enabled or started by the package.
    file(MAKE_DIRECTORY "${_strix_pkg_out}/deb-hip")
    foreach(_hook postinst prerm postrm)
        file(COPY_FILE "${_strix_pkg_dir}/linux/${_hook}.sh" "${_strix_pkg_out}/deb-hip/${_hook}")
        file(CHMOD "${_strix_pkg_out}/deb-hip/${_hook}" PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
            GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
    endforeach()
    file(WRITE "${_strix_pkg_out}/deb-hip/conffiles"
        "${CMAKE_INSTALL_FULL_SYSCONFDIR}/strix/serve.toml\n${CMAKE_INSTALL_FULL_SYSCONFDIR}/strix/strix.env\n")
    set(CPACK_DEBIAN_HIP_PACKAGE_CONTROL_EXTRA "${_strix_pkg_out}/deb-hip/postinst;${_strix_pkg_out}/deb-hip/prerm;"
        "${_strix_pkg_out}/deb-hip/postrm;${_strix_pkg_out}/deb-hip/conffiles")
    string(REPLACE ";;" ";" CPACK_DEBIAN_HIP_PACKAGE_CONTROL_EXTRA "${CPACK_DEBIAN_HIP_PACKAGE_CONTROL_EXTRA}")
    set(CPACK_RPM_HIP_POST_INSTALL_SCRIPT_FILE "${_strix_pkg_dir}/linux/postinst.sh")
    set(CPACK_RPM_HIP_PRE_UNINSTALL_SCRIPT_FILE "${_strix_pkg_dir}/linux/prerm.sh")
    set(CPACK_RPM_HIP_POST_UNINSTALL_SCRIPT_FILE "${_strix_pkg_dir}/linux/postrm.sh")
endif()
set(CPACK_GENERATOR "${_strix_pkg_generators}" CACHE STRING "CPack generators of this tree")
set(CPACK_SOURCE_GENERATOR "")
include(CPack)
