#----------------------------------------------------------------------------------
#  Nz-Videomni - AviUtl2 general-purpose (.aux2) plugin
#  Track A: native foundation (1.0.0 - WebView2 host + JSON-RPC bridge contract v6)
#----------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.20)

# NOTE: CMake's project(VERSION ...) only accepts numeric major.minor.patch
# components and cannot express a pre-release suffix like "-rc1". The
# human-facing, full version string (e.g. "1.0.0") is authored in exactly
# two places: native/src/bridge_core.h's kPluginVersion (reported by the
# "ping" method) and scripts/package.ps1's -Version default (used for the zip
# name and package.ini). Keep VERSION below in sync with the numeric prefix of
# those two, but do not try to append a suffix here.
project(NzVideomni
    VERSION 1.0.0
    DESCRIPTION "LTX 2.3 video generation frontend for AviUtl2"
    LANGUAGES CXX)

# This project targets 64-bit Windows / MSVC only.
if(NOT WIN32)
    message(FATAL_ERROR "Nz-Videomni can only be built on Windows (Win32 API + MSVC).")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Emit compile_commands.json for tooling.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Select the MSVC runtime explicitly. The vendored WebView2LoaderStatic.lib is
# built against the dynamic release CRT (/MD), so link every configuration
# against MultiThreadedDLL to avoid a CRT mismatch. (Requires CMP0091 = NEW,
# which is the default for this cmake_minimum_required version.)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")

# Vendored AviUtl2 SDK (headers only). Comments inside are Shift-JIS, so this
# directory is added as a SYSTEM include and NO /utf-8 flag is used: the
# compiler parses those headers under the default (system ANSI) code page,
# which matches their Shift-JIS encoding on this Japanese Windows host. Our own
# sources are strictly ASCII, so they are safe under any code page.
set(AVIUTL2_SDK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aviutl2_sdk")
set(NATIVE_DIR      "${CMAKE_CURRENT_SOURCE_DIR}/native")
set(THIRD_PARTY_DIR "${NATIVE_DIR}/third_party")

#----------------------------------------------------------------------------------
#  Single-file Web UI embedding (release packaging).
#
#  When NZVIDEOMNI_EMBED_WEBUI=ON the fully-inlined single-page Web UI
#  (webui/dist-single/index.html, produced by `npm run build:single`) is compiled
#  into the DLL as a named RCDATA resource ("NZVIDEOMNI_WEBUI_INDEX"). webview_host.cpp
#  serves it from memory at runtime, so the plugin needs no on-disk webui\ folder.
#  With the option OFF (the default) the DLL is unchanged and the UI is served
#  from the webui\ folder next to it (development mode).
#
#  NZVIDEOMNI_WEBUI_SINGLE_HTML overrides the embedded HTML path (used by CI / local
#  verification to embed an alternate single-file build).
#----------------------------------------------------------------------------------
option(NZVIDEOMNI_EMBED_WEBUI
    "Embed webui/dist-single/index.html into the DLL as an RCDATA resource" OFF)
set(NZVIDEOMNI_WEBUI_SINGLE_HTML
    "${CMAKE_CURRENT_SOURCE_DIR}/webui/dist-single/index.html"
    CACHE FILEPATH "Single-file Web UI HTML embedded when NZVIDEOMNI_EMBED_WEBUI=ON")

set(WEBVIEW2_INCLUDE_DIR "${THIRD_PARTY_DIR}/webview2/include")
set(WEBVIEW2_LIB         "${THIRD_PARTY_DIR}/webview2/x64/WebView2LoaderStatic.lib")
set(WIL_INCLUDE_DIR      "${THIRD_PARTY_DIR}/wil/include")
set(NLOHMANN_INCLUDE_DIR "${THIRD_PARTY_DIR}/nlohmann")
set(DOCTEST_INCLUDE_DIR  "${THIRD_PARTY_DIR}/doctest")

#----------------------------------------------------------------------------------
#  Shared code common to plugin + tests
#  (strconv, logging, the WebView2-independent RPC core, and the WinHTTP client -
#   all unit- or integration-testable without WebView2 or the SDK's UI types)
#----------------------------------------------------------------------------------
set(NZVIDEOMNI_COMMON_SOURCES
    "${NATIVE_DIR}/src/strconv.cpp"
    "${NATIVE_DIR}/src/log.cpp"
    "${NATIVE_DIR}/src/bridge_core.cpp"
    "${NATIVE_DIR}/src/http_client.cpp"
    "${NATIVE_DIR}/src/wic_png.cpp"
    "${NATIVE_DIR}/src/settings.cpp"
    "${NATIVE_DIR}/src/timeline_math.cpp"
    "${NATIVE_DIR}/src/timeline_select.cpp"
    "${NATIVE_DIR}/src/alias_util.cpp"
    "${NATIVE_DIR}/src/track_postprocess.cpp"
    "${NATIVE_DIR}/src/provisional.cpp"
    "${NATIVE_DIR}/src/mf_mp4_writer.cpp"
    "${NATIVE_DIR}/src/media_fps_probe.cpp"
    "${NATIVE_DIR}/src/wav_probe.cpp"
    "${NATIVE_DIR}/src/fs_util.cpp")

#----------------------------------------------------------------------------------
#  Plugin target -> NzVideomni.aux2
#----------------------------------------------------------------------------------
add_library(NzVideomni SHARED
    "${NATIVE_DIR}/src/plugin.cpp"
    "${NATIVE_DIR}/src/bridge.cpp"
    "${NATIVE_DIR}/src/webview_host.cpp"
    ${NZVIDEOMNI_COMMON_SOURCES})

target_include_directories(NzVideomni PRIVATE "${NATIVE_DIR}/src")
target_include_directories(NzVideomni SYSTEM PRIVATE
    "${AVIUTL2_SDK_DIR}"
    "${WEBVIEW2_INCLUDE_DIR}"
    "${WIL_INCLUDE_DIR}"
    "${NLOHMANN_INCLUDE_DIR}")

target_compile_definitions(NzVideomni PRIVATE
    UNICODE _UNICODE WIN32_LEAN_AND_MEAN NOMINMAX)

# WebView2 static loader plus the Win32 libraries it and our host code need.
target_link_libraries(NzVideomni PRIVATE
    "${WEBVIEW2_LIB}"
    comctl32
    comdlg32
    ole32
    oleaut32
    advapi32
    shell32
    shlwapi
    version
    winhttp
    windowscodecs
    user32
    # Media Foundation: IMFSinkWriter mp4 (H.264 + AAC) encoder (mf_mp4_writer)
    # and the IMFSourceReader frame-rate probe (media_fps_probe).
    mfplat
    mfreadwrite
    mfuuid)

# Output name: NzVideomni.aux2 (no "lib" prefix).
set_target_properties(NzVideomni PROPERTIES
    PREFIX ""
    OUTPUT_NAME "NzVideomni"
    SUFFIX ".aux2")

if(MSVC)
    # NOTE: deliberately NO /utf-8 here. The compiler falls back to the system
    # ANSI code page, which matches the Shift-JIS SDK headers. Our sources are
    # ASCII-only, so they compile identically under any code page.
    target_compile_options(NzVideomni PRIVATE /W4 /permissive-)
endif()

#----------------------------------------------------------------------------------
#  Optional: embed the single-file Web UI as an RCDATA resource.
#----------------------------------------------------------------------------------
if(NZVIDEOMNI_EMBED_WEBUI)
    if(NOT EXISTS "${NZVIDEOMNI_WEBUI_SINGLE_HTML}")
        message(FATAL_ERROR
            "NZVIDEOMNI_EMBED_WEBUI=ON but the single-file Web UI was not found:\n"
            "  ${NZVIDEOMNI_WEBUI_SINGLE_HTML}\n"
            "Build it first (in webui/: `npm run build:single`, or run "
            "scripts/build.ps1 -EmbedWebui), or point NZVIDEOMNI_WEBUI_SINGLE_HTML at "
            "an existing single-file HTML.")
    endif()
    enable_language(RC)
    # Bake the absolute path into the generated .rc as a doubled-backslash string
    # literal so rc.exe reads the file at build time.
    file(TO_NATIVE_PATH "${NZVIDEOMNI_WEBUI_SINGLE_HTML}" _nzvideomni_webui_native)
    string(REPLACE "\\" "\\\\" NZVIDEOMNI_WEBUI_SINGLE_HTML_RC "${_nzvideomni_webui_native}")
    set(_nzvideomni_webui_rc "${CMAKE_CURRENT_BINARY_DIR}/webui_embedded.rc")
    configure_file("${NATIVE_DIR}/res/webui.rc.in" "${_nzvideomni_webui_rc}" @ONLY)
    # Re-configure if the embedded HTML changes so the resource is rebuilt.
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${NZVIDEOMNI_WEBUI_SINGLE_HTML}")
    target_sources(NzVideomni PRIVATE "${_nzvideomni_webui_rc}")
    # Make the .rc's compiled .res depend on the embedded HTML directly, so ninja
    # recompiles the resource (and relinks) whenever only the Web UI bundle changes.
    set_source_files_properties("${_nzvideomni_webui_rc}" PROPERTIES OBJECT_DEPENDS "${NZVIDEOMNI_WEBUI_SINGLE_HTML}")
    target_compile_definitions(NzVideomni PRIVATE NZVIDEOMNI_EMBED_WEBUI=1)
    message(STATUS "NzVideomni: embedding Web UI resource from ${NZVIDEOMNI_WEBUI_SINGLE_HTML}")
endif()

#----------------------------------------------------------------------------------
#  Test target -> NzVideomni_tests.exe (doctest, registered with CTest)
#----------------------------------------------------------------------------------
enable_testing()

add_executable(NzVideomni_tests
    "${NATIVE_DIR}/tests/test_main.cpp"
    "${NATIVE_DIR}/tests/test_strconv.cpp"
    "${NATIVE_DIR}/tests/test_bridge_core.cpp"
    "${NATIVE_DIR}/tests/test_http_client.cpp"
    "${NATIVE_DIR}/tests/test_wic_png.cpp"
    "${NATIVE_DIR}/tests/test_settings.cpp"
    "${NATIVE_DIR}/tests/test_timeline_math.cpp"
    "${NATIVE_DIR}/tests/test_timeline_select.cpp"
    "${NATIVE_DIR}/tests/test_alias_util.cpp"
    "${NATIVE_DIR}/tests/test_track_postprocess.cpp"
    "${NATIVE_DIR}/tests/test_provisional.cpp"
    "${NATIVE_DIR}/tests/test_mf_mp4_writer.cpp"
    "${NATIVE_DIR}/tests/test_media_fps_probe.cpp"
    "${NATIVE_DIR}/tests/test_wav_probe.cpp"
    "${NATIVE_DIR}/tests/test_fs_util.cpp"
    ${NZVIDEOMNI_COMMON_SOURCES})

target_include_directories(NzVideomni_tests PRIVATE "${NATIVE_DIR}/src")
target_include_directories(NzVideomni_tests SYSTEM PRIVATE
    "${AVIUTL2_SDK_DIR}"
    "${DOCTEST_INCLUDE_DIR}"
    "${NLOHMANN_INCLUDE_DIR}")

target_compile_definitions(NzVideomni_tests PRIVATE
    UNICODE _UNICODE WIN32_LEAN_AND_MEAN NOMINMAX)

# The WinHTTP client needs winhttp.lib; the WIC PNG encoder needs windowscodecs
# and COM (ole32); the Media Foundation mp4 writer + its read-back test need the
# MF platform / sink-writer / GUID import libraries.
target_link_libraries(NzVideomni_tests PRIVATE
    winhttp windowscodecs ole32 mfplat mfreadwrite mfuuid)

if(MSVC)
    target_compile_options(NzVideomni_tests PRIVATE /W4 /permissive-)
endif()

add_test(NAME NzVideomni_tests COMMAND NzVideomni_tests)
