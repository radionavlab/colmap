# Because this target is header-only, we cannot use the COLMAP_ADD_LIBRARY macro
add_library(open_mvs INTERFACE)
target_sources(open_mvs INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/Interface.h)
target_include_directories(open_mvs INTERFACE .)