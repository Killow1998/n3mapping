if(NOT DEFINED N3MAPPING_CHECK_LIBRARY)
  message(FATAL_ERROR "N3MAPPING_CHECK_LIBRARY is required")
endif()

if(NOT DEFINED N3MAPPING_EXPECTED_FILENAME)
  message(FATAL_ERROR "N3MAPPING_EXPECTED_FILENAME is required")
endif()

if(NOT EXISTS "${N3MAPPING_CHECK_LIBRARY}")
  message(FATAL_ERROR "Expected LIO core library does not exist: ${N3MAPPING_CHECK_LIBRARY}")
endif()

get_filename_component(actual_filename "${N3MAPPING_CHECK_LIBRARY}" NAME)
if(NOT actual_filename STREQUAL N3MAPPING_EXPECTED_FILENAME)
  message(FATAL_ERROR
    "Unexpected LIO core library filename: expected ${N3MAPPING_EXPECTED_FILENAME}, got ${actual_filename}")
endif()

message(STATUS "LIO core library filename matches: ${actual_filename}")
