execute_process(
  COMMAND "${N3MAPPING_GIT_EXECUTABLE}" rev-parse HEAD
  WORKING_DIRECTORY "${N3MAPPING_PRODUCT_SOURCE_DIR}"
  RESULT_VARIABLE _head_result
  OUTPUT_VARIABLE _head
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT _head_result EQUAL 0 OR
   NOT _head STREQUAL N3MAPPING_EXPECTED_PRODUCT_COMMIT)
  message(FATAL_ERROR
    "Product V1 build aborted: source HEAD changed after configure")
endif()

execute_process(
  COMMAND "${N3MAPPING_GIT_EXECUTABLE}"
    status --porcelain --untracked-files=all
  WORKING_DIRECTORY "${N3MAPPING_PRODUCT_SOURCE_DIR}"
  RESULT_VARIABLE _status_result
  OUTPUT_VARIABLE _status
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT _status_result EQUAL 0 OR _status)
  message(FATAL_ERROR
    "Product V1 build aborted: source tree is not clean")
endif()
