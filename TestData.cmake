# Copies test data from SOURCE_DIR to BINARY_DIR.
function(add_test_data_files target_name)
  # ARGN will contain all arguments passed after 'target_name'
  # These are expected to be the relative paths of the data files.
  set(copied_data_files "") # List to store all destination paths (outputs)
  foreach(relative_path ${ARGN})
      set(source_path "${CMAKE_CURRENT_SOURCE_DIR}/${relative_path}")
      set(dest_path "${CMAKE_CURRENT_BINARY_DIR}/${relative_path}")
      # Ensure the source file actually exists
      if(NOT EXISTS "${source_path}")
          message(FATAL_ERROR "Test data source file not found: ${source_path} for target ${target_name}")
      endif()
      add_custom_command(
          TARGET "${target_name}"
          BYPRODUCTS "${dest_path}"
          COMMAND ${CMAKE_COMMAND} -E copy_if_different
                  "${source_path}"
                  "${dest_path}"
          DEPENDS "${source_path}" # Depends on the original source file
          COMMENT "Copying test data ${relative_path} for target ${target_name}"
      )
  endforeach()
endfunction()
