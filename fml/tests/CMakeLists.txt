project(fml_tests CXX)

set(FML_TESTS_SOURCES
    main.cpp
    gradient_descent_test.cpp
    )

add_executable(${PROJECT_NAME} ${FML_TESTS_SOURCES})

# Copy test data into place.
add_custom_command(TARGET ${PROJECT_NAME}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/data/
    ${CMAKE_BINARY_DIR}/fml/tests/data/
    )

target_link_libraries(fml_tests
    fml
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES})