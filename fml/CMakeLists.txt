include_directories(..)


add_executable(test
        main.cpp
        fml.hpp)

set(input_csv ${CMAKE_CURRENT_SOURCE_DIR}/fml/data/in.csv)
set(output_csv ${CMAKE_CURRENT_BINARY_DIR}/fml/data/out.csv)

add_custom_command(
        OUTPUT  ${output_csv}
        COMMAND version1 ${input_csv} ${output_csv}
        DEPENDS version1 ${input_csv}
)
