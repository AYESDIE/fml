include_directories(..)


add_executable(test
        main.cpp
        fml.hpp)

target_link_libraries(test fml)