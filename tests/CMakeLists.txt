add_executable(run_tests test_clusterizer.cpp)


target_link_libraries(run_tests PRIVATE ccl)
target_include_directories(run_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data/event.dat
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)

add_test(NAME CCLTests COMMAND run_tests)
set_tests_properties(CCLTests PROPERTIES  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
