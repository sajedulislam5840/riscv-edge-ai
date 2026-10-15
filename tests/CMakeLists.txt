add_executable(run_tests test_main.cpp)
add_test(NAME CoreEngineTests COMMAND run_tests)
