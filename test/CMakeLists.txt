set(TEST_SOURCES
	"array.cpp"
)

add_executable(test ${TEST_SOURCES})
target_link_libraries(test tdse gtest_main ${MPI_C_LIBRARIES})
