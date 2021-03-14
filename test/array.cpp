#include <gtest/gtest.h>
#include <array.h>

TEST(ArrayTest, ArgMax) {
	auto grid = Grid1d(10000);
	auto arr1 = Array1D<double>(grid);
	arr1.set(1.0);
	arr1(5000) = 2.0;

	EXPECT_EQ(arr1.argmax([](double x) { return x; }), 5000);
}
