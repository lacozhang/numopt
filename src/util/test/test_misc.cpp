#include "util/dynamic_array_impl.h"
#include "gtest/gtest.h"

TEST(DArray, SetIntersection) {
  mltools::DArray<int> a{1, 2, 3, 4}, b{4, 5, 6}, c{4};
  EXPECT_EQ(a.setIntersection(b), c);
}
