// Copyright 2022 Lukashuk Diana
#include <gtest/gtest.h>

#include "./simpson.h"

double func_1(const std::vector<double>& x) { return x[0] * x[1]; }

double func_2(const std::vector<double>& x) { return x[0] * x[1] * x[2]; }

double func_3(const std::vector<double>& x) {
  return x[0] * x[1] * x[2] * x[3];
}

double func_4(const std::vector<double>& x) {
  return x[0] * x[1] * x[2] * x[3] * x[4];
}

double func_5(const std::vector<double>& x) {
  return x[0] * x[1] * x[2] * x[3] * x[4] * x[5];
}

bool test(const std::vector<int>& steps, const FUNC_T& f, double currect) {
  double error = 1e-4;
  BORDER_T temp_b{{1, 18}, {-7, 2}, {1, 3}, {7, 11}, {0.1, 0.9}, {2, 3}};
  BORDER_T border(temp_b.begin(), temp_b.begin() + steps.size());
  return (std::abs(simpson(f, border, steps) - currect) < error);
}

TEST(Simpson, Dimension_2) {
  ASSERT_TRUE(test(std::vector<int>(2, 100), func_1, -3633.75));
}

TEST(Simpson, Dimension_3) {
  ASSERT_TRUE(test(std::vector<int>(3, 12), func_2, -14535));
}

TEST(Simpson, Dimension_4) {
  ASSERT_TRUE(test(std::vector<int>(4, 4), func_3, -523260));
}

TEST(Simpson, Dimension_5) {
  ASSERT_TRUE(test(std::vector<int>(5, 2), func_4, -209304));
}

TEST(Simpson, Dimension_6) {
  ASSERT_TRUE(test(std::vector<int>(6, 1), func_5, -523260));
}
