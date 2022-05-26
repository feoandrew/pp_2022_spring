// Copyright 2022 Lakhov Kirill
#include <gtest/gtest.h>
#include <math.h>

#include <vector>
#include <iostream>
#include "./trapezoidal_rule.h"

// f(x) = x
double f1(std::vector<double> args) {
  return args[0];
}

// f(x, y) = x + y
double f2(std::vector<double> args) {
  return args[0] + args[1];
}

// f(x, y, z) = x + y + z
double f3(std::vector<double> args) {
  return args[0] + args[1] + args[2];
}

// f(x, y, z) = e^(-x-y-z)
double f4(std::vector<double> args) {
  return exp(-args[0]-args[1]-args[2]);
}

// f(x, y) = log10(x^2) + 3y
double f5(std::vector<double> args) {
  return log10(args[0]*args[0]) + 3 * args[1];
}

TEST(lakhov_trapezoidal_rule, func1) {
  std::vector<std::pair<double, double>> intervals(1);
  intervals[0] = {0, 10};
  int interval_count = 5;
  double result = trapezoidalRule(f1, intervals, interval_count);
  ASSERT_NEAR(50, result, 1);
}


TEST(lakhov_trapezoidal_rule, func2) {
  std::vector<std::pair<double, double>> intervals(2);
  intervals[0] = {0, 2};
  intervals[1] = {0, 2};
  int interval_count = 70;
  double result = trapezoidalRule(f2, intervals, interval_count);
  ASSERT_NEAR(8.0, result, 0.2);
}

TEST(lakhov_trapezoidal_rule, func3) {
  std::vector<std::pair<double, double>> intervals(3);
  intervals[0] = {0, 3};
  intervals[1] = {0, 3};
  intervals[2] = {0, 3};
  int interval_count = 100;
  double result = trapezoidalRule(f3, intervals, interval_count);
  ASSERT_NEAR(121.5, result, 1);
}

TEST(lakhov_trapezoidal_rule, func4) {
  std::vector<std::pair<double, double>> intervals(3);
  intervals[0] = {0, 1};
  intervals[1] = {0, 1};
  intervals[2] = {0, 1};
  int interval_count = 100;
  double result = trapezoidalRule(f4, intervals, interval_count);
  ASSERT_NEAR(0.2525, result, 0.01);
}

TEST(lakhov_trapezoidal_rule, func5) {
  std::vector<std::pair<double, double>> intervals(2);
  intervals[0] = {0, 2};
  intervals[1] = {0, 2};
  int interval_count = 100;
  double result = trapezoidalRule(f5, intervals, interval_count);
  ASSERT_NEAR(10.9, result, 0.2);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
