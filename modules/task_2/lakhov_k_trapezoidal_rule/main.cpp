// Copyright 2022 Lakhov Kirill
#include <gtest/gtest.h>
#include <math.h>

#include <vector>
#include "./trapezoidal_rule.h"

#define USE_EFFICIENCY_TESTS 0

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

TEST(lakhov_trapezoidal_rule_omp, func1) {
  std::vector<std::pair<double, double>> intervals(1);
  intervals[0] = {0, 10};
  int interval_count = 10;
  double result_seq = trapezoidalRuleSeq(f1, intervals, interval_count);
  double result_p = trapezoidalRuleParallel(f1, intervals, interval_count);
  ASSERT_NEAR(result_seq, result_p, 0.1);
}


TEST(lakhov_trapezoidal_rule_omp, func2) {
  std::vector<std::pair<double, double>> intervals(2);
  intervals[0] = {0, 2};
  intervals[1] = {0, 2};
  int interval_count = 70;
  double result_seq = trapezoidalRuleSeq(f2, intervals, interval_count);
  double result_p = trapezoidalRuleParallel(f2, intervals, interval_count);
  ASSERT_NEAR(result_seq, result_p, 0.1);
}

TEST(lakhov_trapezoidal_rule_omp, func3) {
  std::vector<std::pair<double, double>> intervals(3);
  intervals[0] = {0, 3};
  intervals[1] = {0, 3};
  intervals[2] = {0, 3};
  int interval_count = 100;
  double result_seq = trapezoidalRuleSeq(f3, intervals, interval_count);
  double result_p = trapezoidalRuleParallel(f3, intervals, interval_count);
  ASSERT_NEAR(result_seq, result_p, 0.1);
}

TEST(lakhov_trapezoidal_rule_omp, func4) {
  std::vector<std::pair<double, double>> intervals(3);
  intervals[0] = {0, 1};
  intervals[1] = {0, 1};
  intervals[2] = {0, 1};
  int interval_count = 100;
  double result_seq = trapezoidalRuleSeq(f4, intervals, interval_count);
  double result_p = trapezoidalRuleParallel(f4, intervals, interval_count);
  ASSERT_NEAR(result_seq, result_p, 0.1);
}

TEST(lakhov_trapezoidal_rule_omp, func5) {
  std::vector<std::pair<double, double>> intervals(2);
  intervals[0] = {0, 2};
  intervals[1] = {0, 2};
  int interval_count = 100;
  double result_seq = trapezoidalRuleSeq(f5, intervals, interval_count);
  double result_p = trapezoidalRuleParallel(f5, intervals, interval_count);
  ASSERT_NEAR(result_seq, result_p, 0.1);
}

#if USE_EFFICIENCY_TESTS == 1

  TEST(lakhov_trapezoidal_rule_omp, ef_test1) {
    std::vector<std::pair<double, double>> intervals(2);
    intervals[0] = {0, 40};
    intervals[1] = {0, 40};
    int interval_count = 1000;

    double st1 = omp_get_wtime();
    double result_seq = trapezoidalRuleSeq(f5, intervals, interval_count);
    double st2 = omp_get_wtime();

    double pt1 = omp_get_wtime();
    double result_p = trapezoidalRuleParallel(f5, intervals, interval_count);
    double pt2 = omp_get_wtime();

    std::cout << "Sequential time: " << st2 - st1 << std::endl;
    std::cout << "Parallel time: " << pt2 - pt1 << std::endl;
    std::cout << "Ratio seq/par: " << (st2 - st1) / (pt2 - pt1) << std::endl;
    ASSERT_NEAR(result_seq, result_p, 0.1);
  }

  TEST(lakhov_trapezoidal_rule_omp, ef_test2) {
    std::vector<std::pair<double, double>> intervals(3);
    intervals[0] = {0, 20};
    intervals[1] = {0, 20};
    intervals[2] = {0, 20};
    int interval_count = 100;

    double st1 = omp_get_wtime();
    double result_seq = trapezoidalRuleSeq(f4, intervals, interval_count);
    double st2 = omp_get_wtime();

    double pt1 = omp_get_wtime();
    double result_p = trapezoidalRuleParallel(f4, intervals, interval_count);
    double pt2 = omp_get_wtime();

    std::cout << "Sequential time: " << st2 - st1 << std::endl;
    std::cout << "Parallel time: " << pt2 - pt1 << std::endl;
    std::cout << "Ratio seq/par: " << (st2 - st1) / (pt2 - pt1) << std::endl;
    ASSERT_NEAR(result_seq, result_p, 0.1);
  }

#endif  // USE_EFFICIENCY_TESTS

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
