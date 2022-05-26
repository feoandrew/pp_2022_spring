// Copyright 2022 Lakhov Kirill
#include <gtest/gtest.h>
#include <math.h>
#include <chrono> // NOLINT [build/c++11]
#include <algorithm>
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

TEST(lakhov_trapezoidal_rule_std, func1) {
  std::vector<std::pair<double, double>> intervals(1);
  intervals[0] = {0, 10};
  int interval_count = 10;
  double result_seq = trapezoidalRuleSeq(f1, intervals, interval_count);
  double result_p = trapezoidalRuleParallel(f1, intervals, interval_count);
  ASSERT_NEAR(result_seq, result_p, 0.1);
}


TEST(lakhov_trapezoidal_rule_std, func2) {
  std::vector<std::pair<double, double>> intervals(2);
  intervals[0] = {0, 2};
  intervals[1] = {0, 2};
  int interval_count = 70;
  double result_seq = trapezoidalRuleSeq(f2, intervals, interval_count);
  double result_p = trapezoidalRuleParallel(f2, intervals, interval_count);
  ASSERT_NEAR(result_seq, result_p, 0.1);
}

TEST(lakhov_trapezoidal_rule_std, func3) {
  std::vector<std::pair<double, double>> intervals(3);
  intervals[0] = {0, 3};
  intervals[1] = {0, 3};
  intervals[2] = {0, 3};
  int interval_count = 100;
  double result_seq = trapezoidalRuleSeq(f3, intervals, interval_count);
  double result_p = trapezoidalRuleParallel(f3, intervals, interval_count);
  ASSERT_NEAR(result_seq, result_p, 0.1);
}

TEST(lakhov_trapezoidal_rule_std, func4) {
  std::vector<std::pair<double, double>> intervals(3);
  intervals[0] = {0, 1};
  intervals[1] = {0, 1};
  intervals[2] = {0, 1};
  int interval_count = 100;
  double result_seq = trapezoidalRuleSeq(f4, intervals, interval_count);
  double result_p = trapezoidalRuleParallel(f4, intervals, interval_count);
  ASSERT_NEAR(result_seq, result_p, 0.1);
}

TEST(lakhov_trapezoidal_rule_std, func5) {
  std::vector<std::pair<double, double>> intervals(2);
  intervals[0] = {0, 2};
  intervals[1] = {0, 2};
  int interval_count = 100;
  double result_seq = trapezoidalRuleSeq(f5, intervals, interval_count);
  double result_p = trapezoidalRuleParallel(f5, intervals, interval_count);
  ASSERT_NEAR(result_seq, result_p, 0.1);
}

#if USE_EFFICIENCY_TESTS == 1

  TEST(lakhov_trapezoidal_rule_std, ef_test1) {
    std::vector<std::pair<double, double>> intervals(2);
    intervals[0] = {0, 40};
    intervals[1] = {0, 40};
    int interval_count = 1000;

    auto start_seq = std::chrono::high_resolution_clock::now();
    double result_seq = trapezoidalRuleSeq(f5, intervals, interval_count);
    auto end_seq = std::chrono::high_resolution_clock::now();

    auto start_par = std::chrono::high_resolution_clock::now();
    double result_p = trapezoidalRuleParallel(f5, intervals, interval_count);
    auto end_par = std::chrono::high_resolution_clock::now();

    auto seq_time = std::chrono::duration_cast
        <std::chrono::nanoseconds>(end_seq - start_seq);
    auto par_time = std::chrono::duration_cast
        <std::chrono::nanoseconds>(end_par - start_par);
    std::cout << "Sequential time: " << seq_time.count() << std::endl;
    std::cout << "Parallel time: " << par_time.count() << std::endl;
    std::cout << "Ratio seq/par: " << static_cast<double>(seq_time.count())
                                      / par_time.count() << std::endl;
    ASSERT_NEAR(result_seq, result_p, 0.1);
  }

  TEST(lakhov_trapezoidal_rule_std, ef_test2) {
    std::vector<std::pair<double, double>> intervals(3);
    intervals[0] = {0, 20};
    intervals[1] = {0, 20};
    intervals[2] = {0, 20};
    int interval_count = 100;


    auto start_seq = std::chrono::high_resolution_clock::now();
    double result_seq = trapezoidalRuleSeq(f4, intervals, interval_count);
    auto end_seq = std::chrono::high_resolution_clock::now();

    auto start_par = std::chrono::high_resolution_clock::now();
    double result_p = trapezoidalRuleParallel(f4, intervals, interval_count);
    auto end_par = std::chrono::high_resolution_clock::now();

    auto seq_time = std::chrono::duration_cast
        <std::chrono::nanoseconds>(end_seq - start_seq);
    auto par_time = std::chrono::duration_cast
        <std::chrono::nanoseconds>(end_par - start_par);
    std::cout << "Sequential time: " << seq_time.count() << std::endl;
    std::cout << "Parallel time: " << par_time.count() << std::endl;
    std::cout << "Ratio seq/par: " << static_cast<double>(seq_time.count())
                                      / par_time.count() << std::endl;
    ASSERT_NEAR(result_seq, result_p, 0.1);
  }

#endif  // USE_EFFICIENCY_TESTS

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
