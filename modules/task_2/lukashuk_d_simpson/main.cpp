// Copyright 2022 Lukashuk Diana
#include <gtest/gtest.h>

#include <iostream>

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
  return std::exp(x[0] / x[1]) * std::sin(x[2]) + std::cos(x[3]) / x[4] -
         std::exp(x[5]);
}

bool test(const std::vector<int>& steps, const FUNC_T& f) {
  double error = 1e-4;
  BORDER_T temp_b{{1, 18}, {-7, 2}, {1, 3}, {7, 11}, {0.1, 0.9}, {2, 3}};
  BORDER_T border(temp_b.begin(), temp_b.begin() + steps.size());

  volatile double start_seq = omp_get_wtime();
  double seq = simpson(f, border, steps);
  volatile double end_seq = omp_get_wtime();

  volatile double start_par = omp_get_wtime();
  double par = simpson_par(f, border, steps);
  volatile double end_par = omp_get_wtime();

  double seq_time = end_seq - start_seq;
  double par_time = end_par - start_par;
  std::cout << "Effective: " << seq_time / par_time << '\n';

  return (std::abs(seq - par) < error);
}

TEST(Simpson, Dimension_2) {
  ASSERT_TRUE(test(std::vector<int>(2, 100), func_1));
}

TEST(Simpson, Dimension_3) {
  ASSERT_TRUE(test(std::vector<int>(3, 12), func_2));
}

TEST(Simpson, Dimension_4) {
  ASSERT_TRUE(test(std::vector<int>(4, 4), func_3));
}

TEST(Simpson, Dimension_5) {
  ASSERT_TRUE(test(std::vector<int>(5, 2), func_4));
}

TEST(Simpson, Dimension_6) {
  ASSERT_TRUE(test(std::vector<int>(6, 2), func_5));
}
