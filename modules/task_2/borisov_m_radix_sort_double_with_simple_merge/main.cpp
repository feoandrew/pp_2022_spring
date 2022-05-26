// Copyright 2022 Borisov Maxim

#include "../../../modules/task_2/borisov_m_radix_sort_double_with_simple_merge/radix_sort.h"
#include "gtest/gtest.h"

constexpr std::size_t n_threads = 5;

TEST(borisov_task1, test_size_10) {
  std::vector<double> seq = random_vec(10);
  std::vector<double> par = random_vec(10);
  seq = radix_sort_seq(seq);
  par = radix_sort_omp(par, n_threads);
  ASSERT_EQ(seq, par);
}

TEST(borisov_task1, test_size_100) {
  std::vector<double> seq = random_vec(100);
  std::vector<double> par = random_vec(100);
  seq = radix_sort_seq(seq);
  par = radix_sort_omp(par, n_threads);
  ASSERT_EQ(seq, par);
}

TEST(borisov_task1, test_size_500) {
  std::vector<double> seq = random_vec(500);
  std::vector<double> par = random_vec(500);
  seq = radix_sort_seq(seq);
  par = radix_sort_omp(par, n_threads);
  ASSERT_EQ(seq, par);
}

TEST(borisov_task1, test_size_1000) {
  std::vector<double> seq = random_vec(1000);
  std::vector<double> par = random_vec(1000);
  seq = radix_sort_seq(seq);
  par = radix_sort_omp(par, n_threads);
  ASSERT_EQ(seq, par);
}

TEST(borisov_task1, test_size_2000) {
  std::vector<double> seq = random_vec(2000);
  std::vector<double> par = random_vec(2000);
  seq = radix_sort_seq(seq);
  par = radix_sort_omp(par, n_threads);
  ASSERT_EQ(seq, par);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
