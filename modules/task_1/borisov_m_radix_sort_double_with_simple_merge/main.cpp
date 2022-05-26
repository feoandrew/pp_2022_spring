// Copyright 2022 Borisov Maxim

#include "../../../modules/task_1/borisov_m_radix_sort_double_with_simple_merge/radix_sort.h"
#include "gtest/gtest.h"

TEST(borisov_task1, test_size_10) {
  std::vector<double> vec1 = random_vec(10);
  std::vector<double> vec2 = random_vec(10);
  std::vector<double> result;
  result = radix_sort(vec1, vec2);
  ASSERT_TRUE(std::is_sorted(result.begin(), result.end()));
}

TEST(borisov_task1, test_size_100) {
  std::vector<double> vec1 = random_vec(100);
  std::vector<double> vec2 = random_vec(100);
  std::vector<double> result;
  result = radix_sort(vec1, vec2);
  ASSERT_TRUE(std::is_sorted(result.begin(), result.end()));
}

TEST(borisov_task1, test_size_500) {
  std::vector<double> vec1 = random_vec(500);
  std::vector<double> vec2 = random_vec(500);
  std::vector<double> result;
  result = radix_sort(vec1, vec2);
  ASSERT_TRUE(std::is_sorted(result.begin(), result.end()));
}

TEST(borisov_task1, test_size_1000) {
  std::vector<double> vec1 = random_vec(1000);
  std::vector<double> vec2 = random_vec(1000);
  std::vector<double> result;
  result = radix_sort(vec1, vec2);
  ASSERT_TRUE(std::is_sorted(result.begin(), result.end()));
}

TEST(borisov_task1, test_size_2000) {
  std::vector<double> vec1 = random_vec(2000);
  std::vector<double> vec2 = random_vec(2000);
  std::vector<double> result;
  result = radix_sort(vec1, vec2);
  ASSERT_TRUE(std::is_sorted(result.begin(), result.end()));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
