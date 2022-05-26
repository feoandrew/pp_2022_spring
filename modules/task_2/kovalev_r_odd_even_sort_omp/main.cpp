// Copyright 2022 Kovalev Ruslan
#include <gtest/gtest.h>

#include <ctime>
#include <vector>

#include "./kovalev_r_odd_even_sort_omp.h"

TEST(Sequential_algorithm, radix_sort_1) {
  int size = 100;
  std::vector<int> arr_1(size);
  std::vector<int> arr_2(size);
  std::vector<int> arr_3(size);
  vec_gen(&arr_1, size);
  copy_vectors(&arr_1, &arr_2, size);
  sort(arr_1.begin(), arr_1.end());
  arr_3 = getParallelSort(arr_2);
  bool res = check(&arr_1, &arr_3, size);
  ASSERT_EQ(true, res);
}

TEST(Sequential_algorithm, radix_sort_2) {
  int size = 1000;
  std::vector<int> arr_1(size);
  std::vector<int> arr_2(size);
  std::vector<int> arr_3(size);
  vec_gen(&arr_1, size);
  copy_vectors(&arr_1, &arr_2, size);
  sort(arr_1.begin(), arr_1.end());
  arr_3 = getParallelSort(arr_2);
  bool res = check(&arr_1, &arr_3, size);
  ASSERT_EQ(true, res);
}

TEST(Sequential_algorithm, radix_sort_3) {
  int size = 1001;
  std::vector<int> arr_1(size);
  std::vector<int> arr_2(size);
  std::vector<int> arr_3(size);
  vec_gen(&arr_1, size);
  copy_vectors(&arr_1, &arr_2, size);
  sort(arr_1.begin(), arr_1.end());
  arr_3 = getParallelSort(arr_2);
  bool res = check(&arr_1, &arr_3, size);
  ASSERT_EQ(true, res);
}

TEST(Sequential_algorithm, radix_sort_4) {
  int size = 333;
  std::vector<int> arr_1(size);
  std::vector<int> arr_2(size);
  std::vector<int> arr_3(size);
  vec_gen(&arr_1, size);
  copy_vectors(&arr_1, &arr_2, size);
  sort(arr_1.begin(), arr_1.end());
  arr_3 = getParallelSort(arr_2);
  bool res = check(&arr_1, &arr_3, size);
  ASSERT_EQ(true, res);
}

TEST(Sequential_algorithm, radix_sort_5) {
  int size = 624;
  std::vector<int> arr_1(size);
  std::vector<int> arr_2(size);
  std::vector<int> arr_3(size);
  vec_gen(&arr_1, size);
  copy_vectors(&arr_1, &arr_2, size);
  sort(arr_1.begin(), arr_1.end());
  arr_3 = getParallelSort(arr_2);
  bool res = check(&arr_1, &arr_3, size);
  ASSERT_EQ(true, res);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
