// Copyright 2022 Kovalev Ruslan

#include <gtest/gtest.h>
#include <vector>
#include "../kovalev_r_odd_even_sort/odd_even_sort.h"

TEST(Sequential_algorithm, radix_sort_1) {
  int size = 10;
  std::vector<int> arr_1(size);
  std::vector<int> arr_2(size);
  vec_gen(&arr_1, size);
  copy_vectors(&arr_1, &arr_2, size);
  bool res = check(&arr_1, &arr_2, size);
  ASSERT_EQ(true, res);
}

TEST(Sequential_algorithm, radix_sort_2) {
  int size = 100;
  std::vector<int> arr_1(size);
  std::vector<int> arr_2(size);
  vec_gen(&arr_1, size);
  copy_vectors(&arr_1, &arr_2, size);
  sort(arr_1.begin(), arr_1.end());
  getSequantialSort(&arr_2, size);
  bool res = check(&arr_1, &arr_2, size);
  ASSERT_EQ(true, res);
}

TEST(Sequential_algorithm, radix_sort_3) {
  int size = 25;
  std::vector<int> arr_1(25);
  std::vector<int> arr_2(25);
  std::vector<int> arr_3;
  std::vector<int> arr_4;
  vec_gen(&arr_1, size);
  vec_gen(&arr_2, size);
  for (int i = 0; i < size; i++) {
    arr_4.push_back(arr_1[i]);
    arr_4.push_back(arr_2[i]);
  }
  sort(arr_4.begin(), arr_4.end());
  sort(arr_1.begin(), arr_1.end());
  sort(arr_2.begin(), arr_2.end());
  arr_3 = Merge(&arr_1, &arr_2, size, size);
  bool res = check(&arr_3, &arr_4, size);
  ASSERT_EQ(true, res);
}

TEST(Sequential_algorithm, radix_sort_4) {
  int size = 25;
  std::vector<int> arr_1(25);
  std::vector<int> arr_2(25);
  std::vector<int> arr_3;
  std::vector<int> arr_4;
  vec_gen(&arr_1, size);
  vec_gen(&arr_2, size);
  for (int i = 0; i < size; i++) {
    arr_3.push_back(arr_1[i]);
    arr_3.push_back(arr_2[i]);
  }
  sort(arr_3.begin(), arr_3.end());
  sort(arr_1.begin(), arr_1.end());
  sort(arr_2.begin(), arr_2.end());
  arr_4 = OddEvenMerge(arr_1, arr_2, size, size);
  bool res = check(&arr_3, &arr_4, size);
  ASSERT_EQ(true, res);
}

TEST(Sequential_algorithm, radix_sort_5) {
  int size = 50;
  std::vector<int> arr_1(50);
  std::vector<int> arr_2(50);
  std::vector<int> arr_3;
  std::vector<int> arr_4;
  vec_gen(&arr_1, size);
  vec_gen(&arr_2, size);
  for (int i = 0; i < size; i++) {
    arr_3.push_back(arr_1[i]);
    arr_3.push_back(arr_2[i]);
  }
  sort(arr_3.begin(), arr_3.end());
  sort(arr_1.begin(), arr_1.end());
  sort(arr_2.begin(), arr_2.end());
  arr_4 = OddEvenMerge(arr_1, arr_2, size, size);
  bool res = check(&arr_3, &arr_4, size);
  ASSERT_EQ(true, res);
}






