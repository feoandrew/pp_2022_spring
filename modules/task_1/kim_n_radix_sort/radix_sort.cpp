// Copyright 2022 Kim Nikita
#include <vector>
#include <random>
#include "../../../modules/task_1/kim_n_radix_sort/radix_sort.h"


std::vector<int> getRandomVector(int size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> res(size);
  int sign;
  for (int i = 0; i < size; i++) {
    sign = 1 + gen() % 2 * -2;
    res[i] = gen() % 100 * sign;
  }
  return res;
}

int getMax(std::vector<int> input_vec, int size) {
  int max = abs(input_vec[0]);
  for (int i = 1; i < size; i++)
    if (abs(input_vec[i]) > max)
      max = abs(input_vec[i]);
  return max;
}

std::vector<int> countSort(const std::vector<int>& input_vec, int size, int place) {
  int dig, sign;
  std::vector<int> ans(size);

  // digits of 19 since range is -9 to 9
  std::vector<int> digits(19);
  for (int i = 0; i < size; i++) {
    // add digits to the correct index
    // -9 + 9 -> 0
    // 9 + 9 -> 18
    sign = input_vec[i] < 0 ? -1 : 1;
    dig = (((input_vec[i] * sign) / place) % 10) * sign + 9;
    digits[dig]++;
  }

  for (int i = 1; i < 19; i++)
    digits[i] += digits[i - 1];

  for (int i = size - 1; i >= 0; i--) {
    sign = input_vec[i] < 0 ? -1 : 1;
    dig = (((input_vec[i] * sign) / place) % 10) * sign + 9;
    ans[digits[dig] - 1] = input_vec[i];
    digits[dig]--;
  }
  return ans;
}

std::vector<int> radixSort(const std::vector<int>& input_vec, int size) {
  int max_value = getMax(input_vec, size);
  std::vector<int> res(input_vec);

  for (int place = 1; max_value / place > 0; place *= 10)
    res = countSort(res, size, place);
  return res;
}
