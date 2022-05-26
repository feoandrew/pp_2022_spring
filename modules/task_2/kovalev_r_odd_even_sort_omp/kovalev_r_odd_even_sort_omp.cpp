// Copyright 2022 Kovalev Ruslan
#include <omp.h>

#include <iostream>
#include <utility>
#include <random>
#include <string>
#include <vector>

#include "../../../modules/task_2/kovalev_r_odd_even_sort_omp/kovalev_r_odd_even_sort_omp.h"

void copy_vectors(std::vector<int>* arr_1, std::vector<int>* arr_2, int sz) {
  for (int i = 0; i < sz; i++) {
    arr_2->at(i) = arr_1->at(i);
  }
}

void vec_gen(std::vector<int>* vec, int len) {
  std::random_device dev;
  std::mt19937 gen(dev());
  for (int i = 0; i < len; i++) {
    vec->at(i) = gen() % 10000;
  }
}

bool check(std::vector<int>* arr_1, std::vector<int>* arr_2, int sz) {
  for (int i = 0; i < sz; i++) {
    if (arr_1->at(i) != arr_2->at(i)) {
      std::cout << i << " " << std::endl;
      return false;
    }
  }
  return true;
}

std::vector<int> Merge(std::vector<int>* arr_1, std::vector<int>* arr_2,
                       int sz_1, int sz_2) {
  int i = 0;
  int j = 0;
  int k = 0;
  std::vector<int> res(sz_1 + sz_2);
  while (i < sz_1 && j < sz_2) {
    if (arr_1->at(i) < arr_2->at(j))
      res[k++] = arr_1->at(i++);
    else
      res[k++] = arr_2->at(j++);
  }
  while (i < sz_1) res[k++] = arr_1->at(i++);
  while (j < sz_2) res[k++] = arr_2->at(j++);
  return res;
}

int getMax(std::vector<int>* arr, int sz) {
  int max = arr->at(0);
  for (int i = 1; i < sz; i++)
    if (arr->at(i) > max) max = arr->at(i);
  return max;
}

void countingSort(std::vector<int>* array, int size, int place) {
  const int max = 10;
  std::vector<int> output(size, 0);
  std::vector<int> count(max, 0);
  for (int i = 0; i < size; i++) count[(array->at(i) / place) % 10]++;
  for (int i = 1; i < max; i++) count[i] += count[i - 1];
  for (int i = size - 1; i >= 0; i--) {
    output[count[(array->at(i) / place) % 10] - 1] = array->at(i);
    count[(array->at(i) / place) % 10]--;
  }
  for (int i = 0; i < size; i++) array->at(i) = output[i];
}

void getSequantialSort(std::vector<int>* arr, int sz) {
  int max = getMax(arr, sz);
  for (int place = 1; max / place > 0; place *= 10)
    countingSort(arr, sz, place);
}

void EvenSplitter(std::vector<int>* first, std::vector<int>* second) {
  std::vector<int> even_tmp_buf_1;
  std::vector<int> even_tmp_buf_2;
  for (size_t i = 0; i < first->size(); i += 2)
    even_tmp_buf_1.push_back((*first)[i]);
  for (size_t i = 0; i < second->size(); i += 2)
    even_tmp_buf_2.push_back((*second)[i]);
  *first = even_tmp_buf_1;
  *second = even_tmp_buf_2;
}

void OddSplitter(std::vector<int>* first, std::vector<int>* second) {
  std::vector<int> odd_tmp_buf_1;
  std::vector<int> odd_tmp_buf_2;
  for (size_t i = 1; i < first->size(); i += 2)
    odd_tmp_buf_1.push_back((*first)[i]);
  for (size_t i = 1; i < second->size(); i += 2)
    odd_tmp_buf_2.push_back((*second)[i]);
  *first = odd_tmp_buf_1;
  *second = odd_tmp_buf_2;
}

std::vector<int> GetResult(std::vector<int> first, std::vector<int> second) {
  std::vector<int> res;
  for (size_t i = 0; i < first.size() || i < second.size(); i++) {
    if (i < first.size()) res.push_back(first[i]);
    if (i < second.size()) res.push_back(second[i]);
  }
  for (size_t i = 1; i < first.size() + second.size() - 1; i += 2) {
    if (res[i] > res[i + 1]) std::swap(res[i], res[i + 1]);
  }
  return res;
}

std::vector<int> OddEvenMerge(std::vector<int> first, std::vector<int> second,
                              int sz_1, int sz_2) {
  std::vector<int> first_odd_tmp = first;
  std::vector<int> second_odd_tmp = second;
  std::vector<int> first_even_tmp = first;
  std::vector<int> second_even_tmp = second;
  EvenSplitter(&first_even_tmp, &second_even_tmp);
  OddSplitter(&first_odd_tmp, &second_odd_tmp);
  int sz1 = first_odd_tmp.size();
  int sz2 = second_odd_tmp.size();
  int sz3 = first_even_tmp.size();
  int sz4 = second_even_tmp.size();
  first = Merge(&first_even_tmp, &second_even_tmp, sz3, sz4);
  second = Merge(&first_odd_tmp, &second_odd_tmp, sz1, sz2);
  return GetResult(first, second);
}

std::vector<int> getParallelSort(const std::vector<int>& commonVector) {
  int numberOfThread = omp_get_num_procs();
  int dataPortion = commonVector.size() / numberOfThread;
  std::vector<std::vector<int>> vecOfVec(numberOfThread);
#pragma omp parallel num_threads(numberOfThread)
  {
    int currentThread = omp_get_thread_num();
    std::vector<int> local;
    if (currentThread != numberOfThread - 1) {
      local = {commonVector.begin() + currentThread * dataPortion,
               commonVector.begin() + (currentThread + 1) * dataPortion};
    } else {
      local = {commonVector.begin() + currentThread * dataPortion,
               commonVector.end()};
    }
    int len = local.size();
    getSequantialSort(&local, len);
    vecOfVec[currentThread] = local;
  }
  std::vector<int> resultVector = vecOfVec[0];
  for (int i = 1; i < numberOfThread; ++i) {
    int len_1 = resultVector.size();
    int len_2 = vecOfVec[i].size();
    resultVector = OddEvenMerge(resultVector, vecOfVec[i], len_1, len_2);
  }
  return resultVector;
}
