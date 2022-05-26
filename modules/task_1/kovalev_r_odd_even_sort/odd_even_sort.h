// Copyright 2022 Kovalev Ruslan

#ifndef MODULES_TASK_1_KOVALEV_R_ODD_EVEN_SORT_ODD_EVEN_SORT_H_
#define MODULES_TASK_1_KOVALEV_R_ODD_EVEN_SORT_ODD_EVEN_SORT_H_
#include <vector>

bool check(std::vector<int>* arr_1, std::vector<int>* arr_2, int sz);

void vec_gen(std::vector<int>* vec, int len);

void copy_vectors(std::vector<int>* arr_1, std::vector<int>* arr_2, int sz);

void getSequantialSort(std::vector<int>* arr, int sz);

std::vector<int> Merge(std::vector<int>* arr_1, std::vector<int>* arr_2, int sz_1, int sz_2);

std::vector<int> OddEvenMerge(std::vector<int> first, std::vector<int> second,
                              int sz_1, int sz_2);

#endif  // MODULES_TASK_1_KOVALEV_R_ODD_EVEN_SORT_ODD_EVEN_SORT_H_
