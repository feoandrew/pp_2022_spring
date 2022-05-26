// Copyright 2022 Kovalev Ruslan
#ifndef MODULES_TASK_2_KOVALEV_R_ODD_EVEN_SORT_OMP_KOVALEV_R_ODD_EVEN_SORT_OMP_H_
#define MODULES_TASK_2_KOVALEV_R_ODD_EVEN_SORT_OMP_KOVALEV_R_ODD_EVEN_SORT_OMP_H_


#include <string>
#include <vector>

bool check(std::vector<int>* arr_1, std::vector<int>* arr_2, int sz);
void vec_gen(std::vector<int>* vec, int len);
void copy_vectors(std::vector<int>* arr_1, std::vector<int>* arr_2, int sz);
void getSequantialSort(std::vector<int>* arr, int sz);
std::vector<int> getParallelSort(const std::vector<int>& commonVector);

#endif  // MODULES_TASK_2_KOVALEV_R_ODD_EVEN_SORT_OMP_KOVALEV_R_ODD_EVEN_SORT_OMP_H_
