// Copyright 2022 Belyaev Ilya
#ifndef MODULES_TASK_1_BELYAEV_I_HOAR_SORT_SIMPLE_FUSION_HOAR_SORT_SIMPLE_FUSION_H_
#define MODULES_TASK_1_BELYAEV_I_HOAR_SORT_SIMPLE_FUSION_HOAR_SORT_SIMPLE_FUSION_H_

#include <vector>

std::vector<int> random_gen(int size);
void hoar_sort(int left, int right, std::vector <int>* arr);
bool checker(std::vector <int>* vec);
std::vector<int> merge(const std::vector<int>& a, const std::vector<int>& b);

#endif  // MODULES_TASK_1_BELYAEV_I_HOAR_SORT_SIMPLE_FUSION_HOAR_SORT_SIMPLE_FUSION_H_
