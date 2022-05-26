// Copyright 2022 Shurygina A

#ifndef MODULES_TASK_2_SHURYGINA_A_LABELING_OMP_LABELING_H_
#define MODULES_TASK_2_SHURYGINA_A_LABELING_OMP_LABELING_H_

#include <omp.h>
#include <vector>
#include <utility>

std::vector<int> getRandomImg(int rows, int cols);
std::vector<int> secondMark(std::vector<int>, int, int, std::vector<int>);

std::pair<std::vector<int>, std::vector<int> >
firstMarkOmp(std::vector<int>, int, int);
std::vector<int>
CLabelingOmp(const std::vector<int>& array, int rows, int cols);
std::vector<int>
transformResult(const std::vector<int>& array, int rows, int cols);

#endif  // MODULES_TASK_2_SHURYGINA_A_LABELING_OMP_LABELING_H_
