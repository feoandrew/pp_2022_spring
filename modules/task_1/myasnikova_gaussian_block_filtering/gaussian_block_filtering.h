// Copyright 2022 Myasnikova Varvara

#ifndef MODULES_TASK_1_MYASNIKOVA_GAUSSIAN_BLOCK_FILTERING_GAUSSIAN_BLOCK_FILTERING_H_
#define MODULES_TASK_1_MYASNIKOVA_GAUSSIAN_BLOCK_FILTERING_GAUSSIAN_BLOCK_FILTERING_H_

#include <ctime>
#include <vector>
#include <random>

std::vector<int> CreateMatrix(int row, int column);
std::vector<int> CreateKernel(double sigma);
std::vector<int> GaussFilterSeq(std::vector<int> matrix,
  std::vector<int> kernel, int row, int column);

#endif  // MODULES_TASK_1_MYASNIKOVA_GAUSSIAN_BLOCK_FILTERING_GAUSSIAN_BLOCK_FILTERING_H_
