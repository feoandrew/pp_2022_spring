// Copyright 2022 Kitaev Pavel

#ifndef MODULES_TASK_2_KITAEV_P_BLOCK_GAUSS_BLOCK_GAUSS_H_
#define MODULES_TASK_2_KITAEV_P_BLOCK_GAUSS_BLOCK_GAUSS_H_

#include <vector>
#include <random>
#include <ctime>

std::vector<int> getRandomMatrix(int row, int col);
std::vector<double> getGaussKernel(double sigma);
std::vector<int> SequentialGauss(std::vector<int> img,
  std::vector<double> gauss_kernel, int row, int col);
std::vector<int> ParallelGauss(std::vector<int> img,
  std::vector<double> gauss_kernel, int row, int col);

#endif  // MODULES_TASK_2_KITAEV_P_BLOCK_GAUSS_BLOCK_GAUSS_H_
