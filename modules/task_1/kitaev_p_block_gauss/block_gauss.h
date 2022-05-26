// Copyright 2022 Kitaev Pavel

#ifndef MODULES_TASK_1_KITAEV_P_BLOCK_GAUSS_BLOCK_GAUSS_H_
#define MODULES_TASK_1_KITAEV_P_BLOCK_GAUSS_BLOCK_GAUSS_H_

#include <vector>
#include <random>
#include <ctime>

std::vector<int> getRandomMatrix(int row, int col);
std::vector<int> getGaussKernel(double sigma);
std::vector<int> SequentialGauss(std::vector<int> img,
  int row, int col, double sigma);

#endif  // MODULES_TASK_1_KITAEV_P_BLOCK_GAUSS_BLOCK_GAUSS_H_
