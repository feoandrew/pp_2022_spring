// Copyright 2022 Pudovkin Artem
#ifndef MODULES_TASK_2_PUDOVKIN_A_LINEAR_FILTERING_PUDOVKIN_A_LINEAR_FILTERING_H_
#define MODULES_TASK_2_PUDOVKIN_A_LINEAR_FILTERING_PUDOVKIN_A_LINEAR_FILTERING_H_

#include <vector>

std::vector<std::vector<double>> getRandomMatrix(
    const std::vector<std::vector<double>>::size_type rows,
    const std::vector<std::vector<double>>::size_type cols);
std::vector<std::vector<double>> getSequentialFilter(
    const std::vector<std::vector<double>>& commonMatrixOfColor);
std::vector<std::vector<double>> getParallelFilter(
    const std::vector<std::vector<double>>& commonMatrixOfColor);

#endif  // MODULES_TASK_2_PUDOVKIN_A_LINEAR_FILTERING_PUDOVKIN_A_LINEAR_FILTERING_H_
