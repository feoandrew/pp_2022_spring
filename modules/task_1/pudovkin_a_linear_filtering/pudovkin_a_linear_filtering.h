// Copyright 2022 Pudovkin Artem
#ifndef MODULES_TASK_1_PUDOVKIN_A_LINEAR_FILTERING_PUDOVKIN_A_LINEAR_FILTERING_H_
#define MODULES_TASK_1_PUDOVKIN_A_LINEAR_FILTERING_PUDOVKIN_A_LINEAR_FILTERING_H_

#include <vector>

std::vector<std::vector<double>> getRandomMatrix(
    const std::vector<std::vector<double>>::size_type rows,
    const std::vector<std::vector<double>>::size_type cols);
std::vector<std::vector<double>> getSequentialFilter(
    const std::vector<std::vector<double>>& matrixOfColor);

#endif  // MODULES_TASK_1_PUDOVKIN_A_LINEAR_FILTERING_PUDOVKIN_A_LINEAR_FILTERING_H_
