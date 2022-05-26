// Copyright 2022 Utyugov Denis
#ifndef MODULES_TASK_1_UTYUGOV_D_CANNONS_ALG_CANNONS_ALG_H_
#define MODULES_TASK_1_UTYUGOV_D_CANNONS_ALG_CANNONS_ALG_H_
#include <vector>
std::vector<double> getRandomVector(int size);
std::vector<std::vector<double>> getRndMatrix(int size);
std::vector<std::vector<double>> Multiplicate(
    std::vector<std::vector<double>> A, std::vector<std::vector<double>> B);
std::vector<std::vector<double>> BlockMultiplicate(
    std::vector<std::vector<double>> A, std::vector<std::vector<double>> B,
    int BlSize);

#endif  // MODULES_TASK_1_UTYUGOV_D_CANNONS_ALG_CANNONS_ALG_H_
