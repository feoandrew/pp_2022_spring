// Copyright 2022 George Amberson
#ifndef MODULES_TASK_1_YASHINA_D_LINEAR_BLOCK_FILTRATION_YASHINA_LINEAR_BLOCK_FILTRATION_H_
#define MODULES_TASK_1_YASHINA_D_LINEAR_BLOCK_FILTRATION_YASHINA_LINEAR_BLOCK_FILTRATION_H_
#include <gtest/gtest.h>
#include <list>
#include <vector>
#include <random>
#include <algorithm>
#include <iostream>
template <class T>
T clamp(T value, T max, T min) {
    if (value > max)
        return max;
    if (value < min)
        return min;
    return value;
}

double* create_random_kernel(int size_n, double sigma);
void getRandomImg(std::vector<std::vector<double>>* img,
int weight, int height);
void getSequentialOperations(std::vector<std::vector<double>>* image,
    double* matrix, int weight, int height);
#endif  // MODULES_TASK_1_YASHINA_D_LINEAR_BLOCK_FILTRATION_YASHINA_LINEAR_BLOCK_FILTRATION_H_
