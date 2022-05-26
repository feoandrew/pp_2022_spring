// Copyright 2022 George Amberson
#ifndef MODULES_TASK_2_YASHINA_D_LINEAR_BLOCK_FILTRATION_OMP_YASHINA_D_LINEAR_BLOCK_FILTRATION_OMP_H_
#define MODULES_TASK_2_YASHINA_D_LINEAR_BLOCK_FILTRATION_OMP_YASHINA_D_LINEAR_BLOCK_FILTRATION_OMP_H_
#include <vector>
#include <list>
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
void getParallelOperations(std::vector<std::vector<double>>* image,
    double* matrix, int weight, int height);
#endif  // MODULES_TASK_2_YASHINA_D_LINEAR_BLOCK_FILTRATION_OMP_YASHINA_D_LINEAR_BLOCK_FILTRATION_OMP_H_
