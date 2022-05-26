// Copyright 2022 Denisova Julia
#ifndef MODULES_TASK_2_DENISOVA_RADIX_SORT_OMP_RADIX_SORT_OMP_H_
#define MODULES_TASK_2_DENISOVA_RADIX_SORT_OMP_RADIX_SORT_OMP_H_
#include <omp.h>

#include <algorithm>
#include <deque>
#include <iostream>
#include <random>
#include <vector>

void radixSort(int* arr, int size);
std::vector<int> getRandomVector(int size = -1, int maxEl = -1);
void printVec(std::vector<int> vec);
void parallelSort(int* arr, int size);

#endif  // MODULES_TASK_2_DENISOVA_RADIX_SORT_OMP_RADIX_SORT_OMP_H_
