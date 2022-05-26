// Copyright 2022 Denisova Julia
#ifndef MODULES_TASK_3_DENISOVA_RADIX_SORT_TBB_RADIX_SORT_TBB_H_
#define MODULES_TASK_3_DENISOVA_RADIX_SORT_TBB_RADIX_SORT_TBB_H_

#include <algorithm>
#include <deque>
#include <iostream>
#include <random>
#include <vector>

void radixSort(int* arr, int size);
std::vector<int> getRandomVector(int size = -1, int maxEl = -1);
void printVec(std::vector<int> vec);
void parallelSort(int* arr, int size, int threads);

#endif  // MODULES_TASK_3_DENISOVA_RADIX_SORT_TBB_RADIX_SORT_TBB_H_
