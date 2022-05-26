// Copyright 2022 Kraev Nikita
#ifndef MODULES_TASK_2_KRAEV_N_RADIX_SORT_WITH_EVEN_ODD_MERGE_RADIX_SORT_WITH_EVEN_ODD_MERGE_H_
#define MODULES_TASK_2_KRAEV_N_RADIX_SORT_WITH_EVEN_ODD_MERGE_RADIX_SORT_WITH_EVEN_ODD_MERGE_H_
#include <omp.h>
#include <vector>
#include <random>

void radixSort(std::vector<double>* vec, size_t left, size_t right);
std::vector<double> getRandomVector(int size);
void evenOrOddSplitter(std::vector<double>* vec, size_t left_1, size_t right_1,
                       size_t left_2, size_t right_2, bool check);
void simpleComparator(std::vector<double>* vec, size_t left, size_t right);
int findNumberPhaseMerge(int size);
void parallelRadixSort(std::vector<double>* vec, int number_threads, int size);

#endif  // MODULES_TASK_2_KRAEV_N_RADIX_SORT_WITH_EVEN_ODD_MERGE_RADIX_SORT_WITH_EVEN_ODD_MERGE_H_
