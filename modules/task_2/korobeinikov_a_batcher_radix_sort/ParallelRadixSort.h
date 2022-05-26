// Copyright 2022 Korobeinikov Alexandr
#ifndef MODULES_TASK_2_KOROBEINIKOV_A_BATCHER_RADIX_SORT_PARALLELRADIXSORT_H_
#define MODULES_TASK_2_KOROBEINIKOV_A_BATCHER_RADIX_SORT_PARALLELRADIXSORT_H_

#include<vector>

std::vector<double> CountingSort(std::vector<double> vec, size_t num_byte);
void RadixSort(int left, int right, std::vector<double>* vec);
std::vector<double> getRandomVector(int size, double lower_bound,
    double upper_bound, int seed = -1);
void EvenSplitter(int start, size_t size1, size_t size2,
    std::vector <double>* arr);
void OddSplitter(int start, size_t size1, size_t size2,
    std::vector <double>* arr);
void RadixSortParallel(std::vector <double>* arr, int th = 8);

#endif  // MODULES_TASK_2_KOROBEINIKOV_A_BATCHER_RADIX_SORT_PARALLELRADIXSORT_H_
