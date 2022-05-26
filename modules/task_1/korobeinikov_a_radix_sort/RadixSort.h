// Copyright 2022 Korobeinikov Alexandr
#ifndef MODULES_TASK_1_KOROBEINIKOV_A_RADIX_SORT_RADIXSORT_H_
#define MODULES_TASK_1_KOROBEINIKOV_A_RADIX_SORT_RADIXSORT_H_

#include<vector>

std::vector<double> CountingSort(std::vector<double> vec, size_t num_byte);
std::vector<double> RadixSort(std::vector<double> vec);
std::vector<double> getRandomVector(int size, double lower_bound,
                                        double upper_bound, int seed = -1);

#endif  // MODULES_TASK_1_KOROBEINIKOV_A_RADIX_SORT_RADIXSORT_H_
