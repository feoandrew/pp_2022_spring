// Copyright 2022 Medvedeva Karina
#ifndef MODULES_TASK_2_MEDVEDEVA_K_LINEAR_HISTOGRAM_STRETCHING_LINEAR_HISTOGRAM_STRETCHING_H_
#define MODULES_TASK_2_MEDVEDEVA_K_LINEAR_HISTOGRAM_STRETCHING_LINEAR_HISTOGRAM_STRETCHING_H_

#include <omp.h>
#include <vector>

std::vector<int> getRandomMatrix(std::vector<int>::size_type row_count,
    std::vector<int>::size_type column_count);
std::vector<int> getSequentialOperations(const std::vector<int>& matrix,
    std::vector<int>::size_type row_count,
    std::vector<int>::size_type column_count);
std::vector<int> getParallelOperations(const std::vector<int>& matrix,
    std::vector<int>::size_type row_count,
    std::vector<int>::size_type column_count);

#endif  // MODULES_TASK_2_MEDVEDEVA_K_LINEAR_HISTOGRAM_STRETCHING_LINEAR_HISTOGRAM_STRETCHING_H_
