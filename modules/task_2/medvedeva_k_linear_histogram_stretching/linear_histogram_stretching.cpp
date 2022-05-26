// Copyright 2022 Medvedeva Karina
#include "../../../modules/task_2/medvedeva_k_linear_histogram_stretching/linear_histogram_stretching.h"
#include <random>

std::vector<int> getRandomMatrix(std::vector<int>::size_type row_count, std::vector<int>::size_type column_count) {
    std::random_device dev;
    std::mt19937 gen(dev());
    std::vector<int> vec(row_count * column_count);
    for (std::vector<int>::size_type i = 0; i < row_count; i++) {
        for (std::vector<int>::size_type j = 0; j < column_count; j++) {
            vec[i * column_count + j] = gen() % 100;
        }
    }
    return vec;
}

std::vector<int> getSequentialOperations(const std::vector<int>& matrix,
    std::vector<int>::size_type row_count,
    std::vector<int>::size_type column_count) {
    std::vector<int> res(matrix);
    int max_y = 0;
    int min_y = 255;
    for (std::size_t i = 0; i < column_count * row_count; i++) {
        if (matrix[i] > max_y) {
            max_y = matrix[i];
        }
        if (matrix[i] < min_y) {
            min_y = matrix[i];
        }
    }
    for (std::size_t i = 0; i < column_count * row_count; i++) {
        res[i] = (matrix[i] - min_y) * (255 / (max_y - min_y));
    }
    return res;
}

std::vector<int> getParallelOperations(const std::vector<int>& matrix,
    std::vector<int>::size_type row_count,
    std::vector<int>::size_type column_count) {
    std::vector<int> res(matrix);
    int max_y = 0;
    int min_y = 255;
    int local_max_y = 0;
    int local_min_y = 255;
#pragma omp parallel shared(matrix, row_count, column_count, res) firstprivate(local_max_y, local_min_y)
    {
#pragma omp for
        for (int i = 0; i < static_cast<int>(column_count * row_count); i++) {
            if (matrix[i] > local_max_y) {
                local_max_y = matrix[i];
            }
            if (matrix[i] < local_min_y) {
                local_min_y = matrix[i];
            }
        }
#pragma omp critical
        {
            if (max_y < local_max_y) {
                max_y = local_max_y;
            }
            if (min_y > local_min_y) {
                min_y = local_min_y;
            }
        }
#pragma omp barrier
#pragma omp for
        for (int i = 0; i < static_cast<int>(column_count * row_count); i++) {
            res[i] = (matrix[i] - min_y) * (255 / (max_y - min_y));
        }
    }
    return res;
}
