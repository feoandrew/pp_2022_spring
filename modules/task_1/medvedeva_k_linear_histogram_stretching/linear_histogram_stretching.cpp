// Copyright 2022 Medvedeva Karina
#include <random>
#include "../../../modules/task_1/medvedeva_k_linear_histogram_stretching/linear_histogram_stretching.h"

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
