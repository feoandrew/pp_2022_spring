// Copyright 2022 Medvedeva Karina
#include <gtest/gtest.h>
#include "./linear_histogram_stretching.h"

TEST(Generation_Matrix, can_generate_square_matrix) {
    ASSERT_NO_THROW(getRandomMatrix(15, 15));
}

TEST(Generation_Matrix, can_generate_arbitrary_matrix) {
    ASSERT_NO_THROW(getRandomMatrix(11, 15));
}

TEST(Sequential_Operations, getSequentialOperations_can_work_with_square_matrix) {
    std::vector<int> matrix = getRandomMatrix(15, 15);

    ASSERT_NO_THROW(getSequentialOperations(matrix, 15, 15));
}

TEST(Sequential_Operations, getSequentialOperations_can_work_with_arbitrary_matrix) {
    std::vector<int> matrix = getRandomMatrix(10, 15);

    ASSERT_NO_THROW(getSequentialOperations(matrix, 10, 15));
}

TEST(Sequential_Operations, getSequentialOperations_can_work_with_big_matrix) {
    std::vector<int> matrix = getRandomMatrix(30, 30);

    ASSERT_NO_THROW(getSequentialOperations(matrix, 30, 30));
}

TEST(Sequential_Operations, getSequentialOperations_works_correctly_with_square_matrix) {
    std::vector<int> matrix = {1, 3, 2, 5, 15, 7, 1, 0, 9};
    std::vector<int> res = {17, 51, 34, 85, 255, 119, 17, 0, 153};
    ASSERT_EQ(res, getSequentialOperations(matrix, 3, 3));
}

TEST(Sequential_Operations, getSequentialOperations_works_correctly_with_arbitrary_matrix) {
    std::vector<int> matrix = {2, 17, 10, 4, 7, 3};
    std::vector<int> res = { 0, 255, 136, 34, 85, 17};

    ASSERT_EQ(res, getSequentialOperations(matrix, 2, 3));
}
