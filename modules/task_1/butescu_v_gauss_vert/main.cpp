  // Copyright 2022 Butescu Vladimir
#include <gtest/gtest.h>
#include <vector>
#include "../../../modules/task_1/butescu_v_gauss_vert/vert_gauss.h"

TEST(Gaussian_Filter_, Zero_Rows_Cols) {
    int rows = 0;
    int cols = 0;
    std::vector<double> matrix(rows * cols);

    ASSERT_ANY_THROW(matrix = gauss_filter_sequence(matrix, rows, cols));
}

TEST(Gaussian_Filter_, Zero_Matrix) {
    int rows = 3;
    int cols = 3;
    std::vector<double> matrix(0);

    ASSERT_ANY_THROW(matrix = gauss_filter_sequence(matrix, rows, cols));
}

TEST(Gaussian_Filter_, Test_Matrix_Size_3x3) {
    int rows = 3;
    int cols = 3;
    std::vector<double> matrix(rows * cols);
    matrix[0] = 85;
    matrix[1] = 99;
    matrix[2] = 112;
    matrix[3] = 126;
    matrix[4] = 140;
    matrix[5] = 153;
    matrix[6] = 167;
    matrix[7] = 180;
    matrix[8] = 195;

    std::vector<double> answer(rows * cols);

    answer[0] = 96;
    answer[1] = 107;
    answer[2] = 117;
    answer[3] = 128;
    answer[4] = 139;
    answer[5] = 150;
    answer[6] = 161;
    answer[7] = 171;
    answer[8] = 183;

    matrix = gauss_filter_sequence(matrix, rows, cols);

    ASSERT_EQ(matrix, answer);
}

TEST(Gaussian_Filter_, Test_Random_Matrix_Size_3x3) {
    int rows = 3;
    int cols = 3;
    std::vector<double> matrix = createRandomMatrix(rows, cols);

    ASSERT_NO_THROW(matrix = gauss_filter_sequence(matrix, rows, cols));
}

TEST(Gaussian_Filter_, Test_Random_Matrix_Size_10x10) {
    int rows = 10;
    int cols = 10;
    std::vector<double> matrix = createRandomMatrix(rows, cols);

    ASSERT_NO_THROW(matrix = gauss_filter_sequence(matrix, rows, cols));
}

TEST(Gaussian_Filter_, Test_Random_Matrix_Size_100x100) {
    int rows = 100;
    int cols = 100;
    std::vector<double> matrix = createRandomMatrix(rows, cols);

    ASSERT_NO_THROW(matrix = gauss_filter_sequence(matrix, rows, cols));
}


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
