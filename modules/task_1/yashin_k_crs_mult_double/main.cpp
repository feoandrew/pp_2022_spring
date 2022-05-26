// Copyright 2022 Yashin Kirill
#include <gtest/gtest.h>
#include <vector>
#include "../../modules/task_1/yashin_k_crs_mult_double/sparse_matrix_multiplication_crs.h"

TEST(Yashin_Kirill_Sparse_Matrix, Can_Create_Matrix) {
    Matrix matrix{{0.0, 1.1, 0.0, 0.0, 2.2},
                  {0.0, 0.0, 3.3, 4.4, 0.0},
                  {0.0, 5.5, 0.0, 0.0, 0.0},
                  {0.0, 0.0, 0.0, 6.6, 0.0},
                  {7.7, 0.0, 0.0, 0.0, 0.0}};
    int rows = 5;
    int columns = 5;
    std::vector<double> values = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7};
    std::vector<int> col_index = {1, 4, 2, 3, 1, 3, 0};
    std::vector<int> row_index = {0, 2, 4, 5, 6, 7};
    EXPECT_NO_THROW({sparse_matrix A(matrix);});
    EXPECT_NO_THROW({sparse_matrix B(rows, columns, values, col_index, row_index);});
    sparse_matrix A(matrix);
    sparse_matrix B(rows, columns, values, col_index, row_index);
    ASSERT_EQ(A, B);
}

TEST(Yashin_Kirill_Sparse_Matrix, Can_Compare_Sparse_Matrix) {
    int rows = 5;
    int cols = 5;
    std::vector<double> values = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7};
    std::vector<int> col_index = {1, 4, 2, 3, 1, 3, 0};
    std::vector<int> row_index = {0, 2, 4, 5, 6, 7};

    sparse_matrix A(rows, cols, values, col_index, row_index);
    sparse_matrix B(rows, cols, values, col_index, row_index);
    ASSERT_TRUE(A == B);
}

TEST(Yashin_Kirill_Sparse_Matrix, Can_Transform) {
    sparse_matrix _sparse_matrix(random_matrix(50, 50));
    Matrix matrix(_sparse_matrix.sparce_matrix_to_default());
    sparse_matrix result(matrix);
    ASSERT_EQ(_sparse_matrix, result);
}

TEST(Yashin_Kirill_Sparse_Matrix, Can_Miltiply) {
    Matrix A{{0.0, 1.1, 0.0, 0.0, 2.2},
             {0.0, 0.0, 3.3, 4.4, 0.0},
             {0.0, 5.5, 0.0, 0.0, 0.0},
             {0.0, 0.0, 0.0, 6.6, 0.0},
             {7.7, 0.0, 0.0, 0.0, 0.0}};

    Matrix true_result({{16.94, 0.0, 3.63, 4.84, 0.0},
                       {0.0, 18.15, 0.0, 29.04, 0.0},
                       {0.0, 0.0, 18.15, 24.2, 0.0},
                       {0.0, 0.0, 0.0, 43.56, 0.0},
                       {0.0, 8.47, 0.0, 0.0, 16.94}});

    Matrix result = matrix_multiplication(A, A);

    ASSERT_EQ(result.size(), true_result.size());
    ASSERT_EQ(result[0].size(), true_result[0].size());

    for (size_t i = 0; i < result.size(); i++) {
        for (size_t j = 0; j < result[0].size(); j++)
            ASSERT_DOUBLE_EQ(result[i][j], true_result[i][j]);
    }
}

TEST(Yashin_Kirill_Sparse_Matrix, Can_Multiply_Sparse) {
    int rows = 5;
    int columns = 5;
    std::vector<double> values = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7};
    std::vector<int> col_index = {1, 4, 2, 3, 1, 3, 0};
    std::vector<int> row_index = {0, 2, 4, 5, 6, 7};

    Matrix true_result({{16.94, 0.0, 3.63, 4.84, 0.0},
                       {0.0, 18.15, 0.0, 29.04, 0.0},
                       {0.0, 0.0, 18.15, 24.2, 0.0},
                       {0.0, 0.0, 0.0, 43.56, 0.0},
                       {0.0, 8.47, 0.0, 0.0, 16.94}});

    sparse_matrix A(rows, columns, values, col_index, row_index);
    sparse_matrix result = sparse_multiplication(A, A);
    sparse_matrix _true_result(true_result);

    ASSERT_EQ(_true_result, result);
}

TEST(Yashin_Kirill_Sparse_Matrix, Can_Multiply_Random) {
    Matrix A(random_matrix(10, 10));
    Matrix B(random_matrix(10, 10));

    sparse_matrix sA(A);
    sparse_matrix sB(B);

    sparse_matrix result_sparse = sparse_multiplication(sA, sB);
    Matrix result = matrix_multiplication(A, B);

    sparse_matrix _result(result);

    ASSERT_EQ(_result, result_sparse);
}
