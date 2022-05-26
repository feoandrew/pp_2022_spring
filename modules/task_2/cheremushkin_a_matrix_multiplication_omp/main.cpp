// Copyright 2022 Chremushkin Kirill
#include <gtest/gtest.h>
#include <vector>
#include <ctime>
#include "./ComplexMatrix.h"

TEST(MatrixComplex_Seq, get_works) {
    MatrixComplex matrix(3);
    EXPECT_NO_THROW(matrix.get(1, 1));
}

TEST(MatrixComplex_Seq, get_throws_exception) {
    MatrixComplex matrix(3);
    EXPECT_ANY_THROW(matrix.get(4, 4));
}

TEST(MatrixComplex_Seq, multiplication_works) {
    MatrixComplex matrix1(6);
    MatrixComplex matrix2(6);
    EXPECT_NO_THROW(matrix1.Multiply(matrix1, matrix2));
}

TEST(Matrix_Multiplication_Seq, multiplication_throws_exception) {
    MatrixComplex matrix1(3);
    MatrixComplex matrix2(5);
    MatrixComplex res;
    EXPECT_ANY_THROW(matrix1.Multiply(matrix1, matrix2));
}

TEST(Matrix_Multiplication_Seq, result_matrix_has_correct_size) {
    MatrixComplex matrix1(4);
    MatrixComplex matrix2(4);
    MatrixComplex res;
    res = matrix1.Multiply(matrix1, matrix2);
    EXPECT_EQ(res.getSize(), matrix1.getSize());
    EXPECT_EQ(res.getSize(), matrix2.getSize());
}
TEST(Matrix_Multiplication_OMP, parallel_multiplication_returns_same_result) {
    MatrixComplex matrix1(4);
    MatrixComplex matrix2(4);
    MatrixComplex result1;
    /*double start_time;
    double end_time;
    double search_time;*/
    // start_time = omp_get_wtime();
    result1 = matrix1.Multiply(matrix1, matrix2);
    // end_time = omp_get_wtime();
    MatrixComplex result2;
    // start_time = omp_get_wtime();
    result2 = matrix1.Multiply_parallel(matrix1, matrix2);
    // end_time = omp_get_wtime();
    // search_time = end_time - start_time;
    /*std::cout << "Line time: " << search_time * 4 << std::endl;
    std::cout << "Parallel time: " << search_time << std::endl;*/
    ASSERT_TRUE(result1 == result2);
}
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
