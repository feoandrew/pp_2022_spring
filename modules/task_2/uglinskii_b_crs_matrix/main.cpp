// Copyright 2022 Uglinskii Bogdan
#include <gtest/gtest.h>
#include <omp.h>

#include <iostream>
#include <vector>

#include "./crs_multiplication.h"

TEST(Multiplication_seq, crs_5x5_5) {
  MatrixCRS matrix_A = GenerateRandomMatrixCRS(5, 5, 5);
  MatrixCRS matrix_B = GenerateRandomMatrixCRS(5, 5, 5);

  MatrixCRS matrix_C;
  CRSMultiply(matrix_A, matrix_B, &matrix_C);

  std::vector<std::vector<double>> C;

  NormalMulty(ExpandMatrix(matrix_A), ExpandMatrix(matrix_B), &C);
  MatrixCRS crs_C;
  ConvertToCRS(C, &crs_C);
  ASSERT_TRUE(CompareMatrixCRS(matrix_C, crs_C));
}

TEST(Multiplication_parallel, 3x3_4) {
  double t1, t2, t1_, t2_;

  MatrixCRS matrix_A = GenerateRandomMatrixCRS(3, 3, 4);
  MatrixCRS matrix_B = GenerateRandomMatrixCRS(3, 3, 4);

  MatrixCRS matrix_C;
  t1 = omp_get_wtime();
  CRSMultiply(matrix_A, matrix_B, &matrix_C);
  t2 = omp_get_wtime();

  MatrixCRS C_omp;
  t1_ = omp_get_wtime();
  CRSMultiplyOMP(matrix_A, matrix_B, &C_omp);
  t2_ = omp_get_wtime();
  std::cout << "NO OMP time = " << t2 - t1 << "\nOMP time = " << t2_ - t1_
            << "\nAcceleration = " << (t2 - t1) / (t2_ - t1_) << "\n";
  ASSERT_TRUE(CompareMatrixCRS(matrix_C, C_omp));
}

TEST(Multiplication_parallel, 10x10_7) {
  double t1, t2, t1_, t2_;

  MatrixCRS matrix_A = GenerateRandomMatrixCRS(10, 10, 7);
  MatrixCRS matrix_B = GenerateRandomMatrixCRS(10, 10, 7);

  MatrixCRS matrix_C;
  t1 = omp_get_wtime();
  CRSMultiply(matrix_A, matrix_B, &matrix_C);
  t2 = omp_get_wtime();

  MatrixCRS C_omp;
  t1_ = omp_get_wtime();
  CRSMultiplyOMP(matrix_A, matrix_B, &C_omp);
  t2_ = omp_get_wtime();
  std::cout << "NO OMP time = " << t2 - t1 << "\nOMP time = " << t2_ - t1_
            << "\nAcceleration = " << (t2 - t1) / (t2_ - t1_) << "\n";
  ASSERT_TRUE(CompareMatrixCRS(matrix_C, C_omp));
}

TEST(Multiplication_parallel, 15x15_12) {
  double t1, t2, t1_, t2_;

  MatrixCRS matrix_A = GenerateRandomMatrixCRS(15, 15, 12);
  MatrixCRS matrix_B = GenerateRandomMatrixCRS(15, 15, 12);

  MatrixCRS matrix_C;
  t1 = omp_get_wtime();
  CRSMultiply(matrix_A, matrix_B, &matrix_C);
  t2 = omp_get_wtime();

  MatrixCRS C_omp;
  t1_ = omp_get_wtime();
  CRSMultiplyOMP(matrix_A, matrix_B, &C_omp);
  t2_ = omp_get_wtime();
  std::cout << "NO OMP time = " << t2 - t1 << "\nOMP time = " << t2_ - t1_
            << "\nAcceleration = " << (t2 - t1) / (t2_ - t1_) << "\n";
  ASSERT_TRUE(CompareMatrixCRS(matrix_C, C_omp));
}

TEST(Multiplication_parallel, 20x20_25) {
  double t1, t2, t1_, t2_;

  MatrixCRS matrix_A = GenerateRandomMatrixCRS(20, 20, 40);
  MatrixCRS matrix_B = GenerateRandomMatrixCRS(20, 20, 40);

  MatrixCRS matrix_C;
  t1 = omp_get_wtime();
  CRSMultiply(matrix_A, matrix_B, &matrix_C);
  t2 = omp_get_wtime();

  MatrixCRS C_omp;
  t1_ = omp_get_wtime();
  CRSMultiplyOMP(matrix_A, matrix_B, &C_omp);
  t2_ = omp_get_wtime();
  std::cout << "NO OMP time = " << t2 - t1 << "\nOMP time = " << t2_ - t1_
            << "\nAcceleration = " << (t2 - t1) / (t2_ - t1_) << "\n";
  ASSERT_TRUE(CompareMatrixCRS(matrix_C, C_omp));
}

TEST(Multiplication_parallel, 25x25_35) {
  double t1, t2, t1_, t2_;

  MatrixCRS matrix_A = GenerateRandomMatrixCRS(25, 25, 30);
  MatrixCRS matrix_B = GenerateRandomMatrixCRS(25, 25, 30);

  MatrixCRS matrix_C;
  t1 = omp_get_wtime();
  CRSMultiply(matrix_A, matrix_B, &matrix_C);
  t2 = omp_get_wtime();

  MatrixCRS C_omp;
  t1_ = omp_get_wtime();
  CRSMultiplyOMP(matrix_A, matrix_B, &C_omp);
  t2_ = omp_get_wtime();
  std::cout << "NO OMP time = " << t2 - t1 << "\nOMP time = " << t2_ - t1_
            << "\nAcceleration = " << (t2 - t1) / (t2_ - t1_) << "\n";
  ASSERT_TRUE(CompareMatrixCRS(matrix_C, C_omp));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
