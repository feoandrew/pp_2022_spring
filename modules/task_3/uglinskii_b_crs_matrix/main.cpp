// Copyright 2022 Uglinskii Bogdan
#include <gtest/gtest.h>

#include <iostream>
#include <vector>

#include "./crs_multiplication.h"
#include "tbb/tick_count.h"

TEST(Multiplication_seq, crs_50x50_30) {
  MatrixCRS matrix_A = GenerateRandomMatrixCRS(50, 50, 30);
  MatrixCRS matrix_B = GenerateRandomMatrixCRS(50, 50, 30);

  MatrixCRS matrix_C;
  CRSMultiply(matrix_A, matrix_B, &matrix_C);

  std::vector<std::vector<double>> C;

  NormalMulty(ExpandMatrix(matrix_A), ExpandMatrix(matrix_B), &C);
  MatrixCRS crs_C;
  ConvertToCRS(C, &crs_C);
  ASSERT_TRUE(CompareMatrixCRS(matrix_C, crs_C));
}

TEST(Multiplication_parallel, 3x3_4) {
  tbb::tick_count t1, t2, t1_, t2_;

  MatrixCRS matrix_A = GenerateRandomMatrixCRS(3, 3, 4);
  MatrixCRS matrix_B = GenerateRandomMatrixCRS(3, 3, 4);

  MatrixCRS matrix_C;
  t1 = tbb::tick_count::now();
  CRSMultiply(matrix_A, matrix_B, &matrix_C);
  t2 = tbb::tick_count::now();

  MatrixCRS C_tbb;
  t1_ = tbb::tick_count::now();
  CRSMultiplyTBB(matrix_A, matrix_B, &C_tbb);
  t2_ = tbb::tick_count::now();
  std::cout << "NO TBB time = " << (t2 - t1).seconds()
            << "\nTBB time = " << (t2_ - t1_).seconds() << "\nAcceleration = "
            << (t2 - t1).seconds() / (t2_ - t1_).seconds() << "\n";
  ASSERT_TRUE(CompareMatrixCRS(matrix_C, C_tbb));
}

TEST(Multiplication_parallel, 15x15_25) {
  tbb::tick_count t1, t2, t1_, t2_;

  MatrixCRS matrix_A = GenerateRandomMatrixCRS(15, 15, 25);
  MatrixCRS matrix_B = GenerateRandomMatrixCRS(15, 15, 25);

  MatrixCRS matrix_C;
  t1 = tbb::tick_count::now();
  CRSMultiply(matrix_A, matrix_B, &matrix_C);
  t2 = tbb::tick_count::now();

  MatrixCRS C_tbb;
  t1_ = tbb::tick_count::now();
  CRSMultiplyTBB(matrix_A, matrix_B, &C_tbb);
  t2_ = tbb::tick_count::now();
  std::cout << "NO TBB time = " << (t2 - t1).seconds()
            << "\nTBB time = " << (t2_ - t1_).seconds() << "\nAcceleration = "
            << (t2 - t1).seconds() / (t2_ - t1_).seconds() << "\n";
  ASSERT_TRUE(CompareMatrixCRS(matrix_C, C_tbb));
}

TEST(Multiplication_parallel, 20x20_50) {
  tbb::tick_count t1, t2, t1_, t2_;

  MatrixCRS matrix_A = GenerateRandomMatrixCRS(20, 20, 50);
  MatrixCRS matrix_B = GenerateRandomMatrixCRS(20, 20, 50);

  MatrixCRS matrix_C;
  t1 = tbb::tick_count::now();
  CRSMultiply(matrix_A, matrix_B, &matrix_C);
  t2 = tbb::tick_count::now();

  MatrixCRS C_tbb;
  t1_ = tbb::tick_count::now();
  CRSMultiplyTBB(matrix_A, matrix_B, &C_tbb);
  t2_ = tbb::tick_count::now();
  std::cout << "NO TBB time = " << (t2 - t1).seconds()
            << "\nTBB time = " << (t2_ - t1_).seconds() << "\nAcceleration = "
            << (t2 - t1).seconds() / (t2_ - t1_).seconds() << "\n";
  ASSERT_TRUE(CompareMatrixCRS(matrix_C, C_tbb));
}

TEST(Multiplication_parallel, 25x25_65) {
  tbb::tick_count t1, t2, t1_, t2_;

  MatrixCRS matrix_A = GenerateRandomMatrixCRS(25, 25, 65);
  MatrixCRS matrix_B = GenerateRandomMatrixCRS(25, 25, 65);

  MatrixCRS matrix_C;
  t1 = tbb::tick_count::now();
  CRSMultiply(matrix_A, matrix_B, &matrix_C);
  t2 = tbb::tick_count::now();

  MatrixCRS C_tbb;
  t1_ = tbb::tick_count::now();
  CRSMultiplyTBB(matrix_A, matrix_B, &C_tbb);
  t2_ = tbb::tick_count::now();
  std::cout << "NO TBB time = " << (t2 - t1).seconds()
            << "\nTBB time = " << (t2_ - t1_).seconds() << "\nAcceleration = "
            << (t2 - t1).seconds() / (t2_ - t1_).seconds() << "\n";
  ASSERT_TRUE(CompareMatrixCRS(matrix_C, C_tbb));
}

TEST(Multiplication_parallel, 35x35_75) {
  tbb::tick_count t1, t2, t1_, t2_;

  MatrixCRS matrix_A = GenerateRandomMatrixCRS(35, 35, 75);
  MatrixCRS matrix_B = GenerateRandomMatrixCRS(35, 35, 75);

  MatrixCRS matrix_C;
  t1 = tbb::tick_count::now();
  CRSMultiply(matrix_A, matrix_B, &matrix_C);
  t2 = tbb::tick_count::now();

  MatrixCRS C_tbb;
  t1_ = tbb::tick_count::now();
  CRSMultiplyTBB(matrix_A, matrix_B, &C_tbb);
  t2_ = tbb::tick_count::now();
  std::cout << "NO TBB time = " << (t2 - t1).seconds()
            << "\nTBB time = " << (t2_ - t1_).seconds() << "\nAcceleration = "
            << (t2 - t1).seconds() / (t2_ - t1_).seconds() << "\n";
  ASSERT_TRUE(CompareMatrixCRS(matrix_C, C_tbb));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
