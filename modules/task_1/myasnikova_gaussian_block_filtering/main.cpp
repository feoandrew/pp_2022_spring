// Copyright 2022 Myasnikova Varvara

#include <gtest/gtest.h>
#include <vector>
#include "../../../modules/task_1/myasnikova_gaussian_block_filtering/gaussian_block_filtering.h"

TEST(SeqGaussian, Creating_a_Matrix_small) {
  int row = 10, column = 10;
  ASSERT_NO_THROW(CreateMatrix(row, column));
}

TEST(SeqGaussian, Creating_a_Matrix_big) {
  int row = 100, column = 100;
  ASSERT_NO_THROW(CreateMatrix(row, column));
}

TEST(SeqGaussian, Creating_a_Kernel_1) {
  ASSERT_NO_THROW(CreateKernel(1));
}

TEST(SeqGaussian, Creating_a_Kernel_2) {
  ASSERT_NO_THROW(CreateKernel(5));
}

TEST(SeqGaussian, Creating_a_Kernel_3) {
  ASSERT_NO_THROW(CreateKernel(10));
}

TEST(SeqGaussian, GaussFilterSeq_small) {
  int row = 10, column = 10;
  std::vector<int> matrix = CreateMatrix(row, column);
  std::vector<int> kernel = CreateKernel(5);
  ASSERT_NO_THROW(GaussFilterSeq(matrix, kernel, row, column));
}

TEST(SeqGaussian, GaussFilterSeq_big) {
  int row = 100, column = 100;
  std::vector<int> matrix = CreateMatrix(row, column);
  std::vector<int> kernel = CreateKernel(5);
  ASSERT_NO_THROW(GaussFilterSeq(matrix, kernel, row, column));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
