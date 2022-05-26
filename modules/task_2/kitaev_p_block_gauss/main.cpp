// Copyright 2022 Kitaev Pavel

#include <gtest/gtest.h>
#include <vector>
#include "../../../modules/task_2/kitaev_p_block_gauss/block_gauss.h"

TEST(Sequential_gauss, CanCreateRandomMatrix) {
  ASSERT_NO_THROW(getRandomMatrix(100, 100));
}

TEST(Sequential_gauss, CanCreateGaussKernel) {
  ASSERT_NO_THROW(getGaussKernel(10));
}

TEST(Sequential_gauss, CanSequentialAlg) {
  std::vector<int> mtr = getRandomMatrix(10, 10);
  std::vector<double> gauss_kernel = getGaussKernel(10);
  ASSERT_NO_THROW(SequentialGauss(mtr, gauss_kernel, 10, 10));
}

TEST(Sequential_gauss, CanParallelAld) {
  std::vector<int> mtr = getRandomMatrix(10, 10);
  std::vector<double> gauss_kernel = getGaussKernel(10);
  ASSERT_NO_THROW(ParallelGauss(mtr, gauss_kernel, 10, 10));
}

TEST(Sequential_gauss, test_10x10) {
  std::vector<int> mtr = getRandomMatrix(10, 10);
  std::vector<double> gauss_kernel = getGaussKernel(10);
  std::vector<int> res_sq = SequentialGauss(mtr, gauss_kernel, 10, 10);
  std::vector<int> res_pp = ParallelGauss(mtr, gauss_kernel, 10, 10);
  ASSERT_EQ(res_sq, res_pp);
}

TEST(Sequential_gauss, test_50x50) {
  std::vector<int> mtr = getRandomMatrix(50, 50);
  std::vector<double> gauss_kernel = getGaussKernel(10);
  std::vector<int> res_sq = SequentialGauss(mtr, gauss_kernel, 50, 50);
  std::vector<int> res_pp = ParallelGauss(mtr, gauss_kernel, 50, 50);
  ASSERT_EQ(res_sq, res_pp);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
