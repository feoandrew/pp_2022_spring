// Copyright 2022 Kitaev Pavel

#include <gtest/gtest.h>
#include "../../../modules/task_4/kitaev_p_block_gauss/block_gauss.h"

TEST(Gauss_STD, CanCreateRandomMatrix) {
  ASSERT_NO_THROW(getRandomMatrix(100, 100));
}

TEST(Gauss_TBB, CanCreateGaussKernel) {
  ASSERT_NO_THROW(getGaussKernel(10));
}

TEST(Gauss_STD, CanSequentialAlg) {
  std::vector<int> mtr = getRandomMatrix(10, 10);
  std::vector<double> gauss_kernel = getGaussKernel(10);
  ASSERT_NO_THROW(SequentialGauss(mtr, gauss_kernel, 10, 10));
}

TEST(Gauss_STD, CanParallelAld) {
  std::vector<int> mtr = getRandomMatrix(10, 10);
  std::vector<double> gauss_kernel = getGaussKernel(10);
  ASSERT_NO_THROW(ParallelGauss(mtr, gauss_kernel, 10, 10));
}

TEST(Gauss_STD, test_10x10) {
  std::vector<int> mtr = getRandomMatrix(10, 10);
  std::vector<double> gauss_kernel = getGaussKernel(10);
  std::vector<int> res_sq = SequentialGauss(mtr, gauss_kernel, 10, 10);
  std::vector<int> res_pp = ParallelGauss(mtr, gauss_kernel, 10, 10);
  ASSERT_EQ(res_sq, res_pp);
}

TEST(Gauss_STD, test_50x50) {
  std::vector<int> mtr = getRandomMatrix(50, 50);
  std::vector<double> gauss_kernel = getGaussKernel(10);
  std::vector<int> res_sq = SequentialGauss(mtr, gauss_kernel, 50, 50);
  std::vector<int> res_pp = ParallelGauss(mtr, gauss_kernel, 50, 50);
  ASSERT_EQ(res_sq, res_pp);
}

/*
TEST(Gauss_STD, test_time_2000x2000) {
  std::vector<int> mtr = getRandomMatrix(2000, 2000);
  std::vector<double> gauss_kernel = getGaussKernel(10);

  double t1_seq = clock();
  std::vector<int> res_sq = SequentialGauss(mtr, gauss_kernel, 2000, 2000);
  double t2_seq = clock();

  double t1_pp = clock();
  std::vector<int> res_pp = ParallelGauss(mtr, gauss_kernel, 2000, 2000);
  double t2_pp = clock();

  std::cout << "SQ TIME: " << static_cast<float>(t2_seq - t1_seq) / CLOCKS_PER_SEC << std::endl;
  std::cout << "PP TIME: " << static_cast<float>(t2_pp - t1_pp) / CLOCKS_PER_SEC << std::endl;

  ASSERT_EQ(res_sq, res_pp);
}

TEST(Gauss_STD, test_time_5000x5000) {
  std::vector<int> mtr = getRandomMatrix(5000, 5000);
  std::vector<double> gauss_kernel = getGaussKernel(10);

  double t1_seq = clock();
  std::vector<int> res_sq = SequentialGauss(mtr, gauss_kernel, 5000, 5000);
  double t2_seq = clock();

  double t1_pp = clock();
  std::vector<int> res_pp = ParallelGauss(mtr, gauss_kernel, 5000, 5000);
  double t2_pp = clock();

  std::cout << "SQ TIME: " << static_cast<float>(t2_seq - t1_seq) / CLOCKS_PER_SEC << std::endl;
  std::cout << "PP TIME: " << static_cast<float>(t2_pp - t1_pp) / CLOCKS_PER_SEC << std::endl;

  ASSERT_EQ(res_sq, res_pp);
}

TEST(Gauss_STD, test_time_10000x10000) {
  std::vector<int> mtr = getRandomMatrix(10000, 10000);
  std::vector<double> gauss_kernel = getGaussKernel(10);

  double t1_seq = clock();
  std::vector<int> res_sq = SequentialGauss(mtr, gauss_kernel, 10000, 10000);
  double t2_seq = clock();

  double t1_pp = clock();
  std::vector<int> res_pp = ParallelGauss(mtr, gauss_kernel, 10000, 10000);
  double t2_pp = clock();

  std::cout << "SQ TIME: " << static_cast<float>(t2_seq - t1_seq) / CLOCKS_PER_SEC << std::endl;
  std::cout << "PP TIME: " << static_cast<float>(t2_pp - t1_pp) / CLOCKS_PER_SEC << std::endl;

  ASSERT_EQ(res_sq, res_pp);
}
*/


int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
