// Copyright 2022 Kitaev Pavel

#include <gtest/gtest.h>
#include <vector>
#include "../../../modules/task_1/kitaev_p_block_gauss/block_gauss.h"

TEST(Sequential_gauss, CanCreateRandomMatrix) {
  ASSERT_NO_THROW(getRandomMatrix(100, 100));
}

TEST(Sequential_gauss, CanCreateGaussKernel) {
  ASSERT_NO_THROW(getGaussKernel(10));
}

TEST(Sequential_gauss, test_10x10) {
  std::vector<int> mtr = getRandomMatrix(10, 10);
  ASSERT_NO_THROW(SequentialGauss(mtr, 10, 10, 10));
}

TEST(Sequential_gauss, test_20x20) {
  std::vector<int> mtr = getRandomMatrix(20, 20);
  ASSERT_NO_THROW(SequentialGauss(mtr, 20, 20, 10));
}

TEST(Sequential_gauss, test_30x30) {
  std::vector<int> mtr = getRandomMatrix(30, 30);
  ASSERT_NO_THROW(SequentialGauss(mtr, 30, 30, 10));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
