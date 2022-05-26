// Copyright 2022 Pudovkin Artem
#include <gtest/gtest.h>
#include <vector>

#include "./pudovkin_a_linear_filtering.h"

using v_size = std::vector<std::vector<double>>::size_type;

bool isRightValues(const std::vector<std::vector<double>>& matrix) {
  for (v_size i = 0; i < matrix.size(); ++i) {
    for (v_size j = 0; j < matrix[i].size(); ++j) {
      if (matrix[i][j] > 1 || matrix[i][j] < 0)
        return false;
    }
  }

  return true;
}

TEST(SequentialFilter, TestMatrix5x5) {
  auto matrixOfColor = getRandomMatrix(5, 5);

  auto resultMatrix = getSequentialFilter(matrixOfColor);
  bool isRight = isRightValues(resultMatrix);

  ASSERT_EQ(isRight, true);
}

TEST(SequentialFilter, TestMatrix6x6) {
  auto matrixOfColor = getRandomMatrix(6, 6);

  auto resultMatrix = getSequentialFilter(matrixOfColor);
  bool isRight = isRightValues(resultMatrix);

  ASSERT_EQ(isRight, true);
}

TEST(SequentialFilter, TestMatrix7x7) {
  auto matrixOfColor = getRandomMatrix(7, 7);

  auto resultMatrix = getSequentialFilter(matrixOfColor);
  bool isRight = isRightValues(resultMatrix);

  ASSERT_EQ(isRight, true);
}

TEST(SequentialFilter, TestMatrix8x8) {
  auto matrixOfColor = getRandomMatrix(8, 8);

  auto resultMatrix = getSequentialFilter(matrixOfColor);
  bool isRight = isRightValues(resultMatrix);

  ASSERT_EQ(isRight, true);
}

TEST(SequentialFilter, TestMatrix9x9) {
  auto matrixOfColor = getRandomMatrix(9, 9);

  auto resultMatrix = getSequentialFilter(matrixOfColor);
  bool isRight = isRightValues(resultMatrix);

  ASSERT_EQ(isRight, true);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
