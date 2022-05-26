// Copyright 2022 Pudovkin Artem
#include <gtest/gtest.h>
#include <vector>

#include "./pudovkin_a_linear_filtering.h"

using v_size = std::vector<std::vector<double>>::size_type;

bool IsVectorsEquals(const std::vector<std::vector<double>>& firstMatrix,
                     const std::vector<std::vector<double>>& secondMatrix) {
  for (v_size i = 0; i < firstMatrix.size(); ++i) {
    for (v_size j = 0; j < firstMatrix[i].size(); ++j) {
      if (firstMatrix[i][j] != secondMatrix[i][j])
        return false;
    }
  }

  return true;
}

TEST(ParallelFilter, TestMatrix5x5) {
  auto matrixOfColor = getRandomMatrix(5, 5);

  auto resultSeqMatrix = getSequentialFilter(matrixOfColor);
  auto resultParallelMatrix = getParallelFilter(matrixOfColor);

  bool isEqual = IsVectorsEquals(resultSeqMatrix, resultParallelMatrix);

  ASSERT_EQ(isEqual, true);
}

TEST(ParallelFilter, TestMatrix6x6) {
  auto matrixOfColor = getRandomMatrix(6, 6);

  auto resultSeqMatrix = getSequentialFilter(matrixOfColor);
  auto resultParallelMatrix = getParallelFilter(matrixOfColor);

  bool isEqual = IsVectorsEquals(resultSeqMatrix, resultParallelMatrix);

  ASSERT_EQ(isEqual, true);
}

TEST(ParallelFilter, TestMatrix7x7) {
  auto matrixOfColor = getRandomMatrix(7, 7);

  auto resultSeqMatrix = getSequentialFilter(matrixOfColor);
  auto resultParallelMatrix = getParallelFilter(matrixOfColor);

  bool isEqual = IsVectorsEquals(resultSeqMatrix, resultParallelMatrix);

  ASSERT_EQ(isEqual, true);
}

TEST(ParallelFilter, TestMatrix8x8) {
  auto matrixOfColor = getRandomMatrix(8, 8);

  auto resultSeqMatrix = getSequentialFilter(matrixOfColor);
  auto resultParallelMatrix = getParallelFilter(matrixOfColor);

  bool isEqual = IsVectorsEquals(resultSeqMatrix, resultParallelMatrix);

  ASSERT_EQ(isEqual, true);
}

TEST(ParallelFilter, TestMatrix9x9) {
  auto matrixOfColor = getRandomMatrix(9, 9);

  auto resultSeqMatrix = getSequentialFilter(matrixOfColor);
  auto resultParallelMatrix = getParallelFilter(matrixOfColor);

  bool isEqual = IsVectorsEquals(resultSeqMatrix, resultParallelMatrix);

  ASSERT_EQ(isEqual, true);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
