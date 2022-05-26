// Copyright 2022 Myasnikova Varvara

#include "../../../modules/task_1/myasnikova_gaussian_block_filtering/gaussian_block_filtering.h"

std::vector<int> CreateMatrix(int row, int column) {
  std::random_device rd;
  std::mt19937 generator(rd());
  std::vector<int> vec(row * column);

  for (int i = 0; i < row; ++i) {
    for (int j = 0; j < column; ++j) {
      vec[i * column + j] = generator() % 256;
    }
  }
  return vec;
}

std::vector<int> CreateKernel(double sigma) {
  double normalization = 0;
  std::vector<int> kernel(9, 0);

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      kernel[i * 3 + j] = exp(-(i * 3. + j) *
        (i * 3. + j) / (2. * sigma * sigma));
      normalization += kernel[i * 3 + j];
    }
  }

  for (int i = 0; i < 9; i++) {
    if (normalization != 0) {
      kernel[i] /= normalization;
    }
  }

  return kernel;
}

std::vector<int> GaussFilterSeq(std::vector<int> matrix,
  std::vector<int> kernel, int row, int column) {
  std::vector<int> result(matrix);
  int region[9] = { 0 };

  for (int i = column + 1; i < row * column - column - 1; ++i) {
    if (i % column != 0 && (i + 1) % column != 0) {
      region[0] = i - column - 1;
      region[1] = i - column;
      region[2] = i - column + 1;
      region[3] = i - 1;
      region[4] = i;
      region[5] = i + 1;
      region[6] = i + column - 1;
      region[7] = i + column;
      region[8] = i + column + 1;

      double sum = 0;
      for (int j = 0; j < 9; ++j) {
        sum += static_cast<double>(matrix[region[j]]) * kernel[j];
      }

      result[i] = static_cast<int>(sum);
    }
  }

  return result;
}
