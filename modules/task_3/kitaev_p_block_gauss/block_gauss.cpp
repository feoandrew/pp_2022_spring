// Copyright 2022 Kitaev Pavel
#include <tbb/tbb.h>
#include "../../../modules/task_3/kitaev_p_block_gauss/block_gauss.h"

std::vector<int> getRandomMatrix(int row, int col) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(row * col);
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      vec[i * col + j] = gen() % 256;
    }
  }
  return vec;
}

std::vector<int> getGaussKernel(double sigma) {
  double norm = 0;
  std::vector<int> gauss_kernel(9, 0);

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      gauss_kernel[i * 3 + j] = exp(-(i * 3. + j) *
        (i * 3. + j) / (2. * sigma * sigma));
      norm += gauss_kernel[i * 3 + j];
    }
  }

  for (std::vector<int>::size_type i = 0; i < 9; i++) {
    if (norm != 0) {
      gauss_kernel[i] /= norm;
    }
  }

  return gauss_kernel;
}

std::vector<int> SequentialGauss(std::vector<int> img,
  std::vector<int> gauss_kernel, int row, int col) {
  std::vector<int> res(img);
  int block[9] = { 0 };
  for (int i = col + 1; i < row * col - col - 1; i++) {
    if (i % col != 0 && (i + 1) % col != 0) {
      block[0] = i - col - 1;
      block[1] = i - col;
      block[2] = i - col + 1;
      block[3] = i - 1;
      block[4] = i;
      block[5] = i + 1;
      block[6] = i + col - 1;
      block[7] = i + col;
      block[8] = i + col + 1;

      double sum = 0;
      for (int j = 0; j < 9; j++) {
        sum += static_cast<double>(img[block[j]]) * gauss_kernel[j];
      }

      res[i] = static_cast<int>(sum / 9);
    }
  }

  return res;
}

std::vector<int> ParallelGauss(std::vector<int> img,
  std::vector<int> gauss_kernel, int row, int col) {
  std::vector<int> res(img);
  tbb::task_scheduler_init init;

  tbb::parallel_for(tbb::blocked_range<int>(col + 1, row * col - col - 1, 1),
    [&](const tbb::blocked_range<int>& range) {
    int block[9] = { 0 };
    for (int i = range.begin(); i < range.end(); i++) {
      if (i % col != 0 && (i + 1) % col != 0) {
        block[0] = i - col - 1;
        block[1] = i - col;
        block[2] = i - col + 1;
        block[3] = i - 1;
        block[4] = i;
        block[5] = i + 1;
        block[6] = i + col - 1;
        block[7] = i + col;
        block[8] = i + col + 1;

        double sum = 0;
        for (int j = 0; j < 9; j++) {
          sum += static_cast<double>(img[block[j]]) * gauss_kernel[j];
        }
        res[i] = static_cast<int>(sum / 9);
      }
    }
    });

  return res;
}
