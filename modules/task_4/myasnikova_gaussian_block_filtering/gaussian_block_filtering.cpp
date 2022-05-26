// Copyright 2022 Myasnikova Varvara

#include "../../../3rdparty/unapproved/unapproved.h"
#include "../../../modules/task_4/myasnikova_gaussian_block_filtering/gaussian_block_filtering.h"

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

      if (result[i] > 255) {
        result[i] = 255;
      } else if (result[i] < 0) {
        result[i] = 0;
      }
    }
  }

  return result;
}

std::vector<int> GaussFilterParallel(std::vector<int> matrix,
  std::vector<int> kernel, int row, int column) {
  std::vector<int> result(matrix);
  const int count_thread = std::thread::hardware_concurrency();
  std::thread* ths = new std::thread[count_thread];
  int thread_elements = row / count_thread;
  int ost = row % count_thread;
  int start_index, end_index = column + 1;
  for (int k = 0; k < count_thread; k++) {
    start_index = end_index;
    end_index += (column * thread_elements);
    if (ost > 0) {
      end_index++;
      ost--;
    }
    if (k == count_thread - 1) {
      end_index = row * column - column - 1;
    }
    ths[k] = std::thread([&](int begin, int end) {
      int region[9] = { 0 };
      for (int i = begin; i < end; ++i) {
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
          if (result[i] > 255) {
            result[i] = 255;
          } else if (result[i] < 0) {
            result[i] = 0;
          }
        }
      }
      }, start_index, end_index);
  }
  for (int i = 0; i < count_thread; i++) {
    ths[i].join();
  }
  delete[] ths;
  return result;
}
