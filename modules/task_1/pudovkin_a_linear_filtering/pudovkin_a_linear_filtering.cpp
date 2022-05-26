// Copyright 2022 Pudovkin Artem
#include <random>
#include <vector>

#include "../../../modules/task_1/pudovkin_a_linear_filtering/pudovkin_a_linear_filtering.h"

using std::vector;
using v_size = vector<vector<double>>::size_type;

vector<vector<double>> getRandomMatrix(const v_size rows, const v_size cols) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<> urd(0, 1);
  vector<vector<double>> result(rows);

  for (v_size i = 0; i < rows; ++i) {
    result[i] = vector<double>(cols);
    for (v_size j = 0; j < cols; ++j) {
      result[i][j] = urd(gen);
    }
  }

  return result;
}

template <typename T>
T clamp(T value, T min, T max) {
  if (value > max)
    return max;
  else if (value < min)
    return min;

  return value;
}

double calculateNewPixelColor(const vector<vector<double>>& colorMatrix,
                              int coordinateX,
                              int coordinateY) {
  double result = 0;

  for (int i = -1; i <= 1; i++) {
    for (int j = -1; j <= 1; j++) {
      result +=
          colorMatrix[clamp<int>(coordinateX + j, 0, colorMatrix.size() - 1)]
                     [clamp<int>(coordinateY + i, 0, colorMatrix.size() - 1)] *
          (1.0 / 9.0);
    }
  }

  return clamp<double>(result, 0, 1);
}

vector<vector<double>> getSequentialFilter(
    const vector<vector<double>>& commonMatrixOfColor) {
  vector<vector<double>> result = commonMatrixOfColor;

  for (v_size i = 0; i < commonMatrixOfColor.size(); ++i) {
    for (v_size j = 0; j < commonMatrixOfColor[i].size(); ++j) {
      result[i][j] = calculateNewPixelColor(commonMatrixOfColor, i, j);
    }
  }

  return result;
}
