// Copyright 2022 Utyugov Denis
#include "../../../modules/task_1/utyugov_d_cannons_alg/cannons_alg.h"

#include <random>
#include <vector>

std::vector<double> getRandomVector(int size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<double> vec(size);
  for (int i = 0; i < size; i++) {
    vec[i] = gen() % size;
  }
  return vec;
}

std::vector<std::vector<double>> getRndMatrix(int size) {
  std::vector<std::vector<double>> A(size, std::vector<double>(size));
  for (int i = 0; i < size; i++) {
    A[i] = getRandomVector(size);
  }
  return A;
}

std::vector<std::vector<double>> Multiplicate(
    std::vector<std::vector<double>> A, std::vector<std::vector<double>> B) {
  int size = A.size();
  std::vector<std::vector<double>> C(size, std::vector<double>(size));
  for (int i = 0; i < size; i++)
    for (int j = 0; j < size; j++)
      for (int k = 0; k < size; k++) C[i][j] += A[i][k] * B[k][j];
  return C;
}

std::vector<std::vector<double>> BlockMultiplicate(
    std::vector<std::vector<double>> A, std::vector<std::vector<double>> B,
    int BlSize) {
  int size = A.size();
  std::vector<std::vector<double>> C(size, std::vector<double>(size, 0));
  int Minj, Mink = 0;

  for (int Globj = 0; Globj < size; Globj += BlSize) {
    Minj = std::min(Globj + BlSize, size);
    for (int Globk = 0; Globk < size; Globk += BlSize) {
      Mink = std::min(Globk + BlSize, size);

      for (int i = 0; i < size; i++)
        for (int j = Globj; j < Minj; j++)
          for (int k = Globk; k < Mink; k++) {
            C[i][j] += A[i][k] * B[k][j];
          }
    }
  }
  return C;
}
