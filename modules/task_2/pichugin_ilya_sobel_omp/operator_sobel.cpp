// Copyright 2022 Pichugin Ilya
#include "../../../modules/task_2/pichugin_ilya_sobel_omp/operator_sobel.h"

#include <omp.h>

#include <algorithm>
#include <cmath>

Sobel::Sobel() {
  mSize = 0;
  matrix = nullptr;
}

Sobel::~Sobel() {
  if (matrix) {
    delete[] matrix;
  }
}

Sobel::Sobel(size_t m_size) {
  mSize = m_size;
  matrix = new int[m_size * m_size];

  for (size_t i = 0; i < m_size * m_size; i++) {
    matrix[i] = 0;
  }
}

Sobel::Sobel(const Sobel& c) {
  mSize = c.mSize;
  matrix = new int[mSize * mSize];

  for (size_t i = 0; i < mSize * mSize; i++) {
    matrix[i] = c.matrix[i];
  }
}

Sobel Sobel::operator=(const Sobel& c) {
  mSize = c.mSize;
  matrix = new int[mSize * mSize];

  for (size_t i = 0; i < mSize * mSize; i++) {
    matrix[i] = c.matrix[i];
  }
  return *this;
}

bool Sobel::operator==(const Sobel& c) const {
  for (size_t i = 0; i < mSize * mSize; i++) {
    if (matrix[i] != c.matrix[i]) {
      return false;
    }
  }
  return true;
}

Sobel Sobel::RandomMatrix(size_t m_size) {
  Sobel sob_Class(m_size);
  int min = 0, max = 255;
  int* Matrix = new int[m_size];

  for (size_t i = 0; i < m_size; i++) {
    Matrix[i] = min + (std::rand() * ((max + 1) - min) / RAND_MAX);
    sob_Class.matrix[i] = Matrix[i];
  }

  delete[] Matrix;
  return sob_Class;
}

Sobel Sobel::operator_Sobel(Sobel c) {
  int _size = c.get_Size_Matrix();
  Sobel matrix(_size);
  _size--;

  int kern_x[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  int kern_y[] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

  for (int i = 1; i < _size; i++) {
    for (int j = 1; j < _size; j++) {
      int Gx = 0, Gy = 0, ind = 0, min_pix = 0, max_pix = 255;

      for (int m_i = -1; m_i < 2; m_i++) {
        for (int m_j = -1; m_j < 2; m_j++) {
          Gx += kern_x[ind] * c.get_Matrix(i + m_i, j + m_j);
          Gy += kern_y[ind] * c.get_Matrix(i + m_i, j + m_j);
          ind++;
        }
      }

      double val_gradient = sqrt(Gx * Gx + Gy * Gy);
      int range = val_gradient < min_pix   ? min_pix
                  : val_gradient > max_pix ? max_pix
                                           : val_gradient;
      matrix.set_Matrix(i, j, range);
    }
  }
  return matrix;
}

Sobel Sobel::operator_Sobel_parr(Sobel c) {
  int _size = c.get_Size_Matrix();
  Sobel matrix(_size);
  _size--;

  int kern_x[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  int kern_y[] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

#pragma omp parallel for
  for (int i = 1; i < _size; i++) {
    for (int j = 1; j < _size; j++) {
      int Gx = 0, Gy = 0, ind = 0, min_pix = 0, max_pix = 255;

      for (int m_i = -1; m_i < 2; m_i++) {
        for (int m_j = -1; m_j < 2; m_j++) {
          Gx += kern_x[ind] * c.get_Matrix(i + m_i, j + m_j);
          Gy += kern_y[ind] * c.get_Matrix(i + m_i, j + m_j);
          ind++;
        }
      }

      double val_gradient = sqrt(Gx * Gx + Gy * Gy);
      int range = val_gradient < min_pix   ? min_pix
                  : val_gradient > max_pix ? max_pix
                                           : val_gradient;
      matrix.set_Matrix(i, j, range);
    }
  }
  return matrix;
}

int Sobel::get_Size_Matrix() { return mSize; }
int Sobel::get_Matrix(int i, int j) { return matrix[mSize * i + j]; }
void Sobel::set_Matrix(int i, int j, int val_gradient) {
  matrix[mSize * i + j] = val_gradient;
}
