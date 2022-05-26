// Copyright 2022 Pichugin Ilya
#ifndef MODULES_TASK_3_PICHUGIN_I_SOBEL_TBB_OPERATOR_SOBEL_H_
#define MODULES_TASK_3_PICHUGIN_I_SOBEL_TBB_OPERATOR_SOBEL_H_

#include <tbb/tbb.h>

#include <iostream>

class Sobel {
 private:
  int* matrix;
  size_t mSize;

 public:
  Sobel();
  explicit Sobel(size_t m_size);
  Sobel(const Sobel& c);
  ~Sobel();

  Sobel operator=(const Sobel& c);
  bool operator==(const Sobel& c) const;
  Sobel RandomMatrix(size_t m_size);
  Sobel operator_Sobel(Sobel c);
  Sobel operator_Sobel_tbb(Sobel c);

  int get_Size_Matrix();
  int get_Matrix(int i, int j);
  void set_Matrix(int i, int j, int val_gradient);
};
#endif  // MODULES_TASK_3_PICHUGIN_I_SOBEL_TBB_OPERATOR_SOBEL_H_
