// Copyright 2022 Pichugin Ilya
#ifndef MODULES_TASK_2_PICHUGIN_ILYA_SOBEL_OMP_OPERATOR_SOBEL_H_
#define MODULES_TASK_2_PICHUGIN_ILYA_SOBEL_OMP_OPERATOR_SOBEL_H_

#include <iostream>

class Sobel {
  int* matrix;
  size_t mSize;

 public:
  Sobel();
  ~Sobel();
  explicit Sobel(size_t m_size);
  Sobel(const Sobel& c);

  Sobel operator=(const Sobel& c);
  bool operator==(const Sobel& c) const;
  Sobel RandomMatrix(size_t m_size);
  Sobel operator_Sobel(Sobel c);
  Sobel operator_Sobel_parr(Sobel c);

  int get_Size_Matrix();
  int get_Matrix(int i, int j);
  void set_Matrix(int i, int j, int val_gradient);
};
#endif  // MODULES_TASK_2_PICHUGIN_ILYA_SOBEL_OMP_OPERATOR_SOBEL_H_
