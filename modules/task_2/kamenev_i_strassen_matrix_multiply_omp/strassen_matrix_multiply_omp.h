// Copyright 2022 Kamenev Ilya
#ifndef MODULES_TASK_2_KAMENEV_I_STRASSEN_MATRIX_MULTIPLY_OMP_STRASSEN_MATRIX_MULTIPLY_OMP_H_
#define MODULES_TASK_2_KAMENEV_I_STRASSEN_MATRIX_MULTIPLY_OMP_STRASSEN_MATRIX_MULTIPLY_OMP_H_

void strassen_omp(double* a, double* b, double* c, int size);
void naive_mult(double* a, double* b, double* c, int size);
bool is_exp_of_2(int n);
void get_p1(double* a11, double* a22, double* b11, double* b22, double* slice,
            int size);
void get_p2(double* a21, double* a22, double* b11, double* slice, int size);
void get_p3(double* a11, double* b12, double* b22, double* slice, int size);
void get_p4(double* a22, double* b21, double* b11, double* slice, int size);
void get_p5(double* a11, double* a12, double* b22, double* slice, int size);
void get_p6(double* a21, double* a11, double* b11, double* b12, double* slice,
            int size);
void get_p7(double* a12, double* a22, double* b21, double* b22, double* slice,
            int size);

#endif  // MODULES_TASK_2_KAMENEV_I_STRASSEN_MATRIX_MULTIPLY_OMP_STRASSEN_MATRIX_MULTIPLY_OMP_H_
