// Copyright 2022 Gordeev Viktor
#ifndef MODULES_TASK_1_GORDEEV_V_MULT_COMPLEX_MATRIX_MULT_COMPLEX_MATRIX_H_
#define MODULES_TASK_1_GORDEEV_V_MULT_COMPLEX_MATRIX_MULT_COMPLEX_MATRIX_H_


#include <vector>
#include <string>
#include <stdexcept>
#include <complex>

class MultMatrix {
 private:
    int i;
    int j;
    std::vector<int> rows;
    std::vector<int> columns;
    std::vector<std::complex<int>> mtxVector;

    std::vector<std::vector<std::complex<int>>> getRandomMatrix(int _i, int _j);
    std::vector<std::vector<std::complex<int>>> getEmptyMatrix(int _i, int _j);
 public:
    MultMatrix(int i, int j);
    explicit MultMatrix(std::vector<std::vector<std::complex<int>>> mtx);
    int getLength() const;
    int getHeight() const;
    std::complex<int> getElement(int row, int col) const;
    MultMatrix multMatrix(const MultMatrix &matrix);
};

#endif  // MODULES_TASK_1_GORDEEV_V_MULT_COMPLEX_MATRIX_MULT_COMPLEX_MATRIX_H_
