// Copyright 2022 Olynin Alexander
#ifndef MODULES_TASK_2_OLYNIN_A_MULT_SPARSE_CC_COMPLEX_MAT_MULT_SPARSE_CC_COMPLEX_MAT_H_
#define MODULES_TASK_2_OLYNIN_A_MULT_SPARSE_CC_COMPLEX_MAT_MULT_SPARSE_CC_COMPLEX_MAT_H_

#include <time.h>
#include <random>
#include <vector>

class Complex {
 private:
    double rl;
    double im;
 public:
    explicit Complex(double _rl = 0, double _im = 0): rl(_rl), im(_im) {}
    Complex(const Complex& Tmp): rl(Tmp.rl), im(Tmp.im) {}
    Complex& operator=(Complex Tmp);
    std::vector<Complex> InitVec(std::vector<double> rls =
                                 std::vector<double>(),
                                 std::vector<double> ims =
                                 std::vector<double>());
    double GetRl() { return this->rl; }
    double GetIm() { return this->im; }
    void SetRl(double tmp) { this->rl = tmp; }
    void SetIm(double tmp) { this->im = tmp; }
    Complex operator+(Complex Tmp);
    Complex& operator+=(Complex Tmp);
    Complex operator*(Complex Tmp);
    bool operator==(Complex Tmp);
    bool operator!=(Complex Tmp) { return !(*this == Tmp); }
    bool IsNotZero();
    ~Complex() {}
};

class Matrix {
 private:
    int size;
    int non;
    std::vector<Complex> Entry;
    std::vector<int> irows;
    std::vector<int> shtcols;

 public:
    Matrix(int _size = 0, int _non = 0, const std::vector<Complex>& _Entry =
           std::vector<Complex>(), const std::vector<int>& _irows =
           std::vector<int>(), const std::vector<int>& _shtcols =
           std::vector<int>()): size(_size), non(_non), Entry(_Entry),
           irows(_irows), shtcols(_shtcols) {}
    Matrix(const Matrix& Tmp): size(Tmp.size), non(Tmp.non), Entry(Tmp.Entry),
                               irows(Tmp.irows), shtcols(Tmp.shtcols) {}
    Matrix& operator=(const Matrix& Tmp);
    int GetSize() { return this->size; }
    int GetNon() { return this-> non; }
    std::vector<Complex> GetEntry() { return this->Entry; }
    std::vector<int> GetIrows() { return this->irows; }
    std::vector<int> GetShtcols() { return this->shtcols; }
    Matrix& ClearMatrix();
    Matrix& RandomMatrix(int size, int dist, int cnt = -1, int seed = 0);
    bool operator==(Matrix Tmp);
    bool operator!=(const Matrix& Tmp) { return !(*this == Tmp); }
    Matrix T();
    Matrix operator^(Matrix B);
    Matrix operator*(Matrix B);
    ~Matrix();
};

#endif  // MODULES_TASK_2_OLYNIN_A_MULT_SPARSE_CC_COMPLEX_MAT_MULT_SPARSE_CC_COMPLEX_MAT_H_
