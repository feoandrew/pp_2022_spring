// Copyright 2022 Olynin Alexander
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
#include "../../modules/task_3/olynin_a_mult_sparse_cc_complex_mat/mult_sparse_cc_complex_mat.h"

std::vector<Complex> Complex::InitVec(std::vector<double> rls,
                                      std::vector<double> ims) {
    std::vector<Complex> Ent(rls.size());
    if (rls.size() == ims.size()) {
        for (size_t i = 0; i < rls.size(); i++) {
            Ent[i] = Complex(rls[i], ims[i]);
        }
    } else {
        for (size_t i = 0; i < rls.size(); i++) {
            Ent[i] = Complex(rls[i], 0);
        }
    }
    return Ent;
}
Complex& Complex::operator=(Complex Tmp) {
    this->rl = Tmp.rl;
    this->im = Tmp.im;
    return *this;
}
Complex Complex::operator*(Complex Tmp) {
    Complex Ans;
    Ans.rl = this->rl * Tmp.rl - this->im * Tmp.im;
    Ans.im = this->rl * Tmp.im + this->im * Tmp.rl;
    return Ans;
}
Complex Complex::operator+(Complex Tmp) {
    Complex Ans;
    Ans.rl = this->rl + Tmp.rl;
    Ans.im = this->im + Tmp.im;
    return Ans;
}
Complex& Complex::operator+=(Complex Tmp) {
    this->rl += Tmp.rl;
    this->im += Tmp.im;
    return *this;
}
bool Complex::operator==(Complex Tmp) {
    if ((this->rl - Tmp.rl < 0.00001) && (this->im - Tmp.im < 0.00001)) {
        return true;
    } else {
        return false;
    }
}
bool Complex::IsNotZero() {
    bool ans = false;
    const double ZeroLike = 0.000001;
    if ((fabs(this->rl) > ZeroLike) || (fabs(this->im) > ZeroLike)) {
        ans = true;
    }
    return ans;
}
Matrix& Matrix::operator=(const Matrix& Tmp) {
    this->size = Tmp.size;
    this->non = Tmp.non;
    this->Entry = Tmp.Entry;
    this->irows = Tmp.irows;
    this->shtcols = Tmp.shtcols;

    return *this;
}
Matrix& Matrix::ClearMatrix() {
    this->size = 0;
    this->non = 0;
    this->Entry.clear();
    this->irows.clear();
    this->shtcols.clear();

    return *this;
}
Matrix& Matrix::RandomMatrix(int size, int dist, int cnt, int seed) {
    this->ClearMatrix();
    this->size = size;
    std::mt19937 gen(time(0));
    gen.seed(seed);
    if (cnt < 0) {
        cnt = static_cast<int>(size * 0.01);
    }
    this->non = cnt * size;
    this->Entry.resize(cnt * size);
    this->irows.resize(cnt * size);
    this->shtcols.resize(size + 1);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < cnt; j++) {
            bool flag;
            do {
                this->irows[i * cnt + j] = gen() % size + 1;
                flag = false;
                for (int k = 0; k < j; k++) {
                    if (this->irows[i * cnt + j] == this->irows[i * cnt + k]) {
                        flag = true;
                    }
                }
            } while (flag);
        }
        for (int j = 0; j < cnt - 1; j++) {
            for (int k = 0; k < cnt - 1; k++) {
                if (this->irows[i * cnt + k] > this->irows[i * cnt + k + 1]) {
                    int tmp = this->irows[i * cnt + k];
                    this->irows[i * cnt + k] = this->irows[i * cnt + k + 1];
                    this->irows[i * cnt + k + 1] = tmp;
                }
            }
        }
    }
    for (int i = 0; i < cnt * size; i++) {
        this->Entry[i].SetRl(gen() % dist + 1);
        this->Entry[i].SetIm(gen() % dist + 1);
    }
    int sum = 1;
    for (int i = 0; i < size + 1; i++) {
        this->shtcols[i] = sum;
        sum += cnt;
    }

    return *this;
}
bool Matrix::operator==(Matrix Tmp) {
    bool ans = true;
    if (this->non != Tmp.non) {
        return false;
    }
    if (this->size != Tmp.size) {
        return false;
    }
    for (int i = 0; i < this->non; i++) {
        if (this->Entry[i] != Tmp.Entry[i]) {
            return false;
        }
    }
    if (this->irows != Tmp.irows) {
        return false;
    }
    if (this->shtcols != Tmp.shtcols) {
        return false;
    }
    return ans;
}
Matrix Matrix::T() {
    Matrix Ans;
    Ans.non = this->non;
    Ans.size = this->size;
    Ans.Entry.resize(this->non);
    Ans.irows.resize(this->non);
    Ans.shtcols.resize(this->size + 1);

    for (int i = 0; i < this->non; i++) {
        Ans.shtcols[this->irows[i] - 1]++;
    }
    int sum = 1;
    for (int i = 0; i < this->size + 1; i++) {
        int tmp = Ans.shtcols[i];
        Ans.shtcols[i] = sum;
        sum += tmp;
    }
    std::vector<int> shtcols_tmp = Ans.shtcols;
    for (int i = 0; i < this->size; i++) {
        for (int j = this->shtcols[i]; j < this->shtcols[i + 1]; j++) {
            int r_index = this->irows[j - 1];
            int i_index = shtcols_tmp[r_index - 1];
            Ans.irows[i_index - 1] = i + 1;
            Ans.Entry[i_index - 1] = this->Entry[j - 1];
            shtcols_tmp[r_index - 1]++;
        }
    }
    return Ans;
}
Matrix Matrix::operator^(Matrix B) {
    if (this->size != B.size) {
        return Matrix();
    }
    Matrix A = this->T();

    std::vector<Complex> EntRes;
    std::vector<int> irows_res;
    std::vector<int> shtcol_res = { 1 };

    int non_counter = 1;
    for (int j = 0; j < B.size; j++) {
        std::vector<int> ip(A.size, 0);
        for (int i = B.shtcols[j]; i < B.shtcols[j+1]; i++) {
            ip[B.irows[i-1]-1] = i;
        }
        for (int i = 0; i < A.size; i++) {
            Complex Sum;
            for (int k = A.shtcols[i]; k < A.shtcols[i+1]; k++) {
                int irow = A.irows[k-1];
                int p = ip[irow-1];
                if (p) {
                    Sum += B.Entry[p-1] * A.Entry[k-1];
                }
            }
            if (Sum.IsNotZero()) {
                    irows_res.push_back(i+1);
                    EntRes.push_back(Sum);
                    non_counter++;
            }
        }
        shtcol_res.push_back(non_counter);
    }
    Matrix Ans(A.size, non_counter-1, EntRes, irows_res, shtcol_res);
    return Ans;
}
Matrix Matrix::operator*(Matrix B) {
    if (this->size != B.size) {
        return Matrix();
    }
    Matrix A = this->T();

    std::vector<std::vector<Complex>> Entry_col(A.size);
    std::vector<std::vector<int>> irows_col(A.size);
    std::vector<int> counter(A.size);
    int grainsize = B.size / 5;
    if (grainsize == 0) {
        grainsize = 1;
    }
    parallel_for(tbb::blocked_range<int>(0, B.size, grainsize),
                     [&](const tbb::blocked_range<int>& range) {
        for (int j = range.begin(); j < range.end(); j++) {
            std::vector<int> ip(A.size, 0);
            int non_counter = 0;
            std::vector<Complex> Entry_thread;
            std::vector<int> irows_thread;
            for (int i = B.shtcols[j]; i < B.shtcols[j+1]; i++) {
                ip[B.irows[i-1]-1] = i;
            }
            for (int i = 0; i < A.size; i++) {
                Complex Sum;
                for (int k = A.shtcols[i]; k < A.shtcols[i+1]; k++) {
                    int irow = A.irows[k-1];
                    int p = ip[irow-1];
                    if (p) {
                        Sum += B.Entry[p-1] * A.Entry[k-1];
                    }
                }
                if (Sum.IsNotZero()) {
                    Entry_thread.push_back(Sum);
                    irows_thread.push_back(i+1);
                    non_counter++;
                }
            }
            counter[j] += non_counter;
            Entry_col[j] = Entry_thread;
            irows_col[j] = irows_thread;
        }
    });
    std::vector<Complex> EntryRes;
    std::vector<int> irowsres;
    std::vector<int> shtcolsres = { 1 };
    int sum = 1;
    for (int i = 0; i < A.size; i++) {
        sum += counter[i];
        shtcolsres.push_back(sum);
        EntryRes.insert(EntryRes.end(),
                        Entry_col[i].begin(),
                        Entry_col[i].end());
        irowsres.insert(irowsres.end(),
                        irows_col[i].begin(),
                        irows_col[i].end());
    }
    Matrix Ans(A.size, EntryRes.size(), EntryRes, irowsres, shtcolsres);
    return Ans;
}
Matrix::~Matrix() {
    this->Entry.clear();
    this->irows.clear();
    this->shtcols.clear();
}
