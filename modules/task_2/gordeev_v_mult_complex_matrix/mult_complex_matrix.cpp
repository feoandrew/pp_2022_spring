// Copyright 2022 Gordeev Viktor

#include "../../../modules/task_2/gordeev_v_mult_complex_matrix/mult_complex_matrix.h"

std::vector<std::vector<std::complex<int>>>
MultMatrix::getRandomMatrix(int _i, int _j) {
    std::random_device dev;
    std::mt19937 gen(dev());
    std::vector<std::vector<std::complex<int>>> result;
    for (int k = 0; k < _i; k++) {
        std::vector<std::complex<int>> tmp;
        for (int l = 0; l < _j; l++) {
            if (gen() % 1000 > 350) {
                std::complex<int> complexNumber(0, 0);
                tmp.push_back(complexNumber);
            } else {
                std::complex<int> complexNumber(gen() % 10, (gen() % 10));
                tmp.push_back(complexNumber);
            }
        }
        result.push_back(tmp);
    }
    return result;
}

std::vector<std::vector<std::complex<int>>>
MultMatrix::getEmptyMatrix(int _i, int _j) {
    std::vector<std::vector<std::complex<int>>> mtx;
    for (int k = 0; k < _i; k++) {
        mtx.push_back(std::vector<std::complex<int>>(_j));
    }
    return mtx;
}

MultMatrix::MultMatrix(int _i, int _j) {
    std::vector<std::vector<std::complex<int>>> mtx = getRandomMatrix(_i, _j);
    i = mtx.size();
    j = mtx[0].size();
    std::complex<int> zero(0, 0);
    rows.push_back(0);
    int counter = 0;
    for (int k = 0; k < static_cast<int>(mtx.size()); k++) {
        for (int l = 0; l < static_cast<int>(mtx[0].size()); l++) {
            if (mtx[k][l] != zero) {
                mtxVector.push_back(mtx[k][l]);
                columns.push_back(l);
                counter++;
            }
        }
        rows.push_back(counter);
    }
}

MultMatrix::MultMatrix(std::vector<std::vector<std::complex<int>>> mtx) {
    i = mtx.size();
    j = mtx[0].size();
    std::complex<int> zero(0, 0);
    rows.push_back(0);
    int counter = 0;
    for (int k = 0; k < static_cast<int>(mtx.size()); k++) {
        for (int l = 0; l < static_cast<int>(mtx[0].size()); l++) {
            if (mtx[k][l] != zero) {
                mtxVector.push_back(mtx[k][l]);
                columns.push_back(l);
                counter++;
            }
        }
        rows.push_back(counter);
    }
}

int MultMatrix::getLength() const {
    return i;
}

int MultMatrix::getHeight() const {
    return j;
}

std::complex<int> MultMatrix::getElement(int row, int column) const {
    if (row > i || column > j || row < 0 || column < 0) {
        throw -1;
    }
    int currentColumn;
    for (int k = rows[row]; k < rows[row + 1]; k++) {
        currentColumn = columns[k];
        if (currentColumn == column) {
            return mtxVector[k];
        }
    }
    return std::complex<int>();
}

MultMatrix MultMatrix::multMatrixSequential(const MultMatrix &mtx) {
    if (j != mtx.i) {
        throw -1;
    }
    std::complex<int> tmp;
    std::vector<std::vector<std::complex<int>>> res = getEmptyMatrix(i, mtx.j);
    for (int k = 0; k < i; k++) {
        for (int l = 0; l < mtx.j; l++) {
            tmp = std::complex<int>();
            for (int m = 0; m < j; m++) {
                tmp += getElement(k, m) *
                       mtx.getElement(m, l);
            }
            res[k][l] = tmp;
        }
    }
    return MultMatrix(res);
}

MultMatrix MultMatrix::multMatrixParallel(const MultMatrix &mtx) {
    if (j != mtx.i) {
        throw -1;
    }
    std::complex<int> tmp;
    std::vector<std::vector<std::complex<int>>> res = getEmptyMatrix(i, mtx.j);
#pragma omp parallel for shared(mtx, res) private(tmp)
    for (int k = 0; k < i; k++) {
        for (int l = 0; l < mtx.j; l++) {
            tmp = std::complex<int>();
            for (int m = 0; m < j; m++) {
                tmp += getElement(k, m) *
                       mtx.getElement(m, l);
            }
            res[k][l] = tmp;
        }
    }
    return MultMatrix(res);
}

bool MultMatrix::equal(const MultMatrix &mtx) {
    if (i != mtx.i)
        return false;
    if (j != mtx.j)
        return false;
    if (mtxVector != mtx.mtxVector)
        return false;
    return true;
}
