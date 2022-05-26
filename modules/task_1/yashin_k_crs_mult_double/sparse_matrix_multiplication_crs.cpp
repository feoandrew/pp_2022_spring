// Copyright 2022 Yashin Kirill
#include <vector>

#include "../../modules/task_1/yashin_k_crs_mult_double/sparse_matrix_multiplication_crs.h"

bool sparse_matrix::operator== (const sparse_matrix& matrix) const& {
    if (rows != matrix.rows || col_index != matrix.col_index || columns != matrix.columns
      || row_index != matrix.row_index || values.size() != matrix.values.size())
        return false;

    for (size_t i = 0; i < values.size(); i++)
        if ((std::fabs(matrix.values[i] - values[i]) > 0.00001))
            return false;
    return true;
}

Matrix sparse_matrix::sparce_matrix_to_default() {
    Matrix result(rows, std::vector<double>(columns, 0.0));

    int temp_column = 0;
    for (int i = 0; i < rows; i++) {
        int temp_row = row_index[i + 1] - row_index[i];
        while (temp_row) {
            result[i][col_index[temp_column]] = values[temp_column];
            temp_row -= 1;
            temp_column += 1;
        }
    }
    return result;
}

sparse_matrix sparse_multiplication(const sparse_matrix& A, const sparse_matrix& B) {
    sparse_matrix result;
    result.rows = A.rows;
    result.columns = B.columns;
    result.row_index.push_back(0);
    std::vector<double> temp_result_row(B.columns, 0);

    for (int i = 0; i < A.rows; i++) {
        for (int j = A.row_index[i]; j < A.row_index[i + 1]; j++) {
            int temp_column {A.col_index[j]};
            for (int k = B.row_index[temp_column]; k < B.row_index[temp_column + 1]; k++)
                temp_result_row[B.col_index[k]] += A.values[j] * B.values[k];
        }
        for (int k = 0; k < B.columns; k++) {
            if (temp_result_row[k]) {
                result.values.push_back(temp_result_row[k]);
                result.col_index.push_back(k);
                temp_result_row[k] = 0;
            }
        }
        result.row_index.push_back(result.values.size());
    }
    return result;
}

Matrix matrix_multiplication(const Matrix& A, const Matrix& B) {
    Matrix result(A.size());
    for (size_t i = 0; i < result.size(); i++)
        result[i].resize(B[0].size());

    for (size_t i = 0; i < A.size(); i++) {
        for (size_t j = 0; j < B[0].size(); j++) {
            result[i][j] = 0;
            for (size_t k = 0; k < A[0].size(); k++)
                result[i][j] += A[i][k] * B[k][j];
        }
    }
    return result;
}

Matrix random_matrix(const int& rows, const int& columns) {
    std::random_device rand{};
    std::mt19937 mt {rand()};
    std::uniform_real_distribution<double> rand_value {0.0, 10.0};
    std::uniform_int_distribution<int> rand_probability {0, 100};
    Matrix result{};
    result.resize(rows);
    for (int i = 0; i < rows; i++)
        result[i].resize(columns);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            if (rand_probability(mt) <= 10)
                result[i][j] = rand_value(mt);
        }
    }
    return result;
}
