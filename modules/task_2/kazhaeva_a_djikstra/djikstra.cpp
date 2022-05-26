// Copyright 2022 Kazhaeva Anastasia
#include "../../../modules/task_2/kazhaeva_a_djikstra/djikstra.h"

#include <omp.h>

#include <algorithm>
#include <random>
#include <utility>
#include <vector>

std::vector<std::vector<int>> randomVectorDjikstra(const int size) {
    std::vector<std::vector<int>> matrix(size, std::vector<int>(size));
    std::random_device dev;
    std::mt19937 gen(dev());
    for (int i = 0; i < size; ++i) {
        matrix[i][i] = 0;
        for (int j = i + 1; j < size; ++j) {
            matrix[i][j] = gen() % 100;
            matrix[j][i] = matrix[i][j];
        }
    }
    return matrix;
}

std::vector<int> djikstra(const int start,
const std::vector<std::vector<int>>& matrix) {
    const int INF = 1000000000;
    int const size = static_cast<int>(matrix.size());
    std::vector<int> d(size, INF);
    std::vector<int> v(size, 1);
    int temp, minindex, min;
    d[start] = 0;
    do {
        minindex = INF;
        min = INF;
        for (int i = 0; i < size; i++) {
            if ((v[i] == 1) && (d[i] < min)) {
                min = d[i];
                minindex = i;
            }
        }
        if (minindex != INF) {
            for (int i = 0; i < size; i++) {
                if (matrix[minindex][i] > 0) {
                    temp = min + matrix[minindex][i];
                    if (temp < d[i]) {
                        d[i] = temp;
                    }
                }
            }
            v[minindex] = 0;
        }
    } while (minindex < INF);


    return d;
}

std::vector<std::vector<int>> parallelDjikstra(int numThreads,
const std::vector<std::vector<int>>& matrix) {
    int count = 0;
    int const size = static_cast<int>(matrix.size());
    std::vector<std::vector<int>> result(size,
        std::vector<int>(size, 0));
#pragma omp parallel for shared(size, result, matrix) private(count) \
    schedule(static) num_threads(numThreads)
    for (count = 0; count < size; ++count) {
        std::vector<int> tmp = djikstra(count, matrix);
        for (int i = 0; i < size; ++i) {
            result[count][i] = tmp[i];
        }
    }
    return result;
}

std::vector<std::vector<int>> sequentialDjikstra(
const std::vector<std::vector<int>>& matrix) {
    int const size = static_cast<int>(matrix.size());
    std::vector<std::vector<int>> result(size,
        std::vector<int>(size, 0));
    std::vector<int> tmp(size);

    for (int count = 0; count < size; ++count) {
        tmp = djikstra(count, matrix);
        for (int i = 0; i < size; ++i) {
            result[count][i] = tmp[i];
        }
    }
    return result;
}
