// Copyright 2022 Kazhaeva Anastasia
#include "../../../modules/task_1/kazhaeva_a_djikstra/djikstra_seq.h"

std::vector<int> djikstra(const int current,
const std::vector<std::vector<int>>& matrix) {
    const int INF = 1000000;
    int const size = static_cast<int>(matrix.size());
    std::vector<int> d(size, INF);
    std::vector<int> v(size, 1);
    int temp, minindex, min;
    d[current] = 0;
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
            for (int i = 0; i < size; i++)
                if (matrix[minindex][i] > 0) {
                    temp = min + matrix[minindex][i];
                    if (temp < d[i])
                        d[i] = temp;
                }
            v[minindex] = 0;
        }
    } while (minindex < INF);

    return d;
}
