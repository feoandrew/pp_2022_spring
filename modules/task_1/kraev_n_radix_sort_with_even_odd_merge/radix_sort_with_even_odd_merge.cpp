// Copyright 2022 Kraev Nikita
#include "../../../modules/task_1/kraev_n_radix_sort_with_even_odd_merge/radix_sort_with_even_odd_merge.h"

std::vector<double> getRandomVector(int size) {
    std::vector<double> random_vec(size);

    std::random_device rd;
    std::default_random_engine generator(rd());
    double a = -5000.0, b = 5000.0;
    std::uniform_real_distribution<double> dis(a, b);

    for (int i = 0; i < size; ++i) {
        random_vec[i] = dis(generator);
    }

    return random_vec;
}

void radixSort(std::vector<double>* vec) {
    size_t size = vec->size();
    size_t local_size;
    int tmp;
    int j = 0;
    std::vector<std::vector<double>> counters(256);
    for (int index = 0; index < 7; index++) {
        for (size_t i = 0; i < size; i++) {
            tmp = static_cast<int>(*((unsigned char*)&(*vec)[i] + index));
            counters[tmp].push_back((*vec)[i]);
        }

        for (int i = 0; i < 256; i++) {
            if (!counters[i].empty()) {
                local_size = counters[i].size();
                for (size_t k = 0; k < local_size; k++) {
                    (*vec)[j] = counters[i][k];
                    ++j;
                }
                counters[i].clear();
            }
        }
        j = 0;
    }

    for (size_t i = 0; i < size; i++) {
        tmp = static_cast<int>(*((unsigned char*)&(*vec)[i] + 7));
        counters[tmp].push_back((*vec)[i]);
    }

    for (int i = 255; i >= 128; i--) {
        if (!counters[i].empty()) {
            local_size = counters[i].size();
            for (size_t k = local_size - 1; k >= 1; k--) {
                (*vec)[j] = counters[i][k];
                j++;
            }
            (*vec)[j] = counters[i][0];
            j++;
        }
    }

    for (int i = 0; i < 128; i++) {
        if (!counters[i].empty()) {
            local_size = counters[i].size();
            for (size_t k = 0; k < local_size; k++) {
                (*vec)[j] = counters[i][k];
                j++;
            }
        }
    }
}
