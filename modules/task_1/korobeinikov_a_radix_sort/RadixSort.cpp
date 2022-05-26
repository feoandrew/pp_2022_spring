// Copyright 2022 Korobeinikov Alexandr
#include<vector>
#include <ctime>
#include <random>
#include "../../../modules/task_1/korobeinikov_a_radix_sort/RadixSort.h"

std::vector<double> CountingSort(std::vector<double> vec, size_t num_byte) {
    size_t cnt[256] = { 0 };
    std::vector<double> res(vec.size());
    int n = vec.size();

    unsigned char* byteArray = reinterpret_cast<unsigned char*>(vec.data());

    for (int i = 0; i < n; ++i) {
        ++cnt[byteArray[8 * i + num_byte]];
    }

    size_t a = 0;

    for (int j = 0; j < 256; ++j) {
        size_t b = cnt[j];
        cnt[j] = a;
        a += b;
    }

    for (int i = 0; i < n; ++i) {
        size_t dst = cnt[byteArray[8 * i + num_byte]]++;
        res[dst] = vec[i];
    }

    return res;
}

std::vector<double> RadixSort(std::vector<double> vec) {
    std::vector<double> tmp = vec;
    int n = tmp.size();
    for (int i = 0; i < 8; ++i) {
        tmp = CountingSort(tmp, i);
    }
    int count_negative = 0;
    for (double i : vec) {
        if (i < 0) count_negative++;
    }
    std::vector<double> tmp1(vec.size());
    for (int i = 0; i < count_negative; ++i) {
        tmp1[i] = tmp[n - i - 1];
    }
    for (int i = count_negative; i < n; ++i) {
        tmp1[i] = tmp[i - count_negative];
    }
    return tmp1;
}

std::vector<double>getRandomVector(int size, double lower_bound,
                                             double upper_bound, int seed) {
    std::mt19937 mt;
    if (seed == -1) {
        mt = std::mt19937(static_cast<int>(time(0)));
    } else {
        mt = std::mt19937(seed);
    }
    std::uniform_real_distribution<double> rand(-500.0, 500.0);
    std::vector<double> vec(size);
    for (int i = 0; i < size; i++) {
        vec[i] = rand(mt);
    }
    return vec;
}
