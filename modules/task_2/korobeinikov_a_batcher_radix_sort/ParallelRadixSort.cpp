// Copyright 2022 Korobeinikov Alexandr
#include <omp.h>
#include <limits.h>
#include <algorithm>
#include<vector>
#include <ctime>
#include <random>
#include <utility>
#include "../../../modules/task_2/korobeinikov_a_batcher_radix_sort/ParallelRadixSort.h"

const int inf = INT_MAX;

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

void RadixSort(int left, int right, std::vector<double>* vec) {
    std::vector<double> tmp(right - left + 1);
    int n = tmp.size();
    for (int i = 0; i < n; ++i) {
        tmp[i] = vec->at(left + i);
    }
    for (int i = 0; i < 8; ++i) {
        tmp = CountingSort(tmp, i);
    }
    int count_negative = 0;
    for (double i : tmp) {
        if (i < 0) count_negative++;
    }
    std::vector<double> tmp2(tmp.size());
    for (int i = 0; i < count_negative; ++i) {
        tmp2[i] = tmp[n - i - 1];
    }
    for (int i = count_negative; i < n; ++i) {
        tmp2[i] = tmp[i - count_negative];
    }
    for (int i = left; i <= right; ++i) {
        vec->at(i) = tmp2[i - left];
    }
}

void EvenSplitter(int start, size_t size1, size_t size2,
    std::vector <double>* arr) {
    std::vector <double> tmp(size1);
    for (size_t i = 0; i < size1; i += 2) {
        tmp[i] = arr->at(start + i);
    }
    size2 = start + size1 + size2;
    size_t first_iter = 0;
    size_t second_iter = start + size1;
    size_t common_iter = start;
    while (first_iter < size1 && second_iter < size2) {
        if (tmp[first_iter] < arr->at(second_iter)) {
            arr->at(common_iter) = tmp[first_iter];
            first_iter += 2;
        } else {
            arr->at(common_iter) = arr->at(second_iter);
            second_iter += 2;
        }
        common_iter += 2;
    }
    if (first_iter >= size1) {
        for (size_t i = second_iter; i < size2; i += 2, common_iter += 2) {
            arr->at(common_iter) = arr->at(i);
        }
    } else {
        for (size_t i = first_iter; i < size1; i += 2, common_iter += 2) {
            arr->at(common_iter) = tmp[i];
        }
    }
}

void OddSplitter(int start, size_t size1, size_t size2,
    std::vector <double>* arr) {
    std::vector <double> tmp(size1);
    for (size_t i = 1; i < size1; i += 2) {
        tmp[i] = arr->at(start + i);
    }
    size2 = start + size1 + size2;
    size_t first_iter = 1;
    size_t second_iter = start + size1 + 1;
    size_t common_iter = start + 1;
    while (first_iter < size1 && second_iter < size2) {
        if (tmp[first_iter] < arr->at(second_iter)) {
            arr->at(common_iter) = tmp[first_iter];
            first_iter += 2;
        } else {
            arr->at(common_iter) = arr->at(second_iter);
            second_iter += 2;
        }
        common_iter += 2;
    }
    if (first_iter >= size1) {
        for (size_t i = second_iter; i < size2; i += 2, common_iter += 2) {
            arr->at(common_iter) = arr->at(i);
        }
    } else {
        for (size_t i = first_iter; i < size1; i += 2, common_iter += 2) {
            arr->at(common_iter) = tmp[i];
        }
    }
}

void RadixSortParallel(std::vector<double>* arr, int th) {
    int n = arr->size();
    int num = th;
    omp_set_num_threads(num);
    int tmp = n / num;
    int a = tmp + tmp % 2;
    int tmp2 = n - (num - 1) * a;
    int b = tmp2 + tmp2 % 2;
    int delta = (num - 1) * a + b - n;
    for (int i = 0; i < delta; ++i) {
        arr->push_back(inf);
    }
#pragma omp parallel
    {
#pragma omp for
        for (int i = 0; i < num; ++i) {
            int left = i * a;
            int right = (i + 1 == num) ? n + delta - 1 : left + a - 1;
            RadixSort(left, right, arr);
        }
    }
    int count_m = num / 2;
    int k = 1;
    int last = 0;
    while (count_m > 0) {
#pragma omp parallel
        {
            int thread = omp_get_thread_num();
            if (thread < 2 * count_m) {
                if (thread % 2 == 0) {
                    EvenSplitter(thread * a * k, a * k,
                        (thread == num - 2 ? a * last + b : a * k), arr);
                } else {
                    OddSplitter((thread - 1) * a * k, a * k,
                        (thread == num - 1 ? a * last + b : a * k), arr);
                }
            }
        }
#pragma omp parallel
        {
            int thread = omp_get_thread_num();
            if (thread < 2 * count_m) {
                if (thread % 2 == 0) {
                    int start = thread * a * k + 1;
                    int fin = (thread + 1 == num - 1 ? arr->size() :
                        (thread + 2) * a * k);
                    for (int i = start; i < fin - 1; i += 4) {
                        if (arr->at(i) > arr->at(i + 1)) {
                            std::swap(arr->at(i), arr->at(i + 1));
                        }
                    }
                } else {
                    int start = (thread - 1) * a * k + 3;
                    int fin = (thread == num - 1 ? arr->size() :
                        (thread + 1) * a * k);
                    for (int i = start; i < fin - 1; i += 4) {
                        if (arr->at(i) > arr->at(i + 1)) {
                            std::swap(arr->at(i), arr->at(i + 1));
                        }
                    }
                }
            }
        }
        if (num % 2 == 0) {
            last += k;
        }
        num -= count_m;
        count_m = num / 2;
        k *= 2;
    }
    for (int i = 0; i < delta; ++i) {
        arr->pop_back();
    }
}
