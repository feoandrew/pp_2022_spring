// Copyright 2022 Korobeinikov Alexandr

#include <limits.h>
#include <tbb/tbb.h>
#include <tbb/task.h>
#include <tbb/blocked_range.h>
#include <random>
#include <ctime>
#include <algorithm>
#include <vector>
#include <utility>
#include "../../../modules/task_3/korobeinikov_a_radix_batcher/batcher_tbb.h"

tbb::task* EvenSplitter::execute() {
    for (int i = 0; i < size1; i += 2)
        tmp[i] = mas[i];
    double* mas2 = mas + size1;
    int a = 0;
    int b = 0;
    int i = 0;
    while ((a < size1) && (b < size2)) {
        if (tmp[a] <= mas2[b]) {
            mas[i] = tmp[a];
            a += 2;
        } else {
            mas[i] = mas2[b];
            b += 2;
        }
        i += 2;
    }
    if (a == size1) {
        for (int j = b; j < size2; j += 2, i += 2)
            mas[i] = mas2[j];
    } else {
        for (int j = a; j < size1; j += 2, i += 2)
            mas[i] = tmp[j];
    }
    return NULL;
}
tbb::task* OddSplitter::execute() {
    for (int i = 1; i < size1; i += 2)
        tmp[i] = mas[i];
    double* mas2 = mas + size1;
    int a = 1;
    int b = 1;
    int i = 1;
    while ((a < size1) && (b < size2)) {
        if (tmp[a] <= mas2[b]) {
            mas[i] = tmp[a];
            a += 2;
        } else {
            mas[i] = mas2[b];
            b += 2;
        }
        i += 2;
    }
    if (a == size1) {
        for (int j = b; j < size2; j += 2, i += 2)
            mas[i] = mas2[j];
    } else {
        for (int j = a; j < size1; j += 2, i += 2)
            mas[i] = tmp[j];
    }
    return NULL;
}

void SimpleComparator::operator()(const tbb::blocked_range<int>& r) const {
    int begin = r.begin(), end = r.end();
    for (int i = begin; i < end; i++)
        if (mas[2 * i] < mas[2 * i - 1]) {
            double _tmp = mas[2 * i - 1];
            mas[2 * i - 1] = mas[2 * i];
            mas[2 * i] = _tmp;
        }
}

tbb::task* RadixParallelSorter::execute() {
    if (size <= portion) {
        RadixSort(mas, 0, size - 1);
    } else {
        int s = size / 2 + (size / 2) % 2;
        RadixParallelSorter& sorter1 = *new(allocate_child())
            RadixParallelSorter(mas, tmp, s, portion);
        RadixParallelSorter& sorter2 = *new(allocate_child())
            RadixParallelSorter(mas + s, tmp + s, size - s,
                portion);
        set_ref_count(3);
        spawn(sorter1);
        spawn_and_wait_for_all(sorter2);
        // std::cout << s << " " << size - s << '\n';
        EvenSplitter& splitter1 = *new(allocate_child())
            EvenSplitter(mas, tmp, s, size - s);
        OddSplitter& splitter2 = *new(allocate_child())
            OddSplitter(mas, tmp, s, size - s);
        set_ref_count(3);
        spawn(splitter1);
        spawn_and_wait_for_all(splitter2);

        tbb::parallel_for(tbb::blocked_range<int>(1, (size + 1) / 2),
            SimpleComparator(mas));
    }
    return NULL;
}
void TbbParallelSort(double* inp, int size, int nThreads) {
    double* tmp = new double[size];
    int portion = size / nThreads;
    if (size % nThreads != 0)
        portion++;
    // std::cout << portion << '\n';
    RadixParallelSorter& sorter = *new(tbb::task::allocate_root())
        RadixParallelSorter(inp, tmp, size, portion);
    tbb::task::spawn_root_and_wait(sorter);
    delete[] tmp;
}

std::vector<double> CountingSort(std::vector<double> vec, size_t num_byte) {
    size_t cnt[256] = { 0 };
    std::vector<double> res(vec.size());
    size_t n = vec.size();

    unsigned char* byteArray = reinterpret_cast<unsigned char*>(vec.data());

    for (size_t i = 0; i < n; ++i) {
        ++cnt[byteArray[8 * i + num_byte]];
    }

    size_t a = 0;

    for (int j = 0; j < 256; ++j) {
        size_t b = cnt[j];
        cnt[j] = a;
        a += b;
    }

    for (size_t i = 0; i < n; ++i) {
        size_t dst = cnt[byteArray[8 * i + num_byte]]++;
        res[dst] = vec[i];
    }

    return res;
}

void RadixSort(double* vec, int left, int right) {
    std::vector<double> tmp(right - left + 1);
    size_t n = tmp.size();
    for (size_t i = 0; i < n; ++i) {
        tmp[i] = vec[left + i];
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
    for (size_t i = count_negative; i < n; ++i) {
        tmp2[i] = tmp[i - count_negative];
    }
    for (int i = left; i <= right; ++i) {
        vec[i] = tmp2[i - left];
    }
}

void getRandomArray(double* arr, int size) {
    std::mt19937 gen(time(0));
    std::uniform_int_distribution <int> dist(-1000, 1000);
    for (int i = 0; i < size; ++i) {
        arr[i] = dist(gen);
    }
    return;
}

bool checkCorrectnessOfSort(double* arr, int size) {
    return std::is_sorted(arr, arr + size);
}
