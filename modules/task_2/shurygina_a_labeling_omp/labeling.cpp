// Copyright 2022 Shurygina A

#include <omp.h>
#include <iostream>
#include <random>
#include <ctime>
#include <vector>
#include <algorithm>
#include <utility>
#include "../../../modules/task_2/shurygina_a_labeling_omp/labeling.h"

std::vector<int> getRandomImg(int rows, int cols) {
    if ((rows <= 0) || (cols <= 0))
        throw std::invalid_argument("Wrong input data");
    std::mt19937 gen;
    gen.seed(static_cast<unsigned int>(time(0)));
    std::vector<int> array(rows * cols);
    for (int i = 0; i < cols; i++)
        array[i] = 0;
    for (int i = 1; i < rows - 1; i++)
        for (int j = 0; j < cols; j++) {
            if ((j == 0) || (j == cols - 1))
                array[i * cols + j] = 0;
            else
                array[i * cols + j] = static_cast<int>(gen() % 2);
        }
    for (int i = 0; i < cols; i++)
        array[(rows - 1) * cols + i] = 0;
    return array;
}

std::pair<std::vector<int>, std::vector<int> >
          firstMarkOmp(std::vector<int> img, int rows, int cols) {
    int threads = 2;
    std::vector<int> sets(rows * cols);
    std::vector<int> strbeg(threads);
    int num = (rows - 2) / threads;
    int rem = (rows - 2) % threads;
    for (int i = 0; i < threads; i++) {
        strbeg[i] = i * num + 1;
    }
    std::vector<int> count(threads);
    for (int i = 0; i < threads; i++) {
        if (i == threads - 1)
            count[i] = num + rem;
        else
            count[i] = num;
    }
    for (int i = 0; i < rows * cols; i++)
        sets[i] = i;

#pragma omp parallel for shared(sets, img) num_threads(threads)
    for (int th = 0; th < threads; th++) {
        for (int i = strbeg[th]; i < strbeg[th] + count[th]; i++) {
            for (int j = 1; j < cols - 1; j++) {
                if (img[i * cols + j] == 0)
                    continue;
                if ((img[i * cols + j - 1] == 0)
                      && ((img[(i - 1) * cols + j] == 0)
                           || (i == strbeg[th]))) {
                    img[i * cols + j] = i * cols + j + 1;
                    continue;
                }
                if ((img[i * cols + j - 1] != 0)
                       && ((img[(i - 1) * cols + j] == 0)
                            || (i == strbeg[th]))) {
                    img[i * cols + j] = img[i * cols + j - 1];
                    continue;
                }
                if ((img[i * cols + j - 1] == 0)
                      && (img[(i - 1) * cols + j] != 0)) {
                    img[i * cols + j] = img[(i - 1) * cols + j];
                    continue;
                }
                if ((img[i * cols + j - 1] != 0)
                      && (img[(i - 1) * cols + j] != 0)) {
                    int max, min;
                    if (img[i * cols + j - 1] < img[(i - 1) * cols + j]) {
                        max = img[(i - 1) * cols + j];
                        min = img[i * cols + j - 1];
                    } else {
                        max = img[i * cols + j - 1];
                        min = img[(i - 1) * cols + j];
                    }
                    while (sets[max] != max) {
                        max = sets[max];
                    }
                    while (sets[min] != min)
                        min = sets[min];
                    if (min != max)
                        sets[max] = min;
                    img[i * cols + j] = min;
                }
            }
        }
    }
    for (int i = 1; i < threads; i++) {
        int str = strbeg[i];
        for (int j = 0; j < cols; j++) {
            if ((img[(str - 1) * cols + j] != 0)
                   && (img[str * cols + j] != 0)) {
                int first = sets[img[(str - 1) * cols + j]];
                int second = sets[img[str * cols + j]];
                if (first != second) {
                    int max;
                    if (first > second)
                        max = first;
                    else
                        max = second;
                    while (sets[max] != max)
                        max = sets[max];
                    int min;
                    if (first < second)
                        min = first;
                    else
                        min = second;
                    while (sets[min] != min)
                        min = sets[min];
                    if (max != min)
                        sets[max] = min;
                }
            }
        }
    }
    return { img, sets };
}

std::vector<int> secondMark(std::vector<int> arr, int rows, int cols,
                 std::vector<int> sets) {
    std::vector<int> res(rows * cols);
    for (int i = 0; i < rows * cols; i++)
        res[i] = arr[i];
    for (int i = cols; i < (rows - 1) * cols; i++) {
        if (arr[i] == 0)
            continue;
        if (sets[arr[i]] == arr[i]) {
            res[i] = arr[i];
        } else {
            while (sets[arr[i]] != arr[i])
                arr[i] = sets[arr[i]];
            res[i] = arr[i];
        }
    }
    return res;
}

std::vector<int> transformResult
                (const std::vector<int>& img, int rows, int cols) {
    std::vector<int> res(rows * cols, 0);
    std::vector<int> count;
    for (int i = 0; i < rows * cols; i++) {
        if (img[i] != 0) {
            bool flag = true;
            for (uint32_t j = 0; j < count.size(); j++)
                if (count[j] == img[i])
                    flag = false;
            if (flag)
                count.push_back(img[i]);
        }
    }
    for (int i = 0; i < rows * cols; i++) {
        if (img[i] != 0) {
            int flag = 0;
            for (uint32_t j = 0; j < count.size(); j++)
                if (count[j] == img[i])
                    flag = j + 1;
            res[i] = flag;
        }
    }
    return res;
}


std::vector<int> CLabelingOmp(const std::vector<int>& array,
             int rows, int cols) {
    std::vector<int> res(rows * cols);
    std::pair<std::vector<int>, std::vector<int> > firstm
            = firstMarkOmp(array, rows, cols);
    std::vector<int> arr = firstm.first;
    std::vector<int> sets = firstm.second;
    res = secondMark(arr, rows, cols, sets);
    return transformResult(res, rows, cols);
}
