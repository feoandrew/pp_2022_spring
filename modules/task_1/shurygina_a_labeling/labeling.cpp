// Copyright 2022 Shurygina A

#include <iostream>
#include <random>
#include <ctime>
#include <vector>
#include "../../../modules/task_1/shurygina_a_labeling/labeling.h"

std::vector<int> getRandomImg(int rows, int cols) {
    if ((rows <= 0) || (cols <= 0))
        throw std::invalid_argument("Matrix can't be created");
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
        firstPass(std::vector<int> img, int rows, int cols) {
    int label = 0;
    std::vector<int> equiv_table;
    equiv_table.push_back(0);
    for (int i = 1; i < rows - 1; i++)
        for (int j = 1; j < cols - 1; j++) {
            if (img[i * cols + j] == 0)
                continue;
            if ((img[i * cols + j - 1] == 0)
                   && (img[(i - 1) * cols + j] == 0)) {
                label++;
                equiv_table.push_back(label);
                img[i * cols + j] = label;
                continue;
            }
            if ((img[i * cols + j - 1] != 0)
                    && (img[(i - 1) * cols + j] == 0)) {
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
                int tmp, tmpch;
                if (equiv_table[img[i * cols + j - 1]]
                        < equiv_table[img[(i - 1) * cols + j]]) {
                    tmp = equiv_table[img[i * cols + j - 1]];
                    tmpch = equiv_table[img[(i - 1) * cols + j]];
                } else {
                    tmp = equiv_table[img[(i - 1) * cols + j]];
                    tmpch = equiv_table[img[i * cols + j - 1]];
                }
                img[i * cols + j] = tmp;
                equiv_table[tmpch] = tmp;
                for (std::size_t k = 0; k < equiv_table.size(); k++)
                    if (equiv_table[k] == tmpch)
                        equiv_table[k] = tmp;
            }
        }
    return { img, equiv_table };
}

std::vector<int> secondPass(std::vector<int> img, int rows,
          int cols, std::vector<int> equiv_table) {
    for (int i = cols; i < (rows - 1) * cols; i++) {
        if (img[i] == 0)
            continue;
        img[i] = equiv_table[img[i]];
    }
    return img;
}

std::vector<int>
      labeling(const std::vector<int>& img, int rows, int cols) {
    std::vector<int> res(rows * cols);
    std::pair<std::vector<int>, std::vector<int> > firstm
         = firstPass(img, rows, cols);
    std::vector<int> my_img = firstm.first;
    std::vector<int> eq_tab = firstm.second;
    res = secondPass(my_img, rows, cols, eq_tab);
    return res;
}
