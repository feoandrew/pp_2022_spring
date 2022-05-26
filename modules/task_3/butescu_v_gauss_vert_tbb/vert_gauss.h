//  Copyright 2022 Butescu Vladimir
#ifndef MODULES_TASK_3_BUTESCU_V_GAUSS_VERT_TBB_VERT_GAUSS_H_
#define MODULES_TASK_3_BUTESCU_V_GAUSS_VERT_TBB_VERT_GAUSS_H_

#include <tbb/tick_count.h>
#include <tbb/parallel_for.h>
#include <vector>
#include <random>
#include <ctime>



int clamp(int value, int max, int min);
std::vector<double> createRandomMatrix(int rows, int cols);
std::vector<double> gauss_filter_sequence(const std::vector<double>& matrix,
    int rows, int cols);
std::vector<double> gauss_filter_parralel_tbb(const std::vector<double>& matrix,
    int rows, int cols);

#endif  // MODULES_TASK_3_BUTESCU_V_GAUSS_VERT_TBB_VERT_GAUSS_H_
