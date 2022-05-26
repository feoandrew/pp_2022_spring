// Copyright 2022 Lakhov Kirill
#ifndef MODULES_TASK_2_LAKHOV_K_TRAPEZOIDAL_RULE_TRAPEZOIDAL_RULE_H_
#define MODULES_TASK_2_LAKHOV_K_TRAPEZOIDAL_RULE_TRAPEZOIDAL_RULE_H_
#include <omp.h>
#include <vector>
#include <functional>
#include <utility>

double _trapezoidalRule(std::function<double(std::vector<double>)> func,
                       const std::vector<double>& args, int position,
                       const std::vector<std::pair<double, double>>& intervals,
                       std::vector<int> interval_count);

double trapezoidalRuleSeq(std::function<double(std::vector<double>)> func,
                       const std::vector<std::pair<double, double>>& intervals,
                       int interval_count);

double trapezoidalRuleParallel(std::function<double(std::vector<double>)> func,
                       const std::vector<std::pair<double, double>>& intervals,
                       int interval_count);

#endif  // MODULES_TASK_2_LAKHOV_K_TRAPEZOIDAL_RULE_TRAPEZOIDAL_RULE_H_
