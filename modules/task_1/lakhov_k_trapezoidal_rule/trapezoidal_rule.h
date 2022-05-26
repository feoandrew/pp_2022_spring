// Copyright 2022 Lakhov Kirill
#ifndef MODULES_TASK_1_LAKHOV_K_TRAPEZOIDAL_RULE_TRAPEZOIDAL_RULE_H_
#define MODULES_TASK_1_LAKHOV_K_TRAPEZOIDAL_RULE_TRAPEZOIDAL_RULE_H_
#include <vector>
#include <functional>
#include <utility>

double _trapezoidalRule(std::function<double(std::vector<double>)> func,
                       const std::vector<double>& args, int position,
                       const std::vector<std::pair<double, double>>& intervals,
                       int interval_count);

double trapezoidalRule(std::function<double(std::vector<double>)> func,
                       const std::vector<std::pair<double, double>>& intervals,
                       int interval_count);

#endif  // MODULES_TASK_1_LAKHOV_K_TRAPEZOIDAL_RULE_TRAPEZOIDAL_RULE_H_
