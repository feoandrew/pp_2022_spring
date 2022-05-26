// Copyright 2022 Lakhov Kirill

#include "../../../modules/task_1/lakhov_k_trapezoidal_rule/trapezoidal_rule.h"
#include <vector>
#include <functional>

double _trapezoidalRule(std::function<double(std::vector<double>)> func,
                       const std::vector<double>& args, int position,
                       const std::vector<std::pair<double, double>>& intervals,
                       int interval_count) {
        double result = 0;
        int dimensions = args.size();
        double left = intervals[position].first;
        double right = intervals[position].second;
        double interval_size = (right - left)/interval_count;

        if (position == dimensions - 1) {
          double x = left;
          std::vector<double> current_args(args);
          std::vector<double> next_args(args);
          double interval_dimension = 1;
          for (int i = 0; i < dimensions; i++) {
              double interval = (intervals[i].second - intervals[i].first)
                                /interval_count;
              interval_dimension *= interval;
          }
          for (int i = 0; i < interval_count; i++) {
            current_args[position] = x;
            next_args[position] = x + interval_size;
            result += ((func(current_args) + func(next_args))/2)
                      *interval_dimension;
            x += interval_size;
          }
        } else {
          double x = left;
          std::vector<double> next_args(args);
          for (int i = 0; i < interval_count; i++) {
            next_args[position] = x + interval_size;
            result += _trapezoidalRule(func, next_args,
                                      position + 1, intervals, interval_count);
            x += interval_size;
          }
        }
        return result;
}

double trapezoidalRule(std::function<double(std::vector<double>)> func,
                       const std::vector<std::pair<double, double>>& intervals,
                       int interval_count) {
        int position = 0;
        int dimensions = intervals.size();
        std::vector<double> args(dimensions);
        for (int i = 0; i < dimensions; i++) {
          args[i] = intervals[i].first;
        }
        return _trapezoidalRule(func, args, position,
                                intervals, interval_count);
}
