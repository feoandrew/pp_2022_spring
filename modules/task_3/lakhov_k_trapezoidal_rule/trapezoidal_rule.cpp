// Copyright 2022 Lakhov Kirill
#include "../../../modules/task_3/lakhov_k_trapezoidal_rule/trapezoidal_rule.h"

#include <tbb/tbb.h>

#include <vector>
#include <iostream>
#include <functional>

typedef std::vector<std::vector<std::pair<double, double>>> integral_intervals;

double _trapezoidalRule(std::function<double(std::vector<double>)> func,
                       const std::vector<double>& args, int position,
                       const std::vector<std::pair<double, double>>& intervals,
                       std::vector<int> interval_count) {
        double result = 0;
        int dimensions = intervals.size();
        double left = intervals[position].first;
        double right = intervals[position].second;
        double interval_size = (right - left)/interval_count[position];

        if (position == dimensions - 1) {
          double x = left;
          std::vector<double> current_args(args);
          std::vector<double> next_args(args);
          double interval_dimension = 1;
          for (int i = 0; i < dimensions; i++) {
              double interval = (intervals[i].second - intervals[i].first)
                                /interval_count[i];
              interval_dimension *= interval;
          }
          for (int i = 0; i < interval_count[position]; i++) {
            current_args[position] = x;
            next_args[position] = x + interval_size;
            result += ((func(current_args) + func(next_args))/2)
                      *interval_dimension;
            x += interval_size;
          }
        } else {
          double x = left;
          std::vector<double> next_args(args);
          for (int i = 0; i < interval_count[position]; i++) {
            next_args[position] = x + interval_size;
            result += _trapezoidalRule(func, next_args,
                                      position + 1, intervals, interval_count);
            x += interval_size;
          }
        }
        return result;
}

double trapezoidalRuleSeq(std::function<double(std::vector<double>)> func,
                       const std::vector<std::pair<double, double>>& intervals,
                       int interval_count) {
        int position = 0;
        int dimensions = intervals.size();
        std::vector<double> args(dimensions);
        std::vector<int> interval_count_v(dimensions);
        for (int i = 0; i < dimensions; i++) {
          args[i] = intervals[i].first;
          interval_count_v[i] = interval_count;
        }
        return _trapezoidalRule(func, args, position,
                                intervals, interval_count_v);
}

double trapezoidalRuleParallel(std::function<double(std::vector<double>)> func,
                       const std::vector<std::pair<double, double>>& intervals,
                       int interval_count) {
    int threads = tbb::task_scheduler_init::default_num_threads();
    int position = 0;
    int dimensions = intervals.size();
    std::vector<double> args(dimensions);
    std::vector<int> interval_count_v(dimensions);

    for (int i = 0; i < dimensions; i++) {
         args[i] = intervals[i].first;
         interval_count_v[i] = i == 0 ?
                                      (interval_count / threads) :
                                       interval_count;
      }

      integral_intervals intervals_to_thread(threads);
      double fist_dimension_interval =
                          (intervals[0].second - intervals[0].first) / threads;
      double current_x = intervals[0].first;
      for (int i = 0; i < threads; i++) {
        std::vector<std::pair<double, double>> current_intervals(intervals);
        current_intervals[0].first = current_x;
        current_x += fist_dimension_interval;
        current_intervals[0].second = current_x;
        intervals_to_thread[i] = current_intervals;
    }

  double result = tbb::parallel_reduce(
    tbb::blocked_range<integral_intervals::const_iterator>
                    (intervals_to_thread.begin(), intervals_to_thread.end()),
    0.0,
    [&](tbb::blocked_range<integral_intervals::const_iterator> range,
        double result) {
        auto task = range.begin();
        while (task != range.end()) {
          result += _trapezoidalRule(func, args,
                                     position, (*task), interval_count_v);
          ++task;
        }
        return result;
      }, std::plus<double>());
  return result;
}
