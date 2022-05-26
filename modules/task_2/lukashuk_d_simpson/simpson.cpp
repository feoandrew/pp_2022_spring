// Copyright 2022 Lukashuk Diana
#include "../../../modules/task_2/lukashuk_d_simpson/simpson.h"

typedef std::vector<double> VEC_T;

double simpson(const FUNC_T& f, const BORDER_T& borders,
               const std::vector<int>& steps) {
  int iter = 1;
  int dimen = static_cast<int>(steps.size());
  double integral = 0;
  int pow_6 = static_cast<int>(std::pow(6, dimen));
  VEC_T h(dimen);
  std::vector<VEC_T> param(dimen, VEC_T(6));
  VEC_T args(dimen);
  for (int i = 0; i < dimen; i++) {
    h[i] = (borders[i].second - borders[i].first) / steps[i];
    iter *= steps[i];
  }
  for (int i = 0, x = i; i < iter; i++, x = i) {
    for (int d = 0; d < dimen; d++) {
      param[d][0] = borders[d].first + (x % steps[d]) * h[d];
      param[d][1] = param[d][0] + h[d] / 2;
      param[d][4] = param[d][3] = param[d][2] = param[d][1];
      param[d][5] = param[d][0] + h[d];
      x /= steps[d];
    }
    for (int j = 0, x = j; j < pow_6; j++, x = j) {
      for (int d = 0; d < dimen; d++) {
        args[d] = param[d][x % 6];
        x /= 6;
      }
      integral += f(args);
    }
  }
  for (int i = 0; i < dimen; i++) {
    integral *= h[i] / 6.0;
  }
  return integral;
}

int x;
#pragma omp threadprivate(x)
double simpson_par(const FUNC_T& f, const BORDER_T& borders,
                   const std::vector<int>& steps) {
  int iter = 1;
  int dimen = static_cast<int>(steps.size());
  double integral = 0;
  int pow_6 = static_cast<int>(std::pow(6, dimen));
  VEC_T h(dimen);
  std::vector<VEC_T> param(dimen, VEC_T(6));
  std::vector<double> args(dimen);
  int i, j;
  for (i = 0; i < dimen; i++) {
    h[i] = (borders[i].second - borders[i].first) / steps[i];
    iter *= steps[i];
  }
// omp_set_num_threads(12);
#pragma omp parallel shared(dimen, iter, h, borders, steps, pow_6) \
        private(i, j) firstprivate(param, args) reduction(+ : integral)
#pragma omp for schedule(dynamic)
  for (i = 0; i < iter; i++) {
    x = i;
    for (int d = 0; d < dimen; d++) {
      param[d][0] = borders[d].first + (x % steps[d]) * h[d];
      param[d][1] = param[d][0] + h[d] / 2;
      param[d][4] = param[d][3] = param[d][2] = param[d][1];
      param[d][5] = param[d][0] + h[d];
      x /= steps[d];
    }
    for (j = 0; j < pow_6; j++) {
      x = j;
      for (int d = 0; d < dimen; d++) {
        args[d] = param[d][x % 6];
        x /= 6;
      }
      integral += f(args);
    }
  }
  for (int i = 0; i < dimen; i++) {
    integral *= h[i] / 6.0;
  }
  return integral;
}
