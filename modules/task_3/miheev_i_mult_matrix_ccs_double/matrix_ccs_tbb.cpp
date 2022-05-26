// Copyright 2022 Miheev Ivan

#include "../../../modules/task_3/miheev_i_mult_matrix_ccs_double/matrix_ccs_tbb.h"

bool isZero(const double num) { return std::abs(num) < 0.00000001; }
bool isEqual(double x, double y) { return std::fabs(x - y) < 0.00000001; }
inline int Grain(int x) {
  if (x)
    return x;
  else
    return 1;
}

SprMatCCS::~SprMatCCS() {
  this->val.clear();
  this->rows.clear();
  this->ptr.clear();
}

SprMatCCS& SprMatCCS::randMat(int _dim, int spr) {
  std::random_device rd;
  std::mt19937 gen(rd());
  if (spr <= 0) {
    spr = 1;
  }

  this->clrMat();
  this->dim = _dim;
  this->cap = spr * _dim;
  this->val.resize(spr * _dim);
  this->rows.resize(spr * _dim);
  this->ptr.resize(_dim + 1);

  for (int i = 0; i < _dim; i++) {
    for (int j = 0; j < spr; j++) {
      bool b;
      do {
        this->rows[i * spr + j] = gen() % _dim + 1;
        b = false;
        for (int k = 0; k < j; k++) {
          if (this->rows[i * spr + j] == this->rows[i * spr + k]) {
            b = true;
          }
        }
      } while (b);
    }

    for (int k = 0; k < spr - 1; k++) {
      if (this->rows[i * spr + k] > this->rows[i * spr + k + 1]) {
        int tmp = this->rows[i * spr + k];
        this->rows[i * spr + k] = this->rows[i * spr + k + 1];
        this->rows[i * spr + k + 1] = tmp;
      }
    }
  }

  int rng = 100;  // upper bound of values
  for (int i = 0; i < spr * _dim; i++) {
    this->val[i] = gen() % rng;
  }

  int sum = 1;
  for (int i = 0; i < _dim + 1; i++) {
    this->ptr[i] = sum;
    sum += spr;
  }

  return *this;
}
SprMatCCS& SprMatCCS::clrMat() {
  this->cap = 0;
  this->dim = 0;
  this->val.clear();
  this->rows.clear();
  this->ptr.clear();

  return *this;
}
SprMatCCS SprMatCCS::transMat() {
  SprMatCCS res;
  res.dim = this->dim;
  res.cap = this->cap;
  res.val.resize(this->cap);
  res.rows.resize(this->cap);
  res.ptr.resize(this->dim + 1);

  for (int i = 0; i < this->cap; i++) {
    res.ptr[this->rows[i] - 1]++;
  }

  int tmp, sum = 1;
  for (int i = 0; i < this->dim + 1; i++) {
    tmp = res.ptr[i];
    res.ptr[i] = sum;
    sum += tmp;
  }
  std::vector<int> _ptr = res.ptr;
  for (int i = 0; i < this->dim; i++) {
    for (int j = this->ptr[i]; j < this->ptr[i + 1]; j++) {
      int r_ind = this->rows[j - 1];
      int i_ind = _ptr[r_ind - 1];
      res.rows[i_ind - 1] = i + 1;
      res.val[i_ind - 1] = this->val[j - 1];
      _ptr[r_ind - 1]++;
    }
  }
  return res;
}

SprMatCCS& SprMatCCS::operator=(const SprMatCCS& mat) {
  this->cap = mat.cap;
  this->dim = mat.dim;
  this->val = mat.val;
  this->rows = mat.rows;
  this->ptr = mat.ptr;

  return *this;
}
bool SprMatCCS::operator==(SprMatCCS mat) {
  if (this->dim != mat.dim || this->cap != mat.cap || this->rows != mat.rows ||
      this->ptr != mat.ptr)
    return false;
  for (int i = 0; i < this->cap; i++)
    if (!isEqual(val[i], mat.val[i])) return false;
  return true;
}
SprMatCCS SprMatCCS::operator*(SprMatCCS mat) {
  if (this->dim != mat.dim) throw "wrong sizes";
  SprMatCCS trans = this->transMat();

  std::vector<double> val_res;
  std::vector<int> rows_res;
  std::vector<int> ptr_res{1};

  int capacity = 1;
  double sum;
  for (int i = 0; i < mat.dim; i++) {
    std::vector<int> ip(trans.dim, 0);
    for (int j = mat.ptr[i]; j < mat.ptr[i + 1]; j++) {
      ip[mat.rows[j - 1] - 1] = j;
    }
    for (int j = 0; j < trans.dim; j++) {
      sum = 0;
      for (int k = trans.ptr[j]; k < trans.ptr[j + 1]; k++) {
        int p = ip[trans.rows[k - 1] - 1];
        if (p) {
          sum += mat.val[p - 1] * trans.val[k - 1];
        }
      }
      if (!isZero(sum)) {
        rows_res.push_back(j + 1);
        val_res.push_back(sum);
        capacity++;
      }
    }
    ptr_res.push_back(capacity);
  }
  SprMatCCS res(trans.dim, capacity - 1, val_res, rows_res, ptr_res);
  return res;
}

SprMatCCS SprMatCCS::ParallelMult(SprMatCCS mat) {
  if (this->dim != mat.dim) throw "wrong sizes";
  SprMatCCS trans = this->transMat();

  tbb::concurrent_vector<tbb::concurrent_vector<double>> val_tdd(trans.dim);
  tbb::concurrent_vector<tbb::concurrent_vector<int>> rows_tdd(trans.dim);
  tbb::concurrent_vector<int> ptr_counter(trans.dim);
  const int grainsize = mat.dim / 6;

  parallel_for(tbb::blocked_range<int>(0, mat.dim, Grain(grainsize)),
               [&](const tbb::blocked_range<int>& rg) {
                 for (int j = rg.begin(); j < rg.end(); j++) {
                   tbb::concurrent_vector<int> ip(trans.dim, 0);
                   int ctr = 0;
                   for (int i = mat.ptr[j]; i < mat.ptr[j + 1]; i++) {
                     ip[mat.rows[i - 1] - 1] = i;
                   }
                   for (int i = 0; i < trans.dim; i++) {
                     double sum = 0;
                     for (int k = trans.ptr[i]; k < trans.ptr[i + 1]; k++) {
                       int irow = trans.rows[k - 1];
                       int p = ip[irow - 1];
                       if (p) {
                         sum += mat.val[p - 1] * trans.val[k - 1];
                       }
                     }
                     if (!isZero(sum)) {
                       val_tdd[j].push_back(sum);
                       rows_tdd[j].push_back(i + 1);
                       ctr++;
                     }
                   }
                   ptr_counter[j] += ctr;
                 }
               });

  std::vector<double> val_res;
  std::vector<int> rows_res;
  std::vector<int> ptr_res{1};
  int sum = 1;
  for (int i = 0; i < trans.dim; i++) {
    sum += ptr_counter[i];
    ptr_res.push_back(sum);
    val_res.insert(val_res.end(), val_tdd[i].begin(), val_tdd[i].end());
    rows_res.insert(rows_res.end(), rows_tdd[i].begin(), rows_tdd[i].end());
  }
  SprMatCCS res(trans.dim, static_cast<int>(val_res.size()), val_res, rows_res,
                ptr_res);
  return res;
}

void SprMatCCS::shwVal() {
  for (int i = 0; i < this->cap; i++) {
    std::cout << val[i] << " ";
  }
  std::cout << "\n";
  for (int i = 0; i < this->cap; i++) {
    std::cout << rows[i] << " ";
  }
  std::cout << "\n";
  for (int i = 0; i < static_cast<int>(ptr.size()); i++) {
    std::cout << ptr[i] << " ";
  }
  std::cout << "\n";
}
