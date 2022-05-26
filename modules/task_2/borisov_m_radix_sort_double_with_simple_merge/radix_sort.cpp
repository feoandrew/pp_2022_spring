// Copyright 2022 Borisov Maxim

#include "../../../modules/task_2/borisov_m_radix_sort_double_with_simple_merge/radix_sort.h"

std::vector<double> random_vec(int size) {
  std::vector<double> data(size);
  std::mt19937 gen;
  gen.seed(static_cast<uint32_t>(time(0)));
  std::uniform_real_distribution<double> buf(1, 1000);
  std::transform(data.begin(), data.end(), data.begin(),
                 [&](double x) { return buf(gen); });
  return data;
}

std::vector<double> merge(const std::vector<double>& arr1,
                          const std::vector<double>& arr2) {
  int len1 = arr1.size();
  int len2 = arr2.size();
  std::vector<double> out;
  out.resize(len1 + len2);
  int x = 0, y = 0;
  for (int i = 0; i < len1 + len2; i++) {
    if (x >= len1) {
      out[i] = arr2[y++];
    } else if (y >= len2) {
      out[i] = arr1[x++];
    } else if (arr1[x] < arr2[y]) {
      out[i] = arr1[x++];
    } else {
      out[i] = arr2[y++];
    }
  }
  return out;
}
void count_sort(double* in, double* out, int len, int exp) {
  unsigned char* buf = reinterpret_cast<unsigned char*>(in);
  int count[257] = {0};
  int val = 0;
  for (int i = 0; i < len; i++) {
    count[buf[8 * i + exp]]++;
  }
  int j = 0;
  for (; j < 256; j++) {
    if (count[j] != 0) break;
  }
  val = count[j];
  count[j] = 0;
  j++;
  for (; j < 256; j++) {
    int tmp = count[j];
    count[j] = val;
    val += tmp;
  }
  for (int i = 0; i < len; i++) {
    out[count[buf[8 * i + exp]]] = in[i];
    count[buf[8 * i + exp]]++;
  }
}

std::vector<double> radix_sort_seq(const std::vector<double>& data1) {
  int len1 = static_cast<int>(data1.size());

  std::vector<double> in1 = data1;
  std::vector<double> out1(data1.size());

  for (int i = 0; i < 4; i++) {
    count_sort(in1.data(), out1.data(), len1, 2 * i);
    count_sort(out1.data(), in1.data(), len1, 2 * i + 1);
  }
  return in1;
}

std::vector<double> radix_sort_omp(const std::vector<double>& data,
                                   std::size_t n_threads) {
  std::vector<double> result;
#pragma omp parallel num_threads(n_threads)
  {
    int local_size = data.size() / n_threads;
    std::size_t t = omp_get_thread_num();
    std::vector<double> local;
    if (t == n_threads - 1) {
      std::copy(data.begin() + local_size * t, data.end(),
                std::back_inserter(local));
    } else {
      std::copy(data.begin() + local_size * t,
                data.begin() + local_size * (t + 1), std::back_inserter(local));
    }
    local = radix_sort_seq(local);
#pragma omp critical
    result = merge(result, local);
  }
  return result;
}
