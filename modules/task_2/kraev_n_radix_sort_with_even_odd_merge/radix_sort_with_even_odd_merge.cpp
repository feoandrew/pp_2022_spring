// Copyright 2022 Kraev Nikita
#include "../../../modules/task_2/kraev_n_radix_sort_with_even_odd_merge/radix_sort_with_even_odd_merge.h"

void radixSort(std::vector<double>* vec, size_t left, size_t right) {
  size_t local_size;
  int tmp;
  int j = left;
  std::vector<std::vector<double>> counters(256);
  for (int index = 0; index < 7; index++) {
    for (size_t i = left; i < right; i++) {
      tmp = static_cast<int>(*((unsigned char*)&(*vec)[i] + index));
      counters[tmp].push_back((*vec)[i]);
    }

    for (int i = 0; i < 256; i++) {
      if (!counters[i].empty()) {
        local_size = counters[i].size();
        for (size_t k = 0; k < local_size; k++) {
          (*vec)[j] = counters[i][k];
          ++j;
        }
        counters[i].clear();
      }
    }
    j = left;
  }

  for (size_t i = left; i < right; i++) {
    tmp = static_cast<int>(*((unsigned char*)&(*vec)[i] + 7));
    counters[tmp].push_back((*vec)[i]);
  }

  for (int i = 255; i >= 128; i--) {
    if (!counters[i].empty()) {
      local_size = counters[i].size();
      for (size_t k = local_size - 1; k >= 1; k--) {
        (*vec)[j] = counters[i][k];
        j++;
      }
      (*vec)[j] = counters[i][0];
      j++;
    }
  }

  for (int i = 0; i < 128; i++) {
    if (!counters[i].empty()) {
      local_size = counters[i].size();
      for (size_t k = 0; k < local_size; k++) {
        (*vec)[j] = counters[i][k];
        j++;
      }
    }
  }
}

std::vector<double> getRandomVector(int size) {
  std::vector<double> random_vec(size);
  std::random_device rd;
  std::default_random_engine generator(rd());
  double a = -5000.0, b = 5000.0;
  std::uniform_real_distribution<double> dis(a, b);

  for (int i = 0; i < size; ++i) {
    random_vec[i] = dis(generator);
  }

  return random_vec;
}

void evenOrOddSplitter(std::vector<double>* vec, size_t left_1, size_t right_1,
                       size_t left_2, size_t right_2, bool check) {
  size_t a, b;
  if (check)
    a = left_1, b = left_2;
  else
    a = left_1 + 1, b = left_2 + 1;

  std::vector<double> tmp;

  while (a < right_1 && b < right_2) {
    if ((*vec)[a] < (*vec)[b]) {
      tmp.push_back((*vec)[a]);
      a += 2;
    } else {
      tmp.push_back((*vec)[b]);
      b += 2;
    }
  }

  if (a >= right_1) {
    for (size_t j = b; j < right_2; j += 2) {
      tmp.push_back((*vec)[j]);
    }
  } else {
    for (size_t j = a; j < right_1; j += 2) {
      tmp.push_back((*vec)[j]);
    }
  }

  if (check)
    a = left_1, b = left_2;
  else
    a = left_1 + 1, b = left_2 + 1;

  size_t i = 0;

  while (a < right_1) {
    (*vec)[a] = tmp[i];
    a += 2;
    i++;
  }

  while (b < right_2) {
    (*vec)[b] = tmp[i];
    b += 2;
    i++;
  }
}

void simpleComparator(std::vector<double>* vec, size_t left, size_t right) {
  for (size_t i = left; i < right - 1; i++) {
    if ((*vec)[i] > (*vec)[i + 1]) std::swap((*vec)[i], (*vec)[i + 1]);
  }
}

int findNumberPhaseMerge(int size) {
  int k = 1;
  while (size > std::pow(2, k)) {
    k++;
  }
  return k;
}

void parallelRadixSort(std::vector<double>* vec, int number_threads, int size) {
  int local_size = size / number_threads;
  int remainder = size % number_threads;
  std::vector<int> begin(number_threads);
  std::vector<int> end(number_threads);
  int k = findNumberPhaseMerge(number_threads);
  int thread_id;

  begin[0] = 0;
  end[0] = local_size + remainder;
  for (int i = 1; i < number_threads; i++) {
    begin[i] = end[0] + local_size * (i - 1);
    end[i] = begin[i] + local_size;
  }

  omp_set_num_threads(number_threads);

  #pragma omp parallel private(thread_id)
  {
    thread_id = omp_get_thread_num();
    radixSort(vec, begin[thread_id], end[thread_id]);
  #pragma omp barrier
  }

  for (int i = 0; i < k; i++) {
    omp_set_num_threads(number_threads);
    #pragma omp parallel private(thread_id)
    {
      thread_id = omp_get_thread_num();
      if (thread_id % 2 == 0 && thread_id + 1 < number_threads) {
        evenOrOddSplitter(vec, begin[thread_id], end[thread_id],
                          begin[thread_id + 1], end[thread_id + 1], true);
      } else if (thread_id % 2 == 1) {
        evenOrOddSplitter(vec, begin[thread_id - 1], end[thread_id - 1],
                          begin[thread_id], end[thread_id], false);
      }

      #pragma omp barrier

      if (thread_id % 2 == 0 && thread_id + 1 < number_threads) {
        simpleComparator(vec, begin[thread_id], end[thread_id]);
      } else if (thread_id % 2 == 1) {
        simpleComparator(vec, begin[thread_id] - 1, end[thread_id]);
      }

      #pragma omp barrier

      #pragma omp master
      {
        for (int j = 0; j < number_threads - 1; j += 2) {
          begin[j / 2] = begin[j];
          end[j / 2] = end[j + 1];
        }
        if (number_threads % 2 == 1) {
          begin[(number_threads - 1) / 2] = begin[number_threads - 1];
          end[(number_threads - 1) / 2] = end[number_threads - 1];
          number_threads = number_threads / 2 + 1;
        } else {
          number_threads /= 2;
        }
      }
#pragma omp barrier
    }
  }
}
