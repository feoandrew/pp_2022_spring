// Copyright 2018 Nesterov Alexander
#include "../../../modules/task_1/yashina_d_linear_block_filtration/yashina_linear_block_filtration.h"
double* create_random_kernel(int size_n, double sigma) {
    double* v = new double[size_n*size_n];
    std::random_device dev;
    std::mt19937 gen(dev());
    double norm = 0;
    for (int i = -1; i < 2; i++) {
        for (int j = -1; j < 2; j++) {
            int idx = (i + 1) * 3 + (j + 1);
            v[idx] = std::exp(-(i*i + j * j) / sigma * sigma);
        }
    }
    for (int i = 0; i < 9; i++) {
        v[i] /= norm;
    }
    return v;
}
void getRandomImg(std::vector<std::vector<double>>* img,
int weight, int height) {
    std::random_device dev;
    std::mt19937 gen(dev());
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < weight; j++) {
            (*img)[i].push_back(clamp(static_cast<double>(gen()) / gen(),
            0.0, 255.0));
        }
    }
}

double calculatedNewPixelColor(double* matrix,
    const std::vector<std::vector<double>> &image, int height,
    int weight, const int x, const int y) {
    double color = 0;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            int idx = (i + 1) * 3 + j + 1;
            double imgColor =
            image[clamp(x + j, height - 1, 0)][clamp(y + i, weight - 1, 0)];
            color += (imgColor)* matrix[idx];
        }
    }
    return color;
}
void block_work(std::vector<std::vector<double>>* image, double* matrix,
int st_x, int fn_x, int st_y, int fn_y, int height, int weight) {
    for (int y = st_y; y < fn_y; y++) {
        for (int x = st_x; x < fn_x; x++) {
            (*image)[y][x] =
            calculatedNewPixelColor(matrix, *image, height, weight, x, y);
        }
    }
}
void getSequentialOperations(std::vector<std::vector<double>>* image,
    double* matrix, int weight, int height) {
    int block_weight = weight / 4;
    int block_height = height / 4;
    for (int y = 0; y < height; y += block_height) {
        for (int x = 0; x < weight; x += block_weight) {
            block_work(image, matrix, x, std::min(x + block_weight, weight),
            y, std::min(y + block_height, height), height, weight);
        }
    }
}
