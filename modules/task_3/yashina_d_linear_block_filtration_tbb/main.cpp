// Copyright 2022 Yashina Darya
#include <gtest/gtest.h>
#include <vector>
#include "./yashina_d_linear_block_filtration.h"
TEST(Omp, Test_img_1) {
    int weight = 10;
    int height = 10;
    double* m = create_random_kernel(3, 3.0);
    std::vector< std::vector<double> >image(height);
    getRandomImg(&image, weight, height);
    ASSERT_NO_THROW(getParallelOperations(&image, m, weight, height));
    delete[] m;
}

TEST(Omp, Test_img_2) {
    int weight = 12;
    int height = 10;
    double* m = create_random_kernel(3, 3.0);
    std::vector< std::vector<double> >image(height);
    getRandomImg(&image, weight, height);
    std::vector< std::vector<double> > copy_image(image);
    getSequentialOperations(&image, m, weight, height);
    getParallelOperations(&copy_image, m, weight, height);
    ASSERT_EQ(image, copy_image);
    delete[] m;
}

TEST(Omp, Test_img_3) {
    int weight = 12;
    int height = 10;
    double* m = create_random_kernel(3, 3.0);
    std::vector< std::vector<double> >image(height);
    getRandomImg(&image, weight, height);
    std::vector< std::vector<double> > copy_image(image);
    getSequentialOperations(&image, m, weight, height);
    getParallelOperations(&copy_image, m, weight, height);
    ASSERT_EQ(image, copy_image);
    delete[] m;
}
TEST(Omp, Test_img_4) {
    int weight = 12;
    int height = 10;
    double* m = create_random_kernel(3, 3.0);
    std::vector< std::vector<double> >image(height);
    getRandomImg(&image, weight, height);
    std::vector< std::vector<double> > copy_image(image);
    getSequentialOperations(&image, m, weight, height);
    getParallelOperations(&copy_image, m, weight, height);
    ASSERT_EQ(image, copy_image);
    delete[] m;
}
TEST(Omp, Test_img_5) {
    int weight = 12;
    int height = 10;
    double* m = create_random_kernel(3, 3.0);
    std::vector< std::vector<double> >image(height);
    getRandomImg(&image, weight, height);
    std::vector< std::vector<double> > copy_image(image);
    getSequentialOperations(&image, m, weight, height);
    getParallelOperations(&copy_image, m, weight, height);
    ASSERT_EQ(image, copy_image);
    delete[] m;
}
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
