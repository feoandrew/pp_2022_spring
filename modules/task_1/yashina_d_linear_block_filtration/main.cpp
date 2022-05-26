// Copyright 2018 Nesterov Alexander
#include <gtest/gtest.h>
#include <vector>
#include "../yashina_d_linear_block_filtration/yashina_linear_block_filtration.h"
TEST(Sequential, Test_img_1) {
    int weight = 10;
    int height = 10;
    double* m = create_random_kernel(3, 3.0);
    std::vector< std::vector<double> >image(height);
    getRandomImg(&image, weight, height);
    ASSERT_NO_THROW(getSequentialOperations(&image, m, weight, height));
    delete[] m;
}
TEST(Sequential, Test_img_2) {
    int weight = 12;
    int height = 10;
    double* m = create_random_kernel(3, 3.0);
    std::vector< std::vector<double> >image(height);
    getRandomImg(&image, weight, height);
    ASSERT_NO_THROW(getSequentialOperations(&image, m, weight, height));
    delete[] m;
}

TEST(Sequential, Test_img_3) {
    int weight = 14;
    int height = 15;
    double* m = create_random_kernel(3, 3.0);
    std::vector< std::vector<double> >image(height);
    getRandomImg(&image, weight, height);
    ASSERT_NO_THROW(getSequentialOperations(&image, m, weight, height));
    delete[] m;
}
TEST(Sequential, Test_img_4) {
    int weight = 16;
    int height = 17;
    double* m = create_random_kernel(3, 3.0);
    std::vector< std::vector<double> >image(height);
    getRandomImg(&image, weight, height);
    ASSERT_NO_THROW(getSequentialOperations(&image, m, weight, height));
    delete[] m;
}
TEST(Sequential, Test_img_5) {
    int weight = 15;
    int height = 12;
    double* m = create_random_kernel(3, 3.0);
    std::vector< std::vector<double> >image(height);
    getRandomImg(&image, weight, height);
    ASSERT_NO_THROW(getSequentialOperations(&image, m, weight, height));
    delete[] m;
}
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

