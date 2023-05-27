#include <algorithm>
#include <random>
#include <stdexcept>
#include <chrono>
#include <cassert>
#include "tensor.h"

#define NDEBUG
#define assertm(exp, msg) assert(((void)msg, exp))

bool within_acceptable_error(float val, float tval) {
    float acceptable_error = 0.0001;

    if ((tval - acceptable_error) < val && (tval + acceptable_error) > val) {
        return true;
    }
    return false;
}

static std::vector<float> generate_data(size_t size)
{
    static std::uniform_int_distribution<int> distribution(-100, 100);
    static std::default_random_engine generator;

    std::vector<float> data(size);
    std::generate(data.begin(), data.end(), []() { return (float)distribution(generator); });
    return data;
}

void test_move_constructor_cpu()
{
    printf("test_move_constructor_cpu\n");
    Tensor t = Tensor({4,3,2,9,5});
    
    unsigned long nelems = t.nelems;
    float *memptr = t.memory;

    Tensor t2(std::move(t));

    if (t2.shape != std::vector<int>({4,3,2,9,5})) {
        throw std::runtime_error("shape of moved tensor != new shape");
    }

    if (t2.nelems != nelems) {
        throw std::runtime_error("elements of new tensor !=  original");
    }

    if (t2.memory != memptr) {
        throw std::runtime_error("memory location of new tensor != original");
    }

    if (t.nelems != 0) {
        throw std::runtime_error("elements of moved tensor != 0");
    }

    if (t.memory != nullptr)
    {
        throw std::runtime_error("original tensor memory not nullptr");
    }
}

void test_move_constructor_gpu()
{
    printf("test_move_constructor_gpu\n");
    Tensor t = Tensor({4,3,2,9,5});
    
    unsigned long nelems = t.nelems;

    t.move_to_gpu();

    float *memptr = t.memory;

    Tensor t2(std::move(t));

    if (t2.shape != std::vector<int>({4,3,2,9,5})) {
        throw std::runtime_error("shape of moved tensor != new shape");
    }

    if (t2.nelems != nelems) {
        throw std::runtime_error("elements of new tensor !=  original");
    }

    if (t2.memory != memptr) {
        throw std::runtime_error("memory location of new tensor != original");
    }

    if (t.nelems != 0) {
        throw std::runtime_error("elements of moved tensor != 0");
    }

    if (t.memory != nullptr)
    {
        throw std::runtime_error("original tensor memory not nullptr");
    }
}

void test_1d_set_get_cpu()
{
    printf("test_1d_set_get_cpu\n");
    constexpr int size_d1 = 5;
    Tensor t = Tensor({size_d1});

    auto v = generate_data(size_d1);
    
    for (int i = 0; i < t.shape[0]; ++i) {
        float val = v[i];
        t.set({i}, val);
    }

    for (int i = 0; i < t.shape[0]; ++i) {
        float val = t.get({i});

        if (val != v[i]) {
            printf("Error. should be %f. But is: %f\n", v[i], val);
            throw new std::exception();
        }
    }
}

void test_2d_set_get_cpu()
{
    printf("test_2d_set_get_cpu\n");
    constexpr int size_d1 = 5;
    constexpr int size_d2 = 3;
    Tensor t = Tensor({size_d1, size_d2});

    auto v = generate_data(size_d1*size_d2);

    for (int i = 0; i < t.shape[0]; ++i) {
        for (int j = 0; j < t.shape[1]; ++j) {
            float val = v[i * t.shape[1] + j];
            t.set({i, j}, val);
        }
    }

    for (int i = 0; i < t.shape[0]; ++i) {
        for (int j = 0; j < t.shape[1]; ++j) {
            float val = t.get({i, j});

            float tval = v[i * t.shape[1] + j];

            if (val != tval) {
                printf("Error. should be %f. But is: %f\n", tval, val);
                throw new std::exception();
            }
        }
    }

}

void test_3d_set_get_cpu()
{
    printf("test_3d_set_get_cpu\n");
    constexpr int size_d1 = 4;
    constexpr int size_d2 = 5;
    constexpr int size_d3 = 3;
    Tensor t = Tensor({size_d1, size_d2, size_d3});

    auto v = generate_data(size_d1 * size_d2 * size_d3);

    for (int i = 0; i < t.shape[0]; ++i) {
        for (int j = 0; j < t.shape[1]; ++j) {
            for (int k = 0; k < t.shape[2]; ++k) {
                float val = v[i * t.shape[1] * t.shape[2] + j * t.shape[2] + k];
                t.set({i, j, k}, val);
            }
        }
    }

    for (int i = 0; i < t.shape[0]; ++i) {
        for (int j = 0; j < t.shape[1]; ++j) {
            for (int k = 0; k < t.shape[2]; ++k) {
                float val = t.get({i, j, k});

                float tval = v[i * t.shape[1] * t.shape[2] + j * t.shape[2] + k];

                if (val != tval) {
                    printf("Error. should be %f. But is: %f\n", tval, val);
                    throw new std::exception();
                }
            }
        }
    }
}

void test_set_data()
{
    printf("test_set_data\n");

    Tensor t1d({3});
    t1d.set_data({1,2,3});

    assertm(t1d.get({0}) == 1, "error copying");
    assertm(t1d.get({1}) == 2, "error copying");
    assertm(t1d.get({2}) == 3, "error copying");

    Tensor t2d({2, 3}, {1,2,3,4,5,6});

    assertm(t2d.get({0, 0}) == 1, "error copying");
    assertm(t2d.get({0, 1}) == 2, "error copying");
    assertm(t2d.get({0, 2}) == 3, "error copying");
    assertm(t2d.get({1, 0}) == 4, "error copying");
    assertm(t2d.get({1, 1}) == 5, "error copying");
    assertm(t2d.get({1, 2}) == 6, "error copying");

    Tensor t3d({2,2,2}, {1,2,3,4,5,6,7,8});

    assertm(t3d.get({0, 0, 0}) == 1, "error copying");
    assertm(t3d.get({0, 0, 1}) == 2, "error copying");
    assertm(t3d.get({0, 1, 0}) == 3, "error copying");
    assertm(t3d.get({0, 1, 1}) == 4, "error copying");
    assertm(t3d.get({1, 0, 0}) == 5, "error copying");
    assertm(t3d.get({1, 0, 1}) == 6, "error copying");
    assertm(t3d.get({1, 1, 0}) == 7, "error copying");
    assertm(t3d.get({1, 1, 1}) == 8, "error copying");
}

void test_3d_add_cpu()
{
    printf("test_3d_add_cpu\n");
    constexpr int size_d1 = 4;
    constexpr int size_d2 = 5;
    constexpr int size_d3 = 7;
    Tensor t1 = Tensor({size_d1, size_d2, size_d3});
    Tensor t2 = Tensor({size_d1, size_d2, size_d3});

    auto v1 = generate_data(size_d1 * size_d2 * size_d3);
    auto v2 = generate_data(size_d1 * size_d2 * size_d3);

    for (int i = 0; i < t1.shape[0]; ++i) {
        for (int j = 0; j < t1.shape[1]; ++j) {
            for (int k = 0; k < t1.shape[2]; ++k) {
                float val1 = v1[i * t1.shape[1] * t1.shape[2] + j * t1.shape[2] + k];
                float val2 = v2[i * t1.shape[1] * t1.shape[2] + j * t1.shape[2] + k];
                
                t1.set({i, j, k}, val1);
                t2.set({i, j, k}, val2);

            }
        }
    }
    Tensor t3 = t1 + t2;

    for (int i = 0; i < t3.shape[0]; ++i) {
        for (int j = 0; j < t3.shape[1]; ++j) {
            for (int k = 0; k < t3.shape[2]; ++k) {
                float val = t3.get({i, j, k});

                float tval1 = v1[i * t3.shape[1] * t3.shape[2] + j * t3.shape[2] + k];
                float tval2 = v2[i * t3.shape[1] * t3.shape[2] + j * t3.shape[2] + k];

                if (val != tval1 + tval2) {
                    printf("Error. should be %f. But is: %f\n", tval1+tval2, val);
                    throw new std::exception();
                }
            }
        }
    }
}

void test_move_memory_to_gpu_and_back()
{
    printf("test_move_memory_to_gpu_and_back\n");
    constexpr int size_d1 = 4;
    Tensor t1 = Tensor({size_d1});

    auto v = generate_data(size_d1);
    
    for (int i = 0; i < t1.shape[0]; ++i) {
        float val = v[i];
        t1.set({i}, val);
    }

    t1.move_to_gpu();

    if (t1.is_on_gpu == false) {
        printf("Error. _on_gpu flag not set after moving");
        throw new std::exception();
    }

    t1.move_to_ram();

    if (t1.is_on_gpu == true) {
        printf("Error. _on_gpu flag not set after moving back to ram");
        throw new std::exception();
    }

    for (int i = 0; i < t1.shape[0]; ++i) {
        float tval = v[i];
        float val = t1.get({i});
        if (val != tval) {
            printf("Error. should be %f. But is: %f\n", tval, val);
            throw new std::exception();
        }
    }
}


void test_3d_add_gpu()
{
    printf("test_3d_add_gpu\n");
    constexpr int size_d1 = 42;
    constexpr int size_d2 = 52;
    constexpr int size_d3 = 74;
    Tensor t1 = Tensor({size_d1, size_d2, size_d3});
    Tensor t2 = Tensor({size_d1, size_d2, size_d3});

    auto v1 = generate_data(size_d1 * size_d2 * size_d3);
    auto v2 = generate_data(size_d1 * size_d2 * size_d3);

    for (int i = 0; i < t1.shape[0]; ++i) {
        for (int j = 0; j < t1.shape[1]; ++j) {
            for (int k = 0; k < t1.shape[2]; ++k) {
                float val1 = v1[i * t1.shape[1] * t1.shape[2] + j * t1.shape[2] + k];
                float val2 = v2[i * t1.shape[1] * t1.shape[2] + j * t1.shape[2] + k];
                
                t1.set({i, j, k}, val1);
                t2.set({i, j, k}, val2);

            }
        }
    }

    t1.move_to_gpu();
    t2.move_to_gpu();

    Tensor t3 = t1 + t2;

    t3.move_to_ram();

    for (int i = 0; i < t3.shape[0]; ++i) {
        for (int j = 0; j < t3.shape[1]; ++j) {
            for (int k = 0; k < t3.shape[2]; ++k) {
                float val = t3.get({i, j, k});

                float tval1 = v1[i * t3.shape[1] * t3.shape[2] + j * t3.shape[2] + k];
                float tval2 = v2[i * t3.shape[1] * t3.shape[2] + j * t3.shape[2] + k];

                if (val != tval1 + tval2) {
                    printf("Error. should be %f. But is: %f\n", tval1+tval2, val);
                    throw new std::exception();
                }
            }
        }
    }

    Tensor t4 = t1 + t2;

    t4.move_to_ram();

    for (int i = 0; i < t4.shape[0]; ++i) {
        for (int j = 0; j < t4.shape[1]; ++j) {
            for (int k = 0; k < t4.shape[2]; ++k) {
                float val = t4.get({i, j, k});

                float tval1 = v1[i * t4.shape[1] * t4.shape[2] + j * t4.shape[2] + k];
                float tval2 = v2[i * t4.shape[1] * t4.shape[2] + j * t4.shape[2] + k];

                if (val != tval1 + tval2) {
                    printf("Error. should be %f. But is: %f\n", tval1+tval2, val);
                    throw new std::exception();
                }
            }
        }
    }
}

void test_dot_product_cpu()
{
    printf("test_dot_product_cpu\n");   

    Tensor t1({3}, {1,2,3});
    Tensor t2({3}, {2,4,6});

    float val = dot(t1, t2);

    float res = (1*2) + (2 * 4) + (3 * 6);

    assertm(val == res, "Error dot product via direct dot call");

    Tensor r1 = t1 * t2;

    val = r1.memory[0];

    assertm(val == res, "Error dot product via tensor mul. t1 * t2");

    Tensor r2 = t2 * t1;

    val = r2.memory[0];

    assertm(val == res, "Error dot product via tensor mul. t2 * t1");
}

void test_dot_product_gpu()
{
    printf("test_dot_product_gpu\n");   

    Tensor t1({3}, {1,2,3});
    Tensor t2({3}, {2,4,6});

    t1.move_to_gpu();
    t2.move_to_gpu();

    float val = dot(t1, t2);

    float res = (1*2) + (2 * 4) + (3 * 6);

    assertm(val == res, "Error dot product via direct dot call");


    Tensor r1 = t1 * t2;
    r1.move_to_ram();

    val = r1.memory[0];
    printf("%f\n", val);

    assertm(val == res, "Error dot product via tensor mul. t1 * t2");

    Tensor r2 = t2 * t1;
    r2.move_to_ram();

    val = r2.memory[0];

    assertm(val == res, "Error dot product via tensor mul. t2 * t1");
}

void test_mul_with_two_scalar_tensors_cpu()
{
    printf("test_mul_with_two_scalar_tensors_cpu\n");

    Tensor t1({1}, {4}, false);
    Tensor t2({1}, {5}, false);

    Tensor r1 = t1 * t2;

    assertm(r1.memory[0] == 4 * 5, "t1 * t2 wrong");

    Tensor r2 = t2 * t1;

    assertm(r2.memory[0] == 4 * 5, "t2 * t1 wrong");
}

void test_mul_with_scalar_cpu()
{
    printf("test_mul_with_scalar_cpu\n");

    Tensor t({3}, {0.2015, -0.4255, 2.6087});

    Tensor r1 = t * 100.0;

    assertm(r1.shape[0] == 3, "wrong r1 shape");
    assertm(within_acceptable_error(r1.get({0}), 20.15), "t * 100.0. 1st wrong");
    assertm(within_acceptable_error(r1.get({1}), -42.55), "t * 100.0. 2nd wrong");
    assertm(within_acceptable_error(r1.get({2}), 260.87), "t * 100.0. 3rd wrong");

    Tensor s2({1}, {100.0}, false);

    Tensor r2 = t * s2;

    assertm(r2.shape[0] == 3, "wrong r2 shape");
    assertm(within_acceptable_error(r2.get({0}), 20.15), "t * s2. 1st wrong");
    assertm(within_acceptable_error(r2.get({1}), -42.55), "t * s2. 2nd wrong");
    assertm(within_acceptable_error(r2.get({2}), 260.87), "t * s2. 3rd wrong");

    Tensor r3 = s2 * t;
    assertm(r2.shape[0] == 3, "wrong r3 shape");
    assertm(within_acceptable_error(r3.get({0}), 20.15), "s2 * t. 1st wrong");
    assertm(within_acceptable_error(r3.get({1}), -42.55), "s2 * t. 2nd wrong");
    assertm(within_acceptable_error(r3.get({2}), 260.87), "s2 * t. 3rd wrong");

    Tensor t2({2,2,2}, {1,2,3,4,5,6,7,8});

    Tensor r4 = t2 * 0.1;

    assertm(r4.shape[0] == 2, "wrong r4 shape[0]");
    assertm(r4.shape[1] == 2, "wrong r4 shape[1]");
    assertm(r4.shape[2] == 2, "wrong r4 shape[2]");

    assertm(within_acceptable_error(r4.get({0, 0, 0}), 1 * 0.1), "t2 * 0.1. 1st wrong");
    assertm(within_acceptable_error(r4.get({0, 0, 1}), 2 * 0.1), "t2 * 0.1. 2nd wrong");
    assertm(within_acceptable_error(r4.get({0, 1, 0}), 3 * 0.1), "t2 * 0.1. 3rd wrong");
    assertm(within_acceptable_error(r4.get({0, 1, 1}), 4 * 0.1), "t2 * 0.1. 4st wrong");
    assertm(within_acceptable_error(r4.get({1, 0, 0}), 5 * 0.1), "t2 * 0.1. 5st wrong");
    assertm(within_acceptable_error(r4.get({1, 0, 1}), 6 * 0.1), "t2 * 0.1. 6st wrong");
    assertm(within_acceptable_error(r4.get({1, 1, 0}), 7 * 0.1), "t2 * 0.1. 7st wrong");
    assertm(within_acceptable_error(r4.get({1, 1, 1}), 8 * 0.1), "t2 * 0.1. 8st wrong");

    Tensor s3 = Tensor({1}, {0.1}, false);
    Tensor r5 = t2 * s3;

    assertm(r5.shape[0] == 2, "wrong r5 shape[0]");
    assertm(r5.shape[1] == 2, "wrong r5 shape[1]");
    assertm(r5.shape[2] == 2, "wrong r5 shape[2]");

    assertm(within_acceptable_error(r5.get({0, 0, 0}), 1 * 0.1), "t2 * s3. 1st wrong");
    assertm(within_acceptable_error(r5.get({0, 0, 1}), 2 * 0.1), "t2 * s3. 2nd wrong");
    assertm(within_acceptable_error(r5.get({0, 1, 0}), 3 * 0.1), "t2 * s3. 3rd wrong");
    assertm(within_acceptable_error(r5.get({0, 1, 1}), 4 * 0.1), "t2 * s3. 4st wrong");
    assertm(within_acceptable_error(r5.get({1, 0, 0}), 5 * 0.1), "t2 * s3. 5st wrong");
    assertm(within_acceptable_error(r5.get({1, 0, 1}), 6 * 0.1), "t2 * s3. 6st wrong");
    assertm(within_acceptable_error(r5.get({1, 1, 0}), 7 * 0.1), "t2 * s3. 7st wrong");
    assertm(within_acceptable_error(r5.get({1, 1, 1}), 8 * 0.1), "t2 * s3. 8st wrong");

    Tensor s4 = Tensor({1}, {0.1}, false);
    Tensor r6 = s4 * t2;

    assertm(r6.shape[0] == 2, "wrong r6 shape[0]");
    assertm(r6.shape[1] == 2, "wrong r6 shape[1]");
    assertm(r6.shape[2] == 2, "wrong r6 shape[2]");

    assertm(within_acceptable_error(r6.get({0, 0, 0}), 1 * 0.1), "s4 * t2. 1st wrong");
    assertm(within_acceptable_error(r6.get({0, 0, 1}), 2 * 0.1), "s4 * t2. 2nd wrong");
    assertm(within_acceptable_error(r6.get({0, 1, 0}), 3 * 0.1), "s4 * t2. 3rd wrong");
    assertm(within_acceptable_error(r6.get({0, 1, 1}), 4 * 0.1), "s4 * t2. 4st wrong");
    assertm(within_acceptable_error(r6.get({1, 0, 0}), 5 * 0.1), "s4 * t2. 5st wrong");
    assertm(within_acceptable_error(r6.get({1, 0, 1}), 6 * 0.1), "s4 * t2. 6st wrong");
    assertm(within_acceptable_error(r6.get({1, 1, 0}), 7 * 0.1), "s4 * t2. 7st wrong");
    assertm(within_acceptable_error(r6.get({1, 1, 1}), 8 * 0.1), "s4 * t2. 8st wrong");
}

void test_mul_with_scalar_gpu()
{
    printf("test_mul_with_scalar_gpu\n");

    Tensor t({3}, {0.2015, -0.4255, 2.6087}, true);

    Tensor r1 = t * 100.0;
    r1.move_to_ram();

    assertm(r1.shape[0] == 3, "wrong r1 shape");
    assertm(within_acceptable_error(r1.get({0}), 20.15), "t * 100.0. 1st wrong");
    assertm(within_acceptable_error(r1.get({1}), -42.55), "t * 100.0. 2nd wrong");
    assertm(within_acceptable_error(r1.get({2}), 260.87), "t * 100.0. 3rd wrong");

    Tensor s2({1}, {100.0}, true);
    Tensor r2 = t * s2;
    r2.move_to_ram();

    assertm(r2.shape[0] == 3, "wrong r2 shape");
    assertm(within_acceptable_error(r2.get({0}), 20.15), "t * s2. 1st wrong");
    assertm(within_acceptable_error(r2.get({1}), -42.55), "t * s2. 2nd wrong");
    assertm(within_acceptable_error(r2.get({2}), 260.87), "t * s2. 3rd wrong");

    Tensor r3 = s2 * t;
    r3.move_to_ram();
    assertm(r2.shape[0] == 3, "wrong r3 shape");
    assertm(within_acceptable_error(r3.get({0}), 20.15), "s2 * t. 1st wrong");
    assertm(within_acceptable_error(r3.get({1}), -42.55), "s2 * t. 2nd wrong");
    assertm(within_acceptable_error(r3.get({2}), 260.87), "s2 * t. 3rd wrong");

    Tensor t2({2,2,2}, {1,2,3,4,5,6,7,8}, true);

    Tensor r4 = t2 * 0.1;
    r4.move_to_ram();

    assertm(r4.shape[0] == 2, "wrong r4 shape[0]");
    assertm(r4.shape[1] == 2, "wrong r4 shape[1]");
    assertm(r4.shape[2] == 2, "wrong r4 shape[2]");

    assertm(within_acceptable_error(r4.get({0, 0, 0}), 1 * 0.1), "t2 * 0.1. 1st wrong");
    assertm(within_acceptable_error(r4.get({0, 0, 1}), 2 * 0.1), "t2 * 0.1. 2nd wrong");
    assertm(within_acceptable_error(r4.get({0, 1, 0}), 3 * 0.1), "t2 * 0.1. 3rd wrong");
    assertm(within_acceptable_error(r4.get({0, 1, 1}), 4 * 0.1), "t2 * 0.1. 4st wrong");
    assertm(within_acceptable_error(r4.get({1, 0, 0}), 5 * 0.1), "t2 * 0.1. 5st wrong");
    assertm(within_acceptable_error(r4.get({1, 0, 1}), 6 * 0.1), "t2 * 0.1. 6st wrong");
    assertm(within_acceptable_error(r4.get({1, 1, 0}), 7 * 0.1), "t2 * 0.1. 7st wrong");
    assertm(within_acceptable_error(r4.get({1, 1, 1}), 8 * 0.1), "t2 * 0.1. 8st wrong");

    Tensor s3 = Tensor({1}, {0.1}, true);
    Tensor r5 = t2 * s3;
    r5.move_to_ram();

    assertm(r5.shape[0] == 2, "wrong r5 shape[0]");
    assertm(r5.shape[1] == 2, "wrong r5 shape[1]");
    assertm(r5.shape[2] == 2, "wrong r5 shape[2]");

    assertm(within_acceptable_error(r5.get({0, 0, 0}), 1 * 0.1), "t2 * s3. 1st wrong");
    assertm(within_acceptable_error(r5.get({0, 0, 1}), 2 * 0.1), "t2 * s3. 2nd wrong");
    assertm(within_acceptable_error(r5.get({0, 1, 0}), 3 * 0.1), "t2 * s3. 3rd wrong");
    assertm(within_acceptable_error(r5.get({0, 1, 1}), 4 * 0.1), "t2 * s3. 4st wrong");
    assertm(within_acceptable_error(r5.get({1, 0, 0}), 5 * 0.1), "t2 * s3. 5st wrong");
    assertm(within_acceptable_error(r5.get({1, 0, 1}), 6 * 0.1), "t2 * s3. 6st wrong");
    assertm(within_acceptable_error(r5.get({1, 1, 0}), 7 * 0.1), "t2 * s3. 7st wrong");
    assertm(within_acceptable_error(r5.get({1, 1, 1}), 8 * 0.1), "t2 * s3. 8st wrong");

    Tensor s4 = Tensor({1}, {0.1}, true);
    Tensor r6 = s4 * t2;
    r6.move_to_ram();

    assertm(r6.shape[0] == 2, "wrong r6 shape[0]");
    assertm(r6.shape[1] == 2, "wrong r6 shape[1]");
    assertm(r6.shape[2] == 2, "wrong r6 shape[2]");

    assertm(within_acceptable_error(r6.get({0, 0, 0}), 1 * 0.1), "s4 * t2. 1st wrong");
    assertm(within_acceptable_error(r6.get({0, 0, 1}), 2 * 0.1), "s4 * t2. 2nd wrong");
    assertm(within_acceptable_error(r6.get({0, 1, 0}), 3 * 0.1), "s4 * t2. 3rd wrong");
    assertm(within_acceptable_error(r6.get({0, 1, 1}), 4 * 0.1), "s4 * t2. 4st wrong");
    assertm(within_acceptable_error(r6.get({1, 0, 0}), 5 * 0.1), "s4 * t2. 5st wrong");
    assertm(within_acceptable_error(r6.get({1, 0, 1}), 6 * 0.1), "s4 * t2. 6st wrong");
    assertm(within_acceptable_error(r6.get({1, 1, 0}), 7 * 0.1), "s4 * t2. 7st wrong");
    assertm(within_acceptable_error(r6.get({1, 1, 1}), 8 * 0.1), "s4 * t2. 8st wrong");
}

void test_mul_2d_cpu()
{
    printf("test_mul_2d_cpu\n");

    Tensor a({2, 3});
    Tensor b({3, 4});

    //printf("a\n");
    for (int i = 0; i < a.shape[0]; ++i) {
        for (int j = 0; j < a.shape[1]; ++j) {
            float val = i + j + 1;
            a.set({i, j}, val);

            //printf("%f ", val);
        }
        //printf("\n");
    }

    //printf("b\n");
    for (int i = 0; i < b.shape[0]; ++i) {
        for (int j = 0; j < b.shape[1]; ++j) {
            float val = i + j + 5;
            b.set({i, j}, val);

            //printf("%f ", val);
        }
        //printf("\n");
    }

    Tensor c = a * b;

    if (c.shape[0] != a.shape[0] || c.shape[1] != b.shape[1]) {
        //printf("invalid c shape [%d %d]. Should be [%d %d]\n", c.shape[0], c.shape[1], a.shape[0], b.shape[1]);
        throw std::exception();
    }

    //printf("c\n");
    for (int i = 0; i < c.shape[0]; ++i) {
        for (int j = 0; j < c.shape[1]; ++j) {
           // printf("%f ", c.get({i,j}));
        }
        //printf("\n");
    }
}

void test_mul_2d_large_cpu_and_gpu_compare()
{
    printf("test_mul_2d_large_cpu_and_gpu_compare\n");

    Tensor cpu_a({40, 50});
    Tensor cpu_b({50, 70});

    Tensor gpu_a(cpu_a.shape);
    Tensor gpu_b(cpu_b.shape);

    auto v1 = generate_data(cpu_a.shape[0]*cpu_a.shape[1]);
    auto v2 = generate_data(cpu_b.shape[0]*cpu_b.shape[1]);

    for (int i = 0; i < cpu_a.shape[0]; ++i) {
        for (int j = 0; j < cpu_a.shape[1]; ++j) {
            float val1 = v1[i * cpu_a.shape[1] + j]/100.0;
            
            cpu_a.set({i, j}, val1);
            gpu_a.set({i, j}, val1);
        }
    }
    for (int i = 0; i < cpu_b.shape[0]; ++i) {
        for (int j = 0; j < cpu_b.shape[1]; ++j) {
            float val1 = v2[i * cpu_b.shape[1] + j]/100.0;
            
            cpu_b.set({i, j}, val1);
            gpu_b.set({i, j}, val1);

        }
    }

    gpu_a.move_to_gpu();
    gpu_b.move_to_gpu();

    auto start_cpu = std::chrono::high_resolution_clock::now();
    Tensor cpu_c = cpu_a * cpu_b;
    auto end_cpu = std::chrono::high_resolution_clock::now();

    auto start_gpu = std::chrono::high_resolution_clock::now();
    Tensor gpu_c = gpu_a * gpu_b;
    auto end_gpu = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> dur_cpu = end_cpu - start_cpu;    
    std::chrono::duration<double, std::milli> dur_gpu = end_gpu - start_gpu;    


    printf("[%d %d] @ [%d %d]\ncpu: %f ms\ngpu: %f ms\n", 
        cpu_a.shape[0], cpu_a.shape[1], cpu_b.shape[0], cpu_b.shape[1],
        dur_cpu.count(), dur_gpu.count());

    gpu_c.move_to_ram();

    for (int i = 0; i < cpu_c.shape[0]; ++i) {
        for (int j = 0; j < cpu_c.shape[1]; ++j) {
            float cval = cpu_c.get({i, j});
            float gval = gpu_c.get({i, j});

            if (!within_acceptable_error(cval, gval)) {
                printf("cpu_val != gpu_val. cpu: %f, gpu: %f\n", cval, gval);
                throw std::exception();
            }
        }
    }
}

int main(int argc, char **argv)
{
    printf("RUN %s\n", argv[0]);

    test_move_constructor_cpu();
    test_move_constructor_gpu();


    test_1d_set_get_cpu();
    test_2d_set_get_cpu();
    test_3d_set_get_cpu();

    test_set_data();

    test_move_memory_to_gpu_and_back();

    test_3d_add_cpu();
    test_3d_add_gpu();

    test_dot_product_cpu();
    test_dot_product_gpu();

    test_mul_2d_cpu();
    test_mul_2d_large_cpu_and_gpu_compare();

    test_mul_with_two_scalar_tensors_cpu();
    test_mul_with_scalar_cpu();
    test_mul_with_scalar_gpu();

    assertm(__get_existing_tensor_count() == 0, "tensor leaked somewhere");
    printf("!!!!! ALL TESTS PASSED !!!!!\n");
    
}