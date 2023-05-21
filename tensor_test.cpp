#include <algorithm>
#include <random>
#include <stdexcept>
#include "tensor.h"

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
    
    unsigned long nelems = t._nelems;
    float *memptr = t._memory;

    Tensor t2(std::move(t));

    if (t2._shape != std::vector<int>({4,3,2,9,5})) {
        throw std::runtime_error("shape of moved tensor != new shape");
    }

    if (t2._nelems != nelems) {
        throw std::runtime_error("elements of new tensor !=  original");
    }

    if (t2._memory != memptr) {
        throw std::runtime_error("memory location of new tensor != original");
    }

    if (t._nelems != 0) {
        throw std::runtime_error("elements of moved tensor != 0");
    }

    if (t._memory != nullptr)
    {
        throw std::runtime_error("original tensor memory not nullptr");
    }
}

void test_move_constructor_gpu()
{
    printf("test_move_constructor_gpu\n");
    Tensor t = Tensor({4,3,2,9,5});
    
    unsigned long nelems = t._nelems;

    t.move_to_gpu();

    float *memptr = t._memory;

    Tensor t2(std::move(t));

    if (t2._shape != std::vector<int>({4,3,2,9,5})) {
        throw std::runtime_error("shape of moved tensor != new shape");
    }

    if (t2._nelems != nelems) {
        throw std::runtime_error("elements of new tensor !=  original");
    }

    if (t2._memory != memptr) {
        throw std::runtime_error("memory location of new tensor != original");
    }

    if (t._nelems != 0) {
        throw std::runtime_error("elements of moved tensor != 0");
    }

    if (t._memory != nullptr)
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
    
    for (int i = 0; i < t._shape[0]; ++i) {
        float val = v[i];
        t.set({i}, val);
    }

    for (int i = 0; i < t._shape[0]; ++i) {
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

    for (int i = 0; i < t._shape[0]; ++i) {
        for (int j = 0; j < t._shape[1]; ++j) {
            float val = v[i * t._shape[1] + j];
            t.set({i, j}, val);
        }
    }

    for (int i = 0; i < t._shape[0]; ++i) {
        for (int j = 0; j < t._shape[1]; ++j) {
            float val = t.get({i, j});

            float tval = v[i * t._shape[1] + j];

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

    for (int i = 0; i < t._shape[0]; ++i) {
        for (int j = 0; j < t._shape[1]; ++j) {
            for (int k = 0; k < t._shape[2]; ++k) {
                float val = v[i * t._shape[1] * t._shape[2] + j * t._shape[2] + k];
                t.set({i, j, k}, val);
            }
        }
    }

    for (int i = 0; i < t._shape[0]; ++i) {
        for (int j = 0; j < t._shape[1]; ++j) {
            for (int k = 0; k < t._shape[2]; ++k) {
                float val = t.get({i, j, k});

                float tval = v[i * t._shape[1] * t._shape[2] + j * t._shape[2] + k];

                if (val != tval) {
                    printf("Error. should be %f. But is: %f\n", tval, val);
                    throw new std::exception();
                }
            }
        }
    }
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

    for (int i = 0; i < t1._shape[0]; ++i) {
        for (int j = 0; j < t1._shape[1]; ++j) {
            for (int k = 0; k < t1._shape[2]; ++k) {
                float val1 = v1[i * t1._shape[1] * t1._shape[2] + j * t1._shape[2] + k];
                float val2 = v2[i * t1._shape[1] * t1._shape[2] + j * t1._shape[2] + k];
                
                t1.set({i, j, k}, val1);
                t2.set({i, j, k}, val2);

            }
        }
    }
    Tensor t3 = t1 + t2;

    for (int i = 0; i < t3._shape[0]; ++i) {
        for (int j = 0; j < t3._shape[1]; ++j) {
            for (int k = 0; k < t3._shape[2]; ++k) {
                float val = t3.get({i, j, k});

                float tval1 = v1[i * t3._shape[1] * t3._shape[2] + j * t3._shape[2] + k];
                float tval2 = v2[i * t3._shape[1] * t3._shape[2] + j * t3._shape[2] + k];

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
    
    for (int i = 0; i < t1._shape[0]; ++i) {
        float val = v[i];
        t1.set({i}, val);
    }

    t1.move_to_gpu();

    if (t1._on_gpu == false) {
        printf("Error. _on_gpu flag not set after moving");
        throw new std::exception();
    }

    t1.move_to_ram();

    if (t1._on_gpu == true) {
        printf("Error. _on_gpu flag not set after moving back to ram");
        throw new std::exception();
    }

    for (int i = 0; i < t1._shape[0]; ++i) {
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

    for (int i = 0; i < t1._shape[0]; ++i) {
        for (int j = 0; j < t1._shape[1]; ++j) {
            for (int k = 0; k < t1._shape[2]; ++k) {
                float val1 = v1[i * t1._shape[1] * t1._shape[2] + j * t1._shape[2] + k];
                float val2 = v2[i * t1._shape[1] * t1._shape[2] + j * t1._shape[2] + k];
                
                t1.set({i, j, k}, val1);
                t2.set({i, j, k}, val2);

            }
        }
    }

    t1.move_to_gpu();
    t2.move_to_gpu();

    Tensor t3 = t1 + t2;

    t3.move_to_ram();

    for (int i = 0; i < t3._shape[0]; ++i) {
        for (int j = 0; j < t3._shape[1]; ++j) {
            for (int k = 0; k < t3._shape[2]; ++k) {
                float val = t3.get({i, j, k});

                float tval1 = v1[i * t3._shape[1] * t3._shape[2] + j * t3._shape[2] + k];
                float tval2 = v2[i * t3._shape[1] * t3._shape[2] + j * t3._shape[2] + k];

                if (val != tval1 + tval2) {
                    printf("Error. should be %f. But is: %f\n", tval1+tval2, val);
                    throw new std::exception();
                }
            }
        }
    }
}


int main()
{
    test_move_constructor_cpu();
    test_move_constructor_gpu();

    test_1d_set_get_cpu();
    test_2d_set_get_cpu();
    test_3d_set_get_cpu();

    test_3d_add_cpu();

    test_move_memory_to_gpu_and_back();
    test_3d_add_gpu();
}