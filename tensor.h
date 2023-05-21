#pragma once

#include <vector>

template <typename T>
T *alloc_ram(int n_elems)
{
    return (T*)malloc(sizeof(T) * n_elems);
}

template <typename T>
void free_ram(T *ptr)
{
    free(ptr);
}

class Tensor
{
    public:
    std::vector<int> _shape;
    unsigned long int _nelems;
    float *_memory;
    bool _on_gpu;

    Tensor();
    Tensor(std::vector<int> shape, bool _on_gpu=false);
    Tensor(const Tensor &other);
    Tensor(Tensor &&other) noexcept;
    ~Tensor();

    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other);

    Tensor operator+(const Tensor &other);

    Tensor add(const Tensor &other);

    void set(const std::vector<int> &indices, float val);
    float get(const std::vector<int> &indices);

    void move_to_gpu();
    void move_to_ram();

    private:
    int calc_mem_idx(const std::vector<int> &indices) noexcept;
};
