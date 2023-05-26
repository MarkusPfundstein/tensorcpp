#ifndef TENSOR_H
#define TENSOR_H

#include <vector>

class Tensor
{
    public:
    std::vector<int> shape;
    unsigned long int nelems;
    float *memory;
    bool is_on_gpu;

    Tensor();
    Tensor(std::vector<int> shape, bool _on_gpu=false);
    Tensor(std::vector<int> shape, std::vector<float> data, bool _on_gpu=false);
    Tensor(const Tensor &other);
    Tensor(Tensor &&other) noexcept;
    ~Tensor();

    unsigned int dimensions() const;

    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other);


    Tensor add(const Tensor &other) const;
    Tensor operator+(const Tensor &other);

    Tensor mul(const Tensor &b) const;
    Tensor mul(float sclar) const;
    Tensor operator*(const Tensor &b);
    Tensor operator*(float scalar);

    void set(const std::vector<int> &indices, float val);
    float get(const std::vector<int> &indices);

    void move_to_gpu();
    void move_to_ram();

    void set_data(const std::vector<float> &data);

    private:
    int calc_mem_idx(const std::vector<int> &indices) noexcept;
};

float dot(const Tensor &a, const Tensor &b);

#endif