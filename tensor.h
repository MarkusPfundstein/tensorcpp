#ifndef TENSOR_H
#define TENSOR_H

#include <memory>
#include <vector>

class Tensor;
typedef std::shared_ptr<Tensor> TensorPtr;

class Tensor
{
    public:
    std::vector<int> shape;
    unsigned long int nelems;
    float *memory;
    bool is_on_gpu;
    TensorPtr gradient;

    Tensor();
    Tensor(std::vector<int> shape, bool _on_gpu=false);
    Tensor(std::vector<int> shape, std::vector<float> data, bool _on_gpu=false);
    Tensor(const Tensor &other);
    Tensor(Tensor &&other) noexcept;
    ~Tensor();

    unsigned int dimensions() const;

    bool is_scalar() const;
    bool is_vec() const;
    bool is_mat2d() const;

    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other);

    /* a + b */
    Tensor add(const Tensor &other) const;
    Tensor operator+(const Tensor &other) const;
    Tensor add_backwards(const Tensor &other) const;

    /* -a */
    Tensor operator-() const;

    /* a - b */
    Tensor operator-(const Tensor &other) const;

    // a * b
    Tensor mul(const Tensor &b) const;
    Tensor operator*(const Tensor &b) const;
    Tensor mul_backwards(const Tensor &other) const;

    // a @ b (matmul)
    Tensor matmul(const Tensor &b) const;
    Tensor operator&(const Tensor &b) const;
    Tensor matmul_backwards(const Tensor &b) const;

    // a * scalar
    Tensor mul(float sclar) const;
    Tensor operator*(float scalar) const;

    // pow(a, 2)
    Tensor pow(float power) const;
    Tensor pow(const Tensor& power) const;

    Tensor outer(const Tensor &b) const;

    // tanh()
    Tensor tanh() const;

    // transpose
    const Tensor& T_();
    Tensor T() const;

    // dot(a, b)
    static float dot(const Tensor &a, const Tensor &b);

    // relu
    Tensor relu() const;

    // sin
    Tensor sin() const;
    Tensor sin_backwards() const;

    // cos
    Tensor cos() const;

    // a / b
    Tensor operator/(float scalar);
    Tensor operator/(const Tensor &other);

    void set(const std::vector<int> &indices, float val);
    float get(const std::vector<int> &indices) const;

    void move_to_gpu();
    void move_to_ram();

    void set_data(const std::vector<float> &data);

    std::string str() const;

    private:
    int calc_mem_idx(const std::vector<int> &indices) const noexcept;

};



int __get_existing_tensor_count();

#endif