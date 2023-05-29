#include "comp_node.h"
#include <stdexcept>
#include <stdio.h>
#include <sstream>
#include <random>
#include <iostream>

std::string random_string(std::size_t length);
static bool __use_eager_mode__ = false;
static bool __eager_mode_gpu__ = false;

GraphNode::GraphNode(TensorPtr t)
    : tensor(t), left(nullptr), right(nullptr), type(node_type::tensor_node),
      requires_grad(false)
{

}

GraphNode::GraphNode(Tensor &&t)
    : tensor(nullptr), left(nullptr), right(nullptr), type(node_type::tensor_node),
      requires_grad(false)
{
    tensor = std::make_shared<Tensor>(std::move(t));
}

GraphNode::GraphNode(GraphNodePtr l, GraphNodePtr r, node_type _type)
    : tensor(nullptr), left(l), right(r), type(_type),
      requires_grad(false)
{
}

GraphNode::GraphNode(const GraphNode &l, const GraphNode &r, node_type _type)
    : tensor(nullptr), left(nullptr), right(nullptr), type(_type),
      requires_grad(false)
{
    left = std::make_shared<GraphNode>(l);
    right = std::make_shared<GraphNode>(r);
}

GraphNode::GraphNode(GraphNode &&l, GraphNode &&r, node_type _type)
    : tensor(nullptr), left(nullptr), right(nullptr), type(_type),
      requires_grad(false)
{
    left = std::make_shared<GraphNode>(std::forward<GraphNode>(l));
    right = std::make_shared<GraphNode>(std::forward<GraphNode>(r));
}

GraphNode::GraphNode(GraphNode &&l, node_type _type)
    : tensor(nullptr), left(nullptr), right(nullptr), type(_type),
      requires_grad(false)
{
    left = std::make_shared<GraphNode>(std::forward<GraphNode>(l));
}

GraphNode::GraphNode(const GraphNode &other)
    : tensor(other.tensor),
      left(other.left),
      right(other.right),
      type(other.type),
      requires_grad(other.requires_grad)
{
    //printf("COPY GRAPHNODE\n");
}

GraphNode::GraphNode(GraphNode &&other)
    : tensor(std::exchange(other.tensor, nullptr)),
      left(std::exchange(other.left, nullptr)),
      right(std::exchange(other.right, nullptr)),
      type(other.type),
      requires_grad(other.requires_grad)
{
    //printf("MOVE GRAPHNODE\n");
}

GraphNode::~GraphNode()
{
    //printf("delete graphnode \n");
}

void GraphNode::set_require_grad(bool require)
{
    requires_grad = require;
}

TensorPtr GraphNode::grad() const
{
    return tensor ? tensor->gradient : nullptr;
}

GraphNode GraphNode::operator+(const GraphNode &other) const
{
    GraphNode a = GraphNode(*this);
    GraphNode b = GraphNode(other);
    
    GraphNode op(
        std::move(a),
        std::move(b),
        GraphNode::comp_node_add
    );

    if (__use_eager_mode__) {
        op.eval();
    }

    return op;
}

GraphNode GraphNode::operator*(float scalar) const
{
    bool scalar_on_gpu = false;
    if (type == node_type::tensor_node) {
        scalar_on_gpu = tensor->is_on_gpu;
    }

    GraphNode b = GraphNode(Tensor({1}, {scalar}, scalar_on_gpu));

    return this->operator*(std::move(b));
}

GraphNode GraphNode::operator*(const GraphNode &other) const
{
    GraphNode a = GraphNode(*this);
    GraphNode b = GraphNode(other);
    
    GraphNode op(
        std::move(a),
        std::move(b),
        GraphNode::comp_node_mul
    );

    if (__use_eager_mode__) {
        op.eval();
    }

    return op;
}

GraphNode GraphNode::operator/(const GraphNode &other) const
{
    GraphNode a = GraphNode(*this);
    GraphNode b = GraphNode(other);
    
    GraphNode op(
        std::move(a),
        std::move(b),
        GraphNode::comp_node_div
    );

    if (__use_eager_mode__) {
        op.eval();
    }

    return op;
}

GraphNode GraphNode::operator&(const GraphNode &other) const
{
    GraphNode a = GraphNode(*this);
    GraphNode b = GraphNode(other);
    
    GraphNode op(
        std::move(a),
        std::move(b),
        GraphNode::comp_node_matmul
    );

    if (__use_eager_mode__) {
        op.eval();
    }

    return op;
}

GraphNode GraphNode::operator-(const GraphNode &other) const
{
    GraphNode a = GraphNode(*this);
    GraphNode b = GraphNode(other);
    
    GraphNode op(
        std::move(a),
        std::move(b),
        GraphNode::comp_node_min
    );

    if (__use_eager_mode__) {
        op.eval();
    }

    return op;
}
    
GraphNode GraphNode::operator-() const
{
    GraphNode a = GraphNode(*this);
    
    GraphNode op(
        std::move(a),
        GraphNode::comp_node_minself
    );

    if (__use_eager_mode__) {
        op.eval();
    }

    return op;
}

GraphNode GraphNode::pow(float power) const
{
    bool scalar_on_gpu = false;
    if (type == node_type::tensor_node) {
        scalar_on_gpu = tensor->is_on_gpu;
    }

    GraphNode a = GraphNode(*this);
    GraphNode b = GraphNode(Tensor({1}, {power}, scalar_on_gpu));
    
    GraphNode op(
        std::move(a),
        std::move(b),
        GraphNode::comp_node_pow
    );

    if (__use_eager_mode__) {
        op.eval();
    }

    return op;
}

GraphNode GraphNode::tanh() const
{
    GraphNode a = GraphNode(*this);
    
    GraphNode op(
        std::move(a),
        GraphNode::comp_node_tanh
    );

    if (__use_eager_mode__) {
        op.eval();
    }

    return op;
}

GraphNode GraphNode::relu() const
{
    GraphNode a = GraphNode(*this);
    
    GraphNode op(
        std::move(a),
        GraphNode::comp_node_relu
    );

    if (__use_eager_mode__) {
        op.eval();
    }

    return op;
}

GraphNode GraphNode::sin() const
{
    GraphNode a = GraphNode(*this);
    
    GraphNode op(
        std::move(a),
        GraphNode::comp_node_sin
    );

    if (__use_eager_mode__) {
        op.eval();
    }

    return op;
}

GraphNode GraphNode::cos() const
{
    GraphNode a = GraphNode(*this);
    
    GraphNode op(
        std::move(a),
        GraphNode::comp_node_cos
    );

    if (__use_eager_mode__) {
        op.eval();
    }

    return op;
}
void GraphNode::backward()
{
    //TensorPtr start_node = std::shared_ptr<Tensor>(new Tensor({1}, {1}, false));
    Tensor start_grad({1}, {1}, false);
    backward_step(start_grad);
}

TensorPtr GraphNode::get_tensor() const
{
    return tensor ? tensor : nullptr;
}

void GraphNode::backward_step(const Tensor &parent_grad)
{
    std::cout << "\tVisit node " << label() << " [" << tensor->str() << "]" << std::endl;

    // if we reach a node, we check if we need to store gradient. if not we are done
    if (type == node_type::tensor_node) {
        // nothing to do
        if (!requires_grad) {
            std::cout << "\t\t\tno grad required. done" << std::endl;
            return;
        }

        std::cout << "\t\t\tstore gradient" << std::endl;
        TensorPtr new_grad = std::shared_ptr<Tensor>(new Tensor(std::move(parent_grad)));
        this->tensor->gradient = new_grad;
        return;
    }

    Tensor gl;
    Tensor gr;
    switch (type) {
        case node_type::comp_node_mul:
            gl = left->tensor->mul_backwards(*right->tensor);
            gr = right->tensor->mul_backwards(*left->tensor);
            break;
        case node_type::comp_node_sin:
            gl = left->tensor->sin_backwards();
            break;
        case node_type::comp_node_add:
            gl = left->tensor->add_backwards(*right->tensor);
            gr = right->tensor->add_backwards(*left->tensor);
            break;
        default:
            throw std::runtime_error("backward pass for op not implemented");
        break;
    }

    if (gl.memory != nullptr) {
        std::cout << "\t\tgot gradient_left: " << gl.str() << std::endl;
        Tensor g2 = gl * parent_grad;
        left->backward_step(g2);
    }
    if (gr.memory != nullptr) {
        std::cout << "\t\tgot gradient_right: " << gr.str() << std::endl;
        Tensor g2 = gr * parent_grad;
        right->backward_step(g2);
    }

    std::cout << "\tLeave node " << label() << " [" << tensor->str() << "]" << std::endl;
}

TensorPtr GraphNode::eval()
{
    // base case
    if (type == node_type::tensor_node) {
        if (!tensor->is_on_gpu && __eager_mode_gpu__) {
            tensor->move_to_gpu();
        }
        return tensor;
    }

    if (tensor != nullptr) {
        //printf("return from cache %s\n", cached_result->str().c_str());
        return tensor;
    }

    TensorPtr l = left ? left->eval() : nullptr;
    TensorPtr r = right ? right->eval() : nullptr;

    switch (type) {
        case node_type::comp_node_add:
            tensor = std::make_shared<Tensor>((*l) + (*r));
            break;
        case node_type::comp_node_mul:
            tensor = std::make_shared<Tensor>((*l) * (*r));
            break;
        case node_type::comp_node_div:
            tensor = std::make_shared<Tensor>((*l) / (*r));
            break;
        case node_type::comp_node_matmul:
            tensor = std::make_shared<Tensor>((*l) & (*r));
            break;
        case node_type::comp_node_min:
            tensor = std::make_shared<Tensor>((*l) - (*r));
            break;
        case node_type::comp_node_minself:
            tensor = std::make_shared<Tensor>(-(*l));
            break;
        case node_type::comp_node_pow:
            tensor = std::make_shared<Tensor>(l->pow(*r));
            break;
        case node_type::comp_node_tanh:
            tensor = std::make_shared<Tensor>(l->tanh());
            break;
        case node_type::comp_node_relu:
            tensor = std::make_shared<Tensor>(l->relu());
            break;
        case node_type::comp_node_sin:
            tensor = std::make_shared<Tensor>(l->sin());
            break;
        case node_type::comp_node_cos:
            tensor = std::make_shared<Tensor>(l->cos());
            break;
        case node_type::comp_node:
        case node_type::tensor_node:
            throw std::runtime_error("GraphNode::eval(). invalid comp node type");
    }
    return tensor;
}

std::string GraphNode::label() const
{
    switch (type) {
        case GraphNode::node_type::tensor_node:
            return tensor->str();
        case GraphNode::node_type::comp_node_add:
            return "+";
        case GraphNode::node_type::comp_node_mul:
            return "*";
        case GraphNode::node_type::comp_node_div:
            return "/";
        case GraphNode::node_type::comp_node_matmul:
            return "&";
        case GraphNode::comp_node_min:
            return "-";
        case GraphNode::comp_node_minself:
            return "-";
        case GraphNode::comp_node_pow:
            return "^";
        case GraphNode::comp_node_tanh:
            return "tanh";
        case GraphNode::comp_node_relu:
            return "ReLU";
        case GraphNode::comp_node_sin:
            return "sin";
        case GraphNode::comp_node_cos:
            return "cos";
        case GraphNode::node_type::comp_node:
            break;
    }
    return "err";
}

std::string GraphNode::str() const
{
    switch (type) {
        case GraphNode::node_type::tensor_node:
            return "TENSOR";
        case GraphNode::node_type::comp_node_add:
            return "PLUS";
        case GraphNode::node_type::comp_node_mul:
            return "MUL";
        case GraphNode::node_type::comp_node_div:
            return "DIV";
        case GraphNode::comp_node_matmul:
            return "MATMUL";
        case GraphNode::comp_node_min:
            return "MIN";
        case GraphNode::comp_node_minself:
            return "MINSELF";
        case GraphNode::comp_node_pow:
            return "POW";
        case GraphNode::comp_node_tanh:
            return "TANH";
        case GraphNode::comp_node_relu:
            return "RELU";
        case GraphNode::node_type::comp_node:
            break;
        case GraphNode::comp_node_sin:
            return "SIN";
        case GraphNode::comp_node_cos:
            return "COS";
    }
    return "err";
}

void GraphNode::move_to_gpu()
{
    if (type == node_type::tensor_node) {
        if (!tensor->is_on_gpu) {
            tensor->move_to_gpu();
        }
        return;
    }

    if (left) {
        left->move_to_gpu();
    }
    if (right) {
        right->move_to_gpu();
    }
}

void GraphNode::move_to_ram()
{
    if (type == node_type::tensor_node) {
        if (tensor->is_on_gpu) {
            tensor->move_to_ram();
        }
        return;
    }

    if (left) {
        left->move_to_ram();
    }
    if (right) {
        right->move_to_ram();
    }
}

void GraphNode::draw(std::ostream &os) const
{
    os << "digraph G {\n";
    
    draw(os, "Output", 0);

    os << "}" << std::endl;
}

void GraphNode::draw(std::ostream &os, const std::string &parent_name, int depth) const
{
    std::stringstream sname;
    sname << str() << "_" << parent_name << "_" << depth << random_string(4);
    std::string name = sname.str();

    if (type == node_type::tensor_node) {
        os << name << "[label=\"" << (
            tensor == nullptr ? label() : ("[" + label() + "]\\n" + tensor->str())
        ) << "\"]" << std::endl;
        if (depth > 0) {
            os << parent_name << " -> " << name << std::endl;
        }
        return;
    }

    os << name << "[label=\"" << (
        tensor == nullptr ? label() : ("[" + label() + "]\\n" + tensor->str())
    ) << "\"]" << std::endl;
    if (depth > 0) {
        os << parent_name << " -> " << name << std::endl;
    }

    if (left) {
        left->draw(os, name, depth + 1);
    }
    if (right) {
        right->draw(os, name, depth + 1);
    }
}

void GraphNode::set_eager_mode(bool mode, bool on_gpu)
{
    __use_eager_mode__ = mode;
    __eager_mode_gpu__ = on_gpu;
}

std::string random_string(std::size_t length)
{
    const std::string CHARACTERS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

    std::random_device random_device;
    std::mt19937 generator(random_device());
    std::uniform_int_distribution<> distribution(0, CHARACTERS.size() - 1);

    std::string random_string;

    for (std::size_t i = 0; i < length; ++i)
    {
        random_string += CHARACTERS[distribution(generator)];
    }

    return random_string;
}
