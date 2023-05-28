#include "comp_node.h"
#include <stdexcept>
#include <stdio.h>
#include <sstream>
#include <random>

std::string random_string(std::size_t length);
static bool __use_eager_mode__ = false;
static bool __eager_mode_gpu__ = false;

GraphNode::GraphNode(TensorPtr t)
    : tensor(t), cached_result(nullptr), left(nullptr), right(nullptr), type(node_type::tensor_node)
{

}

GraphNode::GraphNode(Tensor &&t)
    : tensor(nullptr), cached_result(nullptr), left(nullptr), right(nullptr), type(node_type::tensor_node)
{
    tensor = std::make_shared<Tensor>(std::move(t));
}

GraphNode::GraphNode(GraphNodePtr l, GraphNodePtr r, node_type _type)
    : tensor(nullptr), cached_result(nullptr), left(l), right(r), type(_type)
{
}

GraphNode::GraphNode(const GraphNode &l, const GraphNode &r, node_type _type)
    : tensor(nullptr), cached_result(nullptr), left(nullptr), right(nullptr), type(_type)
{
    left = std::make_shared<GraphNode>(l);
    right = std::make_shared<GraphNode>(r);
}

GraphNode::GraphNode(GraphNode &&l, GraphNode &&r, node_type _type)
    : tensor(nullptr), cached_result(nullptr), left(nullptr), right(nullptr), type(_type)
{
    left = std::make_shared<GraphNode>(std::forward<GraphNode>(l));
    right = std::make_shared<GraphNode>(std::forward<GraphNode>(r));
}

GraphNode::GraphNode(GraphNode &&l, node_type _type)
    : tensor(nullptr), cached_result(nullptr), left(nullptr), right(nullptr), type(_type)
{
    left = std::make_shared<GraphNode>(std::forward<GraphNode>(l));
}

GraphNode::GraphNode(const GraphNode &other)
    : tensor(other.tensor),
      cached_result(other.cached_result), 
      left(other.left),
      right(other.right),
      type(other.type)
{
    //printf("COPY GRAPHNODE\n");
}

GraphNode::GraphNode(GraphNode &&other)
    : tensor(std::exchange(other.tensor, nullptr)),
      cached_result(std::exchange(other.cached_result, nullptr)),
      left(std::exchange(other.left, nullptr)),
      right(std::exchange(other.right, nullptr)),
      type(other.type)
{
    //printf("MOVE GRAPHNODE\n");
}

GraphNode::~GraphNode()
{
    //printf("delete graphnode \n");
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

TensorPtr GraphNode::eval()
{
    // base case
    if (type == node_type::tensor_node) {
        if (!tensor->is_on_gpu && __eager_mode_gpu__) {
            tensor->move_to_gpu();
        }
        return tensor;
    }

    if (cached_result != nullptr) {
        printf("return from cache %s\n", cached_result->str().c_str());
        return cached_result;
    }

    TensorPtr l = left ? left->eval() : nullptr;
    TensorPtr r = right ? right->eval() : nullptr;

    switch (type) {
        case node_type::comp_node_add:
            cached_result = std::make_shared<Tensor>((*l) + (*r));
            break;
        case node_type::comp_node_mul:
            cached_result = std::make_shared<Tensor>((*l) * (*r));
            break;
        case node_type::comp_node_div:
            cached_result = std::make_shared<Tensor>((*l) / (*r));
            break;
        case node_type::comp_node_matmul:
            cached_result = std::make_shared<Tensor>((*l) & (*r));
            break;
        case node_type::comp_node_min:
            cached_result = std::make_shared<Tensor>((*l) - (*r));
            break;
        case node_type::comp_node_minself:
            cached_result = std::make_shared<Tensor>(-(*l));
            break;
        case node_type::comp_node_pow:
            cached_result = std::make_shared<Tensor>(l->pow(*r));
            break;
        case node_type::comp_node_tanh:
            cached_result = std::make_shared<Tensor>(l->tanh());
            break;
        case node_type::comp_node:
        case node_type::tensor_node:
            throw std::runtime_error("GraphNode::eval(). invalid comp node type");
    }
    return cached_result;
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
        case GraphNode::node_type::comp_node:
            break;
    }
    return "err";
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

    os << name << "[label=\"" << (
        cached_result == nullptr ? label() : ("[" + label() + "]\\n" + cached_result->str())
    ) << "\"]" << std::endl;
    if (depth > 0) {
        os << parent_name << " -> " << name << std::endl;
    }
    if (type == node_type::tensor_node) {
        return;
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
