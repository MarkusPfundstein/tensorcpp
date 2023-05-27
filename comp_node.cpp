#include "comp_node.h"
#include <stdexcept>
#include <stdio.h>


GraphNode::GraphNode(TensorPtr t)
    : tensor(t), left(nullptr), right(nullptr), type(node_type::tensor_node)
{

}

GraphNode::GraphNode(Tensor &&t)
    : tensor(nullptr), left(nullptr), right(nullptr), type(node_type::tensor_node)
{
    tensor = std::make_shared<Tensor>(std::move(t));
}

GraphNode::GraphNode(GraphNodePtr l, GraphNodePtr r, node_type _type)
    : tensor(nullptr), left(l), right(r), type(_type)
{
}

GraphNode::GraphNode(const GraphNode &l, const GraphNode &r, node_type _type)
    : tensor(nullptr), left(nullptr), right(nullptr), type(_type)
{
    left = std::make_shared<GraphNode>(l);
    right = std::make_shared<GraphNode>(r);
}

GraphNode::GraphNode(GraphNode &&l, GraphNode &&r, node_type _type)
    : tensor(nullptr), left(nullptr), right(nullptr), type(_type)
{
    left = std::make_shared<GraphNode>(std::forward<GraphNode>(l));
    right = std::make_shared<GraphNode>(std::forward<GraphNode>(r));
}

GraphNode::GraphNode(const GraphNode &other)
    : tensor(other.tensor),
      left(other.left),
      right(other.right),
      type(other.type)
{
    //printf("COPY GRAPHNODE\n");
}

GraphNode::GraphNode(GraphNode &&other)
    : tensor(std::exchange(other.tensor, nullptr)),
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

GraphNode GraphNode::operator+(const GraphNode &other)
{
    GraphNode a = GraphNode(*this);
    GraphNode b = GraphNode(other);
    
    GraphNode op(
        std::move(a),
        std::move(b),
        GraphNode::comp_node_add
    );

    return op;
}

GraphNode GraphNode::operator*(const GraphNode &other)
{
    GraphNode a = GraphNode(*this);
    GraphNode b = GraphNode(other);
    
    GraphNode op(
        std::move(a),
        std::move(b),
        GraphNode::comp_node_mul
    );

    return op;
}

TensorPtr GraphNode::eval()
{
    // base case
    if (type == node_type::tensor_node) {
        return tensor;
    }

    TensorPtr l = left->eval();
    TensorPtr r = right->eval();

    if (type == node_type::comp_node_add) {
        Tensor c = (*l) + (*r);
        return std::make_shared<Tensor>(std::move(c));
    }
    else if (type == node_type::comp_node_mul) {
        Tensor c = (*l) * (*r);
        return std::make_shared<Tensor>(std::move(c));
    } else {
        throw std::runtime_error("invalid type");
    }
}

void GraphNode::move_to_gpu()
{
    if (type == node_type::tensor_node) {
        if (!tensor->is_on_gpu) {
            tensor->move_to_gpu();
        }
        return;
    }

    left->move_to_gpu();
    right->move_to_gpu();
}

void GraphNode::move_to_ram()
{
    if (type == node_type::tensor_node) {
        if (tensor->is_on_gpu) {
            tensor->move_to_ram();
        }
        return;
    }

    left->move_to_ram();
    right->move_to_ram();
}