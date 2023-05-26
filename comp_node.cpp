#include "comp_node.h"
#include <stdio.h>

GraphNode::~GraphNode()
{
    printf("delete graphnode\n");
}

TensorNode::TensorNode(TensorPtr t)
    : tensor(t)
{

}

TensorPtr TensorNode::eval()
{
    return tensor;
}

CompNode::CompNode(GraphNode *l, GraphNode *r)
    : GraphNode()
{
    set(l, r);
}

void CompNode::set(GraphNode *l, GraphNode *r)
{
    left = std::unique_ptr<GraphNode>(l);
    right = std::unique_ptr<GraphNode>(r);
}

TensorPtr CompNode::eval()
{
    TensorPtr left_tensor = left->eval();
    TensorPtr right_tensor = right->eval();

    return op(left_tensor, right_tensor);
}

TensorPtr CompNode_Add::op(TensorPtr l, TensorPtr r)
{
    Tensor c = (*l) + (*r);
    return std::shared_ptr<Tensor>(new Tensor(std::move(c)));
}

TensorPtr CompNode_Mul::op(TensorPtr l, TensorPtr r)
{
    Tensor c = (*l) * (*r);
    return std::shared_ptr<Tensor>(new Tensor(std::move(c)));
}

#if 0
void CompNode_Add::compute()
{
    if (left->get_type() == type::comp_node) {
        (static_cast<CompNode*>(left.get()))->compute();
    }
    if (right->get_type() == type::comp_node) {

    }
}

CompNode::CompNode(const std::vector<TensorPtr> &_inputs)
    : inputs(_inputs)
{

}

void CompNode::move_to_gpu()
{
    for (auto it = inputs.begin(); it != inputs.end(); ++it) {
        (*it)->move_to_gpu();
    }
    if (output) {
        output->move_to_gpu();
    }
}

void CompNode::move_to_ram()
{
    for (auto it = inputs.begin(); it != inputs.end(); ++it) {
        (*it)->move_to_ram();
    }
    if (output) {
        output->move_to_ram();
    }
}

const Tensor& CompNode::get_output_ref()
{
    return *output;
}

CompNode_Add::CompNode_Add(TensorPtr a, TensorPtr b)
    : CompNode({a, b})
{

}

CompNode_Add::CompNode_Add(Tensor &&a, Tensor &&b)
    : CompNode(
        {
            TensorPtr(new Tensor(std::move(a))), 
            TensorPtr(new Tensor(std::move(b))),
        })
{

}

void CompNode_Add::compute()
{
    TensorPtr a = inputs[0];
    TensorPtr b = inputs[1];

    Tensor c = (*a) + (*b);
    output = TensorPtr(new Tensor(std::move(c)));
}

const char* CompNode_Add::str()
{
    return "+";
}

CompNode_Mul::CompNode_Mul(TensorPtr a, TensorPtr b)
    : CompNode({a, b})
{

}

CompNode_Mul::CompNode_Mul(Tensor &&a, Tensor &&b)
    : CompNode(
        {
            TensorPtr(new Tensor(std::move(a))), 
            TensorPtr(new Tensor(std::move(b))),
        })
{

}

void CompNode_Mul::compute()
{
    TensorPtr a = inputs[0];
    TensorPtr b = inputs[1];

    Tensor c = (*a) * (*b);
    output = TensorPtr(new Tensor(std::move(c)));
}

const char* CompNode_Mul::str()
{
    return "*";
}

#endif