#ifndef __COMP_NODE_H__
#define __COMP_NODE_H__

#include <vector>
#include <memory>
#include "tensor.h"

typedef std::shared_ptr<Tensor> TensorPtr;

class GraphNode
{
    public:

    enum type {
        tensor_node = 1,
        comp_node   = 2
    };

    virtual ~GraphNode();

    virtual const char *str() = 0;
    virtual type get_type() = 0;
    virtual TensorPtr eval() = 0;
};

class TensorNode : public GraphNode
{
    public:
    TensorPtr tensor;

    TensorNode(TensorPtr t);

    const char *str() override { return "t"; };
    type get_type() { return type::tensor_node; };
    TensorPtr eval() override;
};

class CompNode : public GraphNode
{
    protected:
    std::unique_ptr<GraphNode> left;
    std::unique_ptr<GraphNode> right;

    public:
    CompNode(GraphNode *l, GraphNode *r);

    void set(GraphNode *l, GraphNode *r);

    TensorPtr eval() override;
    type get_type() override { return type::comp_node; };

    protected:
    virtual TensorPtr op(TensorPtr l, TensorPtr r) = 0;
};

class CompNode_Add : public CompNode
{
    public:
    CompNode_Add(GraphNode *l, GraphNode *r) : CompNode(l, r) {}

    const char *str() override { return "+"; }

    protected:
    TensorPtr op(TensorPtr l, TensorPtr r) override;
};

class CompNode_Mul : public CompNode
{
    public:
    CompNode_Mul(GraphNode *l, GraphNode *r) : CompNode(l, r) {}

    const char *str() override { return "*"; }

    protected:
    TensorPtr op(TensorPtr l, TensorPtr r) override;
};

#endif