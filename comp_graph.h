#ifndef __COMP_GRAPH_H__
#define __COMP_GRAPH_H__

#include "comp_node.h"

class CompGraph
{
    private:
    std::unique_ptr<CompNode> root;
};

class Variable
{
    private:
    Tensor tensor;

    public:
    Variable(Tensor &&a);
};

#endif