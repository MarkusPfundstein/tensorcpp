#ifndef __COMP_NODE_H__
#define __COMP_NODE_H__

#include <vector>
#include <memory>
#include "tensor.h"

typedef std::shared_ptr<Tensor> TensorPtr;

class GraphNode;
typedef std::shared_ptr<GraphNode> GraphNodePtr;

class GraphNode : public std::enable_shared_from_this<GraphNode>
{

    // graph node holds either tensor or inputs. which are another graph node
    TensorPtr tensor;
    GraphNodePtr left;
    GraphNodePtr right;
    
    public:
    
    enum node_type {
        tensor_node     = 1,
        comp_node       = 100,
        comp_node_add   = 101,
        comp_node_mul   = 102
    };

    node_type type;

    GraphNode(TensorPtr t);
    GraphNode(Tensor &&t);

    GraphNode(GraphNodePtr l, GraphNodePtr r, node_type type);
    GraphNode(const GraphNode &l, const GraphNode &r, node_type type);
    GraphNode(GraphNode &&l, GraphNode &&r, node_type type);

    GraphNode(const GraphNode &other);
    GraphNode(GraphNode &&other);

    ~GraphNode();

    GraphNode operator+(const GraphNode &other);
    GraphNode operator*(const GraphNode &other);

    TensorPtr eval();

    void move_to_gpu();
    void move_to_ram();
};

#endif