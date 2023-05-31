#ifndef __COMP_NODE_H__
#define __COMP_NODE_H__

#include <vector>
#include <memory>
#include <ostream>
#include "tensor.h"



class GraphNode;
typedef std::shared_ptr<GraphNode> GraphNodePtr;

class GraphNode : public std::enable_shared_from_this<GraphNode>
{
    public:
    enum node_type {
        tensor_node       = 1,
        comp_node         = 100,
        comp_node_add     = 101,
        comp_node_mul     = 102,
        comp_node_matmul  = 103,
        comp_node_min     = 104,
        comp_node_minself = 105,
        comp_node_pow     = 106,
        comp_node_tanh    = 107,
        comp_node_div     = 108,
        comp_node_relu    = 109,
        comp_node_sin     = 110,
        comp_node_cos     = 111
    };

    private:
    TensorPtr tensor;
    GraphNodePtr left;
    GraphNodePtr right;

    node_type type;
    bool requires_grad;

    public:
    

    GraphNode(TensorPtr t);
    GraphNode(Tensor &&t);

    GraphNode(GraphNodePtr l, GraphNodePtr r, node_type type);
    GraphNode(const GraphNode &l, const GraphNode &r, node_type type);
    GraphNode(GraphNode &&l, GraphNode &&r, node_type type);
    GraphNode(GraphNode &&l, node_type type);

    GraphNode(const GraphNode &other);
    GraphNode(GraphNode &&other);

    ~GraphNode();

    GraphNode operator+(const GraphNode &other) const;
    GraphNode operator-(const GraphNode &other) const;
    GraphNode operator-() const;
    GraphNode operator*(const GraphNode &other) const;
    GraphNode operator*(float scalar) const;
    GraphNode operator&(const GraphNode &other) const;
    GraphNode operator/(const GraphNode &other) const;

    GraphNode pow(float power) const;
    GraphNode tanh() const;
    GraphNode relu() const;
    GraphNode sin() const;
    GraphNode cos() const;

    TensorPtr grad() const;
    TensorPtr get_tensor() const;

    void backward();
    TensorPtr eval();

    void move_to_gpu();
    void move_to_ram();

    void draw(std::ostream &os) const;

    void set_require_grad(bool require);

    static void set_eager_mode(bool mode, bool gpu=false);

    std::string label() const;    
    private:

    void backward_step(Tensor &last_result);

    std::string str() const;
    void draw(std::ostream& os, const std::string &parent_name, int depth) const;
};

#endif