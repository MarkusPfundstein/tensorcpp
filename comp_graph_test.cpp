#include "comp_node.h"
#include <cstring>
#include <cassert>
#include <iostream>

#define NDEBUG
#define assertm(exp, msg) assert(((void)msg, exp))

void test_compnode_add_cpu()
{
    printf("test_compnode_add_cpu\n");

    std::vector<float> ref({1.4, 2.3, 3.2, 4.1});

    TensorPtr a(new Tensor({4}, {1,2,3,4}));
    TensorPtr b(new Tensor({4}, {0.4,0.3,0.2,0.1}));

    GraphNode add(
        std::make_shared<GraphNode>(a),
        std::make_shared<GraphNode>(b),
        GraphNode::comp_node_add
    );

    TensorPtr c = add.eval();

    assertm(memcmp(c->memory, ref.data(), c->nelems * sizeof(float)) == 0, "add failed");

    GraphNode a1(Tensor({4}, {1,2,3,4}));
    GraphNode a2(Tensor({4}, {0.4,0.3,0.2,0.1}));

    GraphNode r2 = a1 + a2;

    TensorPtr c2 = r2.eval();

    printf("[%f %f %f %f]\n", c2->memory[0],c2->memory[1],c2->memory[2],c2->memory[3]);

    assertm(memcmp(c2->memory, ref.data(), c2->nelems * sizeof(float)) == 0, "add with + interface failed");
}

void test_compnode_mul_cpu()
{
    printf("test_compnode_mul_cpu\n");

    GraphNode a1(Tensor({3}, {1,2,3}));
    GraphNode a2(Tensor({3}, {2,3,4}));

    GraphNode op = a1 * a2;

    TensorPtr c = op.eval();

    float res = 1*2 + 2*3 + 3*4;
    assertm(res == c->memory[0], "mul op dot product failed");
}

void test_compnode_add_gpu()
{
    printf("test_compnode_add_gpu\n");

    std::vector<float> ref({1.4, 2.3, 3.2, 4.1});

    GraphNode a1(Tensor({4}, {1,2,3,4}));
    GraphNode a2(Tensor({4}, {0.4,0.3,0.2,0.1}));

    GraphNode r2 = a1 + a2;

    r2.move_to_gpu();
    TensorPtr c2 = r2.eval();
    c2->move_to_ram();

    printf("[%f %f %f %f]\n", c2->memory[0],c2->memory[1],c2->memory[2],c2->memory[3]);

    assertm(memcmp(c2->memory, ref.data(), c2->nelems * sizeof(float)) == 0, "add with + interface failed");
}

void test_graphnodes_cpu()
{
    printf("test_graphnodes_cpu\n");

    GraphNodePtr mul = std::make_shared<GraphNode>(
        std::make_shared<GraphNode>(TensorPtr(new Tensor({1}, {5}, false))),
        std::make_shared<GraphNode>(TensorPtr(new Tensor({3}, {1,2,3}, false))),
        GraphNode::comp_node_mul
    );

    GraphNodePtr add = std::make_shared<GraphNode>(
        mul,
        std::make_shared<GraphNode>(TensorPtr(new Tensor({3}, {6,7,8}, false))),
        GraphNode::comp_node_add
    );

    TensorPtr out = add->eval();

    assertm(out->memory[0] == 11, "out[0] wrong");
    assertm(out->memory[1] == 17, "out[1] wrong");
    assertm(out->memory[2] == 23, "out[2] wrong");
}

void test_graphnodes_interface_cpu()
{
    printf("test_graphnodes_interface_cpu\n");

    GraphNode a(Tensor({1}, {5}, false));
    GraphNode b(Tensor({3}, {1,2,3}, false));

    GraphNode c = a * b;

    GraphNode d = c + GraphNode(Tensor({3}, {6, 7, 8}, false));

    // final dot
    GraphNode e = c * d;

    TensorPtr out = e.eval();

    assertm(out->memory[0] == 570.0, "invalid result.");
}

void test_graphnodes_interface_gpu()
{
    printf("test_graphnodes_interface_gpu\n");

    bool use_gpu = true;

    GraphNode a(Tensor({1}, {5}, use_gpu));
    GraphNode b(Tensor({3}, {1,2,3}, use_gpu));

    GraphNode c = a * b;

    GraphNode d = c + GraphNode(Tensor({3}, {6, 7, 8}, use_gpu));

    // final dot
    GraphNode e = c * d;

    TensorPtr out = e.eval();
    out->move_to_ram();

    assertm(out->memory[0] == 570.0, "invalid result.");
}

void test_draw()
{
    printf("test_draw\n");

    GraphNode a(Tensor({1}, {5}, false));
    GraphNode b(Tensor({3}, {1,2,3}, false));

    GraphNode c = a * b;

    GraphNode d = c + GraphNode(Tensor({3}, {6, 7, 8}, false));

    // final dot
    GraphNode e = c * d;
    e.move_to_gpu();
    printf("%s\n", e.eval()->str().c_str());

    GraphNode f = e * GraphNode(Tensor({2,3}, {0.01, 0.32, -3.43, 3.2, 4.8, 0.0002}, false));

    GraphNode g = GraphNode(Tensor({3}, {0,1,0}, false));

    GraphNode final = f * g;
    final.move_to_gpu();
    TensorPtr result = final.eval();

    printf("%s\n", result->str().c_str());



    //GraphNode final = GraphNode(Tensor({2}, {0.5, 0.5})) * h;


    //final.draw(std::cerr);

    //final.eval();
}

int main(int argc, char **argv)
{
    printf("RUN %s\n", argv[0]);

    test_compnode_add_cpu();
    test_compnode_add_gpu();

    test_compnode_mul_cpu();

    test_graphnodes_cpu();

    test_graphnodes_interface_cpu();
    test_graphnodes_interface_gpu();

    test_draw();

    printf("tensors left in mem: %d\n", __get_existing_tensor_count());
    assertm(__get_existing_tensor_count() == 0, "tensor leaked somewhere");
    printf("!!!!! ALL TESTS PASSED !!!!!\n");

    return 0;
}