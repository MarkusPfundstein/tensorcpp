#include "comp_node.h"
#include <cstring>
#include <cassert>
#include <iostream>
#include <fstream>
#include "tensor_gpu.h"

#define NDEBUG
#define assertm(exp, msg) assert(((void)msg, exp))

#define NDEBUG
#define assertm(exp, msg) assert(((void)msg, exp))

bool within_acceptable_error(float val, float tval, float acceptable_error=0.0001) {
    if ((tval - acceptable_error) <= val && (tval + acceptable_error) >= val) {
        return true;
    }
    return false;
}

bool check_memory_against_array(std::vector<float> res, const TensorPtr& t, float acceptable_error=0.0001)
{
    if (t->is_on_gpu) {
        t->move_to_ram();
    }

    assert(t != nullptr);
    int failure_index=-1;
    for (unsigned long i = 0; i < t->nelems; ++i) {
        if (!within_acceptable_error(res[i], t->memory[i], acceptable_error)) {
            failure_index = i;
            break;
        }
    }

    if (failure_index >= 0) {
        printf("check_memory_against_array FAILURE. m[%d] = %f. Should be: %f\n",
               failure_index, t->memory[failure_index], res[failure_index]);
    }

    return failure_index == -1;
}

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

void test_compnode_dot_cpu()
{
    printf("test_compnode_dot_cpu\n");

    GraphNode a1(Tensor({3}, {1,2,3}));
    GraphNode a2(Tensor({3}, {2,3,4}));

    GraphNode op = a1 & a2;

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
    GraphNode e = c & d;

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
    GraphNode e = c & d;

    TensorPtr out = e.eval();
    out->move_to_ram();

    assertm(out->memory[0] == 570.0, "invalid result.");
}

void test_graphnodes_min()
{
    printf("test_graphnodes_min\n");
    GraphNode a(Tensor({3}, {7,8,9}));
    GraphNode b(Tensor({3}, {1,2,3}));

    GraphNode c = a - b;

    assert(check_memory_against_array({6, 6, 6}, c.eval()));
}

void test_graphnodes_div()
{
    printf("test_graphnodes_div\n");
    GraphNode a(Tensor({3}, {6,9,16}));
    GraphNode b(Tensor({3}, {2,3,4}));

    GraphNode c = a / b;

    TensorPtr d = c.eval();

    printf("%s\n", d->str().c_str());

    assert(check_memory_against_array({6/2, 9/3, 16/4}, d));
}

void test_graphnodes_minself()
{
    printf("test_graphnodes_minself\n");
    GraphNode b(Tensor({3}, {1,2,3}));

    GraphNode c = -b;

    assert(check_memory_against_array({-1, -2, -3}, c.eval()));
}

void test_graphnodes_tanh_cpu()
{
    printf("test_graphnodes_tanh_cpu\n");

    GraphNode a(Tensor({3}, {0.3015, 0.7785, -0.0818}));

    GraphNode b = a.tanh();
    TensorPtr r = b.eval();

    printf("%s\n", r->str().c_str());
    assert(check_memory_against_array({0.2927,  0.6518, -0.0816}, r));
}

void test_graphnodes_tanh_gpu()
{
    printf("test_graphnodes_tanh_gpu\n");

    GraphNode a(Tensor({3}, {0.3015, 0.7785, -0.0818}));

    GraphNode b = a.tanh();
    b.move_to_gpu();
    TensorPtr r = b.eval();
    r->move_to_ram();

    printf("%s\n", r->str().c_str());
    assert(check_memory_against_array({0.2927,  0.6518, -0.0816}, r));
}

void test_graphnodes_pow_cpu()
{
    printf("test_graphnodes_pow_cpu\n");

    GraphNode a(Tensor({3}, {0.3015, 0.7785, -0.0818}));

    GraphNode b = a.pow(-2);
    TensorPtr r = b.eval();

    printf("%s\n", r->str().c_str());
    assert(check_memory_against_array({11.0008,  1.6500, 149.4491}, r));

    GraphNode t2(Tensor({3}, {0.3015, 0.7785, -0.0818}));
    GraphNode t3(Tensor({3}, {0.3015, 0.7785, -0.0818}));

    GraphNode t4 = t2 * t3; // pointwise mul
    GraphNode r2 = t4.pow(-2);

    assert(check_memory_against_array({1.2102e+02, 2.7225e+00, 2.2335e+04}, r2.eval(), 0.1));
}

void test_graphnodes_pow_gpu()
{
    printf("test_graphnodes_pow_gpu\n");

    GraphNode a(Tensor({3}, {0.3015, 0.7785, -0.0818}));

    GraphNode b = a.pow(-2);
    b.move_to_gpu();
    TensorPtr r = b.eval();
    r->move_to_ram();

    printf("%s\n", r->str().c_str());
    assert(check_memory_against_array({11.0008,  1.6500, 149.4491}, r));

}

void test_draw()
{
    printf("test_draw\n");

    GraphNode a(Tensor({1}, {5}, false));
    GraphNode b(Tensor({3}, {1,2,3}, false));

    GraphNode c = a * b;

    GraphNode d = c + GraphNode(Tensor({3}, {6, 7, 8}, false));

    // final dot
    GraphNode e = c & d;

    std::ofstream of;
    of.open("graphs/graph_first_eval_1.dot", std::ios::out | std::ios::trunc);
    e.draw(of);
    of.close();

    e.move_to_gpu();
    TensorPtr result = e.eval();
    
    printf("%s\n", result->str().c_str());
    of.open("graphs/graph_first_eval_2.dot", std::ios::out | std::ios::trunc);
    e.draw(of);
    of.close();

    GraphNode f = e * GraphNode(Tensor({3, 2}, {0.01, 0.32, -3.43, 3.2, 4.8, 0.0002}, false));

    GraphNode g = GraphNode(Tensor({3}, {0,1,0}, false));

    GraphNode final = g & f;
    final.move_to_gpu();
    
    of.open("graphs/graph_final_eval_1.dot", std::ios::out | std::ios::trunc);
    final.draw(of);
    of.close();

    result = final.eval();
    printf("%s\n", result->str().c_str());

    of.open("graphs/graph_final_eval_2.dot", std::ios::out | std::ios::trunc);
    final.draw(of);
    of.close();
}

void test_network_XOR_with_pretrained_weights_cpu_lazy()
{
    printf("test_network_XOR_with_pretrained_weights_cpu_lazy\n");

    // input [1, 0]
    GraphNode xs1(Tensor({2}, {0,1}));

    // hidden layer 1 (5 nodes)
    GraphNode h1w(Tensor({2,5}, {0.3015, 0.7785, 0.0818, 0.6411, -1.3131,
                                 0.3517, -0.9395, -1.2009, 1.0679, 0.4376}));

    // hidden layer 2 (4 nodes)
    GraphNode h2w(Tensor({5,4},
                    {1.5171,  1.2708,  1.3553, -0.3223,
                     1.4185,  0.2538, -0.9858, -1.7510,
                     -0.4906, -0.3851, -1.3264,  1.8427,
                     -0.5821, -0.7318, -0.6390, -2.0936,
                     -1.1953, -1.0868,  0.4595,  0.4198}));

    // output layer (1)
    GraphNode ow(Tensor({4, 1}, {-1.0577, -0.2073, -1.1130, -2.1461}));

    GraphNode v1 = xs1 & h1w;
    GraphNode y1 = v1.tanh();
    GraphNode v2 = y1 & h2w;
    GraphNode y2 = v2.tanh();
    GraphNode v3 = y2 & ow;
    GraphNode y3 = v3.tanh();

    //y3.move_to_gpu();
    TensorPtr out1 = y3.eval();
    //out1->move_to_ram();
    printf("%s\n", out1->str().c_str());
    assert(check_memory_against_array({0.955728}, out1));
}

void test_network_XOR_with_pretrained_weights_cpu_eager()
{
    printf("test_network_XOR_with_pretrained_weights_cpu_eager\n");

    GraphNode::set_eager_mode(true);

    // input [1, 0]
    GraphNode xs1(Tensor({2}, {0,1}));

    // hidden layer 1 (5 nodes)
    GraphNode h1w(Tensor({2,5}, {0.3015, 0.7785, 0.0818, 0.6411, -1.3131,
                                 0.3517, -0.9395, -1.2009, 1.0679, 0.4376}));

    // hidden layer 2 (4 nodes)
    GraphNode h2w(Tensor({5,4},
                    {1.5171,  1.2708,  1.3553, -0.3223,
                     1.4185,  0.2538, -0.9858, -1.7510,
                     -0.4906, -0.3851, -1.3264,  1.8427,
                     -0.5821, -0.7318, -0.6390, -2.0936,
                     -1.1953, -1.0868,  0.4595,  0.4198}));

    // output layer (1)
    GraphNode ow(Tensor({4, 1}, {-1.0577, -0.2073, -1.1130, -2.1461}));

    GraphNode v1 = xs1 & h1w;
    GraphNode y1 = v1.tanh();
    GraphNode v2 = y1 & h2w;
    GraphNode y2 = v2.tanh();
    GraphNode v3 = y2 & ow;
    GraphNode y3 = v3.tanh();

    //y3.move_to_gpu();
    TensorPtr out1 = y3.eval();
    //out1->move_to_ram();
    printf("%s\n", out1->str().c_str());
    assert(check_memory_against_array({0.955728}, out1));

    GraphNode::set_eager_mode(false);
}

void test_network_XOR_with_pretrained_weights_gpu_lazy()
{
    printf("test_network_XOR_with_pretrained_weights_gpu_lazy\n");

    // input [1, 0]
    GraphNode xs1(Tensor({2}, {0,1}));

    // hidden layer 1 (5 nodes)
    GraphNode h1w(Tensor({2,5}, {0.3015, 0.7785, 0.0818, 0.6411, -1.3131,
                                 0.3517, -0.9395, -1.2009, 1.0679, 0.4376}));

    // hidden layer 2 (4 nodes)
    GraphNode h2w(Tensor({5,4},
                    {1.5171,  1.2708,  1.3553, -0.3223,
                     1.4185,  0.2538, -0.9858, -1.7510,
                     -0.4906, -0.3851, -1.3264,  1.8427,
                     -0.5821, -0.7318, -0.6390, -2.0936,
                     -1.1953, -1.0868,  0.4595,  0.4198}));

    // output layer (1)
    GraphNode ow(Tensor({4, 1}, {-1.0577, -0.2073, -1.1130, -2.1461}));

    GraphNode v1 = xs1 & h1w;
    GraphNode y1 = v1.tanh();
    GraphNode v2 = y1 & h2w;
    GraphNode y2 = v2.tanh();
    GraphNode v3 = y2 & ow;
    GraphNode y3 = v3.tanh();

    //y3.move_to_gpu();
    TensorPtr out1 = y3.eval();
    //out1->move_to_ram();
    printf("%s\n", out1->str().c_str());
    assert(check_memory_against_array({0.955728}, out1));
}

void test_network_XOR_with_pretrained_weights_gpu_eager()
{
    printf("test_network_XOR_with_pretrained_weights_gpu_eager\n");

    // activate eager mode and force gpu.
    GraphNode::set_eager_mode(true, true);

    // input [1, 0]
    GraphNode xs1(Tensor({2}, {0,1}));

    // hidden layer 1 (5 nodes)
    GraphNode h1w(Tensor({2,5}, {0.3015, 0.7785, 0.0818, 0.6411, -1.3131,
                                 0.3517, -0.9395, -1.2009, 1.0679, 0.4376}));

    // hidden layer 2 (4 nodes)
    GraphNode h2w(Tensor({5,4},
                    {1.5171,  1.2708,  1.3553, -0.3223,
                     1.4185,  0.2538, -0.9858, -1.7510,
                     -0.4906, -0.3851, -1.3264,  1.8427,
                     -0.5821, -0.7318, -0.6390, -2.0936,
                     -1.1953, -1.0868,  0.4595,  0.4198}));

    // output layer (1)
    GraphNode ow(Tensor({4, 1}, {-1.0577, -0.2073, -1.1130, -2.1461}));

    GraphNode v1 = xs1 & h1w;
    GraphNode y1 = v1.tanh();
    GraphNode v2 = y1 & h2w;
    GraphNode y2 = v2.tanh();
    GraphNode v3 = y2 & ow;
    GraphNode y3 = v3.tanh();

    std::ofstream of;
    of.open("graphs/network_XOR_gpu_eager.dot", std::ios::out | std::ios::trunc);
    y3.draw(of);
    of.close();

    TensorPtr out1 = y3.eval();
    out1->move_to_ram();
    printf("%s\n", out1->str().c_str());
    assert(check_memory_against_array({0.955728}, out1));

    GraphNode::set_eager_mode(false);
}

void test_derivative_sin()
{
    printf("test_derivative_sin\n");

    GraphNode a(Tensor({1}, {1.2}, false));
    a.set_require_grad(true);

    GraphNode c = a.sin();

    auto out = c.eval();

    c.backward();

    assert(a.grad() != nullptr);
    assert(within_acceptable_error(a.grad()->memory[0], 0.3624));
}

void test_derivative_scalar_mul()
{
    printf("test_derivative_scalar_mul\n");
    GraphNode a(Tensor({1}, {5.0}, false));
    GraphNode b(Tensor({1}, {3.0}, false));
    
    b.set_require_grad(true);

    GraphNode c = a * b;

    auto out = c.eval();

    c.backward();

    assert(b.grad() != nullptr);
    assert(b.grad()->memory[0] == 5.0);
    assert(a.grad() == nullptr);
}

void test_derivate_simple_scalar_mul_sin()
{
    printf("test_derivate_simple_scalar_mul_sin\n");
    GraphNode a(Tensor({1}, {5.0}, false));
    GraphNode b(Tensor({1}, {3.0}, false));
    
    b.set_require_grad(true);

    GraphNode c = (a * b).sin();

    auto out = c.eval();

    c.backward();

    assert(b.grad() != nullptr);
    assert(within_acceptable_error(b.grad()->memory[0], -3.7984));
    assert(a.grad() == nullptr);
}

void test_derivate_simple_scalar_add_sin()
{
    printf("test_derivate_simple_scalar_add_sin\n");
    GraphNode a(Tensor({1}, {5.0}, false));
    GraphNode x(Tensor({1}, {2.3}, false));
    GraphNode b(Tensor({1}, {3.0}, false));
    GraphNode c(Tensor({1}, {-2.3}, false));

    b.set_require_grad(true);
    x.set_require_grad(true);

    GraphNode r = (c * (a + x * b).sin()).sin();

    auto out = r.eval();

    r.backward();

    assert(a.grad() == nullptr);

    assert(b.grad() != nullptr);
    assert(within_acceptable_error(b.grad()->memory[0], -0.6176));

    assert(x.grad() != nullptr);
    assert(within_acceptable_error(x.grad()->memory[0], -0.8056));
}

void test_derivative_2()
{
    printf("test_derivative_2\n");

    GraphNode y(Tensor({1}, {1.0}, false));

    GraphNode x(Tensor({2}, {1.0, 0.0}));
    GraphNode w(Tensor({2}, {0.32, -0.49}));

    GraphNode z = x & w;

    GraphNode o = z.relu();

    GraphNode J = (y - o).pow(2.0) * 0.5f;

    J.eval();

    std::ofstream of;
    of.open("graphs/test_derivative_1.dot", std::ios::out | std::ios::trunc);
    J.draw(of);
    of.close();
}

int main(int argc, char **argv)
{
    printf("RUN %s\n", argv[0]);

    test_compnode_add_cpu();
    //test_compnode_add_gpu();

    test_compnode_dot_cpu();

    test_graphnodes_cpu();

    test_graphnodes_interface_cpu();
    //test_graphnodes_interface_gpu();

    test_graphnodes_min();
    test_graphnodes_minself();
    test_graphnodes_div();

    test_graphnodes_tanh_cpu();
    //test_graphnodes_tanh_gpu();

    test_graphnodes_pow_cpu();
    //test_graphnodes_pow_gpu();

    test_draw();
    test_network_XOR_with_pretrained_weights_cpu_lazy();
    test_network_XOR_with_pretrained_weights_cpu_eager();
    //test_network_XOR_with_pretrained_weights_gpu_lazy();
    //test_network_XOR_with_pretrained_weights_gpu_eager();

    test_derivative_sin();
    test_derivative_scalar_mul();
    test_derivate_simple_scalar_mul_sin();
    test_derivate_simple_scalar_add_sin();
    //test_derivative_1();

    printf("!!!!! ALL TESTS PASSED !!!!!\n");

    gpu_reset();

    return 0;
}