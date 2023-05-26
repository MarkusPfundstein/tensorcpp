#include "comp_graph.h"
#include <cstring>
#include <cassert>

#define NDEBUG
#define assertm(exp, msg) assert(((void)msg, exp))

void test_compnode_add_cpu()
{
    printf("test_compnode_add_cpu\n");

    TensorPtr a(new Tensor({4}, {1,2,3,4}));
    TensorPtr b(new Tensor({4}, {0.4,0.3,0.2,0.1}));

    CompNode_Add add(new TensorNode(a), new TensorNode(b));

    TensorPtr c = add.eval();

    std::vector<float> ref({1.4, 2.3, 3.2, 4.1});
    assertm(memcmp(c->memory, ref.data(), c->nelems * sizeof(float)) == 0, "add failed");
}

void test_compnode_mul_cpu()
{
    printf("test_compnode_mul_cpu\n");

    TensorPtr a(new Tensor({3}, {1,2,3}));
    TensorPtr b(new Tensor({3}, {2,3,4}));

    CompNode_Mul op(new TensorNode(a), new TensorNode(b));

    TensorPtr c = op.eval();

    float res = 1*2 + 2*3 + 3*4;
    assertm(res == c->memory[0], "mul op dot product failed");
}

#if 0
void test_compnode_add_gpu()
{
    printf("test_compnode_add_gpu\n");
    
    CompNode_Add add(
        Tensor({4}, {1,2,3,4}),
        Tensor({4}, {0.4,0.3,0.2,0.1})
    );

    add.move_to_gpu();
    add.compute();
    add.move_to_ram();

    const Tensor &c = add.get_output_ref();

    std::vector<float> ref({1.4, 2.3, 3.2, 4.1});
    assertm(memcmp(c.memory, ref.data(), c.nelems * sizeof(float)) == 0, "bla");
}

#endif

void test_graphnodes_cpu()
{
    printf("test_graphnodes_cpu\n");

    CompNode_Mul *mul = new CompNode_Mul(
        new TensorNode(TensorPtr(new Tensor({1}, {5}, false))),
        new TensorNode(TensorPtr(new Tensor({3}, {1,2,3})))
    );

    CompNode_Add add(
        mul,
        new TensorNode(TensorPtr(new Tensor({3}, {6,7,8})))
    );

    TensorPtr out = add.eval();

    assertm(out->memory[0] == 11, "out[0] wrong");
    assertm(out->memory[1] == 17, "out[1] wrong");
    assertm(out->memory[2] == 23, "out[2] wrong");
}

int main(int argc, char **argv)
{
    printf("RUN %s\n", argv[0]);

    test_compnode_add_cpu();
    test_compnode_mul_cpu();

    test_graphnodes_cpu();

    printf("!!!!! ALL TESTS PASSED !!!!!\n");

    return 0;
}