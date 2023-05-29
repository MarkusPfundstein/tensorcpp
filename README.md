# tensorcpp

tensorcpp is a Tensor library with autograd. It has a cuda and cpu backend. It is an educational program and should be treated as such.

Right now, it will only compile on Linux and with CUDA GPUs of architecture compute_50 and sm_50.

## Building & Running

- `./build.bash` for compiling and linking
- `./build/tensor_test` for tensor class related unit tests
- `./build/comp_graph_test` for computation graph related unit tests

## Example

### Raw tensor ops

```
Tensor t1({3,2}, {0.01, 0.32, -3.43, 3.2, 4.8, 0.0002});
Tensor t2({3}, {0,1,0});

t1.move_to_gpu();
t2.move_to_gpu();
Tensor r1 = t2 & t1;
r1.move_to_ram();

assert(within_acceptable_error(r1.memory[0], -3.43));
assert(within_acceptable_error(r1.memory[1], 3.2));
```

### Pre-trained feed forward network using computation graph

```
// eager mode on GPU for immediate computations
GraphNode::set_eager_mode(true, true);

// input [1, 0]
GraphNode xs1(Tensor({2}, {0,1}));

// hidden layer 1 (5 nodes)
GraphNode h1w(Tensor({2,5}, { 0.3015, 0.7785, 0.0818, 0.6411, -1.3131,
                              0.3517, -0.9395, -1.2009, 1.0679, 0.4376}));

// hidden layer 2 (4 nodes)
GraphNode h2w(Tensor({5,4}, { 1.5171,  1.2708,  1.3553, -0.3223,
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

TensorPtr out1 = y3.eval();
out1->move_to_ram();
printf("%s\n", out1->str().c_str());
assert(check_memory_against_array({0.955728}, out1));
```

### Autograd using computation graph

```
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
```

## LICENSE

GNU
