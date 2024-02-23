#include <tvm/ir/module.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>
#include <tvm/support/with.h>
#include <tvm/target/target.h>
#include <tvm/te/operation.h>
#include <tvm/te/tensor.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/var.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

using namespace tvm;
using namespace tvm::runtime;
using namespace tvm::tir;
using namespace tvm::te;

int main(int argc, char** argv) {
    // create prim func
    const tvm::runtime::PackedFunc* create_prim_func = tvm::runtime::Registry::Get("te.CreatePrimFunc");
    if (!create_prim_func) {
        std::cerr << "te.CreatePrimFunc expression not found" << std::endl;
        return -1;
    }

    // the script gen function
    const tvm::runtime::PackedFunc* script_gen = tvm::runtime::Registry::Get("node.TVMScriptPrinterScript");
    if (!script_gen) {
        std::cerr << "node.TVMScriptPrinterScript expression not found" << std::endl;
        return -1;
    }

    // c code gen
    const tvm::runtime::PackedFunc* code_gen_c = tvm::runtime::Registry::Get("target.build.c");
    if (!code_gen_c) {
        std::cerr << "target.build.c expression not found" << std::endl;
        return -1;
    }

    // generate input tensor: a
    Array<PrimExpr> input_shape{8};
    te::Tensor tensor_a = tvm::te::placeholder(input_shape, DataType::Float(32), "A");

    Range range(0, 8);
    tvm::DataType dtype = range->extent.dtype();
    tvm::tir::Var var("i", dtype);
    tvm::tir::IterVar iter(range, var, IterVarType::kDataPar);

    ProducerLoad loader(tensor_a, {iter->var});
    PrimExpr const_val(1.0f);

    // b = a + 1.0
    PrimExpr b = add(loader, const_val);

    Array<PrimExpr> body{b};
    Array<IterVar> dim_var{iter};

    ComputeOp compute("compute", "", {}, dim_var, body);
    int num_outputs = compute->num_outputs();
    te::Tensor tensor_b = compute.output(0);
    Array<te::Tensor> tensors{tensor_a, tensor_b};

    PrimFunc func = (*create_prim_func)(tensors, nullptr);
    tvm::IRModule mod = IRModule::FromExpr(func);

    std::string result = (*script_gen)(mod, nullptr);
    std::cout << "tir script: " << std::endl << result << std::endl << std::endl;

    runtime::Module module_c = (*code_gen_c)(mod, Target("c"));
    return 0;
}