#include <tvm/ir/module.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>
#include <tvm/script/ir_builder/base.h>
#include <tvm/script/ir_builder/tir/frame.h>
#include <tvm/support/with.h>
#include <tvm/tir/function.h>
#include <tvm/tir/var.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

using namespace tvm;
using namespace tvm::runtime;
using namespace tvm::script::ir_builder::tir;
using namespace tvm::script::ir_builder;
using namespace tvm::tir;

int main(int argc, char** argv) {
    // get the prim func packed function
    const tvm::runtime::PackedFunc* prim_func = tvm::runtime::Registry::Get("script.ir_builder.tir.PrimFunc");
    if (!prim_func) {
        std::cerr << "script.ir_builder.tir.PrimFunc expression not found" << std::endl;
        return -1;
    }

    // function attributes
    const tvm::runtime::PackedFunc* attrs_gen = tvm::runtime::Registry::Get("script.ir_builder.tir.FuncAttrs");
    if (!attrs_gen) {
        std::cerr << "script.ir_builder.tir.FuncAttrs expression not found" << std::endl;
        return -1;
    }

    // the Var gen function
    const tvm::runtime::PackedFunc* var_gen = tvm::runtime::Registry::Get("script.ir_builder.tir.Handle");
    if (!var_gen) {
        std::cerr << "script.ir_builder.tir.Handle expression not found" << std::endl;
        return -1;
    }

    // the Var set function
    const tvm::runtime::PackedFunc* set_var = tvm::runtime::Registry::Get("script.ir_builder.tir.Arg");
    if (!set_var) {
        std::cerr << "script.ir_builder.tir.Arg expression not found" << std::endl;
        return -1;
    }

    // the buffer gen function
    const tvm::runtime::PackedFunc* buffer_gen = tvm::runtime::Registry::Get("script.ir_builder.tir.MatchBuffer");
    if (!buffer_gen) {
        std::cerr << "script.ir_builder.tir.MatchBuffer expression not found" << std::endl;
        return -1;
    }

    // the serial gen function
    const tvm::runtime::PackedFunc* serial_gen = tvm::runtime::Registry::Get("script.ir_builder.tir.Serial");
    if (!serial_gen) {
        std::cerr << "script.ir_builder.tir.Serial expression not found" << std::endl;
        return -1;
    }

    // the block gen function
    const tvm::runtime::PackedFunc* block_gen = tvm::runtime::Registry::Get("script.ir_builder.tir.Block");
    if (!block_gen) {
        std::cerr << "script.ir_builder.tir.Block expression not found" << std::endl;
        return -1;
    }

    // the axis gen function
    const tvm::runtime::PackedFunc* axis_gen = tvm::runtime::Registry::Get("script.ir_builder.tir.AxisSpatial");
    if (!axis_gen) {
        std::cerr << "script.ir_builder.tir.AxisSpatial expression not found" << std::endl;
        return -1;
    }

    // the reads gen function
    const tvm::runtime::PackedFunc* reads_gen = tvm::runtime::Registry::Get("script.ir_builder.tir.Reads");
    if (!reads_gen) {
        std::cerr << "script.ir_builder.tir.Reads expression not found" << std::endl;
        return -1;
    }

    // the writes gen function
    const tvm::runtime::PackedFunc* writes_gen = tvm::runtime::Registry::Get("script.ir_builder.tir.Writes");
    if (!writes_gen) {
        std::cerr << "script.ir_builder.tir.Writes expression not found" << std::endl;
        return -1;
    }

    // buffer load
    const tvm::runtime::PackedFunc* buffer_load = tvm::runtime::Registry::Get("tir.BufferLoad");
    if (!buffer_load) {
        std::cerr << "tir.BufferLoad expression not found" << std::endl;
        return -1;
    }

    // the float32 gen function
    const tvm::runtime::PackedFunc* float32_gen = tvm::runtime::Registry::Get("script.ir_builder.tir.Float32");
    if (!float32_gen) {
        std::cerr << "script.ir_builder.tir.Float32 expression not found" << std::endl;
        return -1;
    }

    // add op
    const tvm::runtime::PackedFunc* add_op = tvm::runtime::Registry::Get("tir._OpAdd");
    if (!add_op) {
        std::cerr << "tir._OpAdd expression not found" << std::endl;
        return -1;
    }

    // buffer store
    const tvm::runtime::PackedFunc* buffer_store = tvm::runtime::Registry::Get("script.ir_builder.tir.BufferStore");
    if (!buffer_store) {
        std::cerr << "script.ir_builder.tir.BufferStore expression not found" << std::endl;
        return -1;
    }

    // the script gen function
    const tvm::runtime::PackedFunc* script_gen = tvm::runtime::Registry::Get("node.TVMScriptPrinterScript");
    if (!script_gen) {
        std::cerr << "node.TVMScriptPrinterScript expression not found" << std::endl;
        return -1;
    }

    // Pretty print
    const tvm::runtime::PackedFunc* pretty_print = tvm::runtime::Registry::Get("relay.ir.PrettyPrint");
    if (!pretty_print) {
        std::cerr << "relay.ir.PrettyPrint expression not found" << std::endl;
        return -1;
    }

    // generate prim function frame
    IRBuilder ir_builder;
    {
        With<IRBuilder> _0(ir_builder);
        With<PrimFuncFrame> _1((*prim_func)(false));

        // set function attributes
        Map<String, ObjectRef> attrs;
        attrs.Set("global_symbol", String("main"));
        attrs.Set("tir.noalias", Bool(true));

        // // set function attributes
        (*attrs_gen)(attrs);

        // set function vars
        Var var_a = (*var_gen)(runtime::DataType::Void(), "global", false, false);
        Var var_b = (*var_gen)(runtime::DataType::Void(), "global", false, false);

        var_a = (*set_var)(String("a"), var_a);
        var_b = (*set_var)(String("b"), var_b);

        // the buffers for vars
        Array<PrimExpr> shape_a{8};
        Array<PrimExpr> shape_b{8};
        Buffer buffer_a = (*buffer_gen)(var_a, shape_a, DataType::Float(32), Optional<Var>(tvm::NullOpt),
                                        Array<PrimExpr>(), PrimExpr(), "global", -1, 0, "default", Array<IntImm>());

        Buffer buffer_b = (*buffer_gen)(var_b, shape_b, DataType::Float(32), Optional<Var>(tvm::NullOpt),
                                        Array<PrimExpr>(), PrimExpr(), "global", -1, 0, "default", Array<IntImm>());

        // the for loop
        ForFrame serial = (*serial_gen)(PrimExpr(0), PrimExpr(8), nullptr);
        With<ForFrame> for_loop(std::move(serial));
        {
            BlockFrame block = (*block_gen)(String("B"), false);
            With<BlockFrame> block_b(std::move(block));
            {
                Var it = (*axis_gen)(for_loop.get()->get()->doms[0], for_loop.get()->get()->vars[0], DataType::Int(32));
                (*reads_gen)(Array<ObjectRef>{(*buffer_load)(buffer_a, Array<PrimExpr>{it}, tvm::Span())});
                (*writes_gen)(Array<ObjectRef>{(*buffer_load)(buffer_b, Array<PrimExpr>{it}, tvm::Span())});
                PrimExpr const_val = (*float32_gen)(PrimExpr(1.0f), false);

                auto v1 = (*buffer_load)(buffer_b, Array<PrimExpr>{it}, tvm::Span());
                auto v2 = (*buffer_load)(buffer_a, Array<PrimExpr>{it}, tvm::Span());
                auto val = (*add_op)(v2, const_val, tvm::Span());
                (*buffer_store)(buffer_b, val, Array<PrimExpr>{it});
            }
        }
    }

    PrimFunc prim_func_expr = ir_builder->Get<PrimFunc>();
    tvm::IRModule mod = IRModule::FromExpr(prim_func_expr);

    std::string result = (*script_gen)(mod, nullptr);
    std::cout << "tir script: " << std::endl << result << std::endl << std::endl;

    tvm::String result_text = (*pretty_print)(mod);
    std::cout << "tir text: " << std::endl << result_text << std::endl;
    return 0;
}