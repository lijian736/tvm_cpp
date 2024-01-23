#include <tvm/ir/module.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

using namespace tvm;
using namespace tvm::relay;
using namespace tvm::relay::transform;
using namespace tvm::runtime;

int main(int argc, char** argv) {
    // Var node in relay
    const tvm::runtime::PackedFunc* var_gen = tvm::runtime::Registry::Get("relay.ir.Var");
    if (!var_gen) {
        std::cerr << "relay.ir.Var expression not found" << std::endl;
        return -1;
    }

    // Resize 2d
    const tvm::runtime::PackedFunc* resize2d = tvm::runtime::Registry::Get("relay.op.image._make.resize2d");
    if (!resize2d) {
        std::cerr << "relay.op.image._make.resize2d expression not found" << std::endl;
        return -1;
    }

    // type infer
    const tvm::runtime::PackedFunc* type_infer = tvm::runtime::Registry::Get("relay._transform.InferType");
    if (!type_infer) {
        std::cerr << "relay._transform.InferType expression not found" << std::endl;
        return -1;
    }

    // pass run
    const tvm::runtime::PackedFunc* pass_run = tvm::runtime::Registry::Get("transform.RunPass");
    if (!pass_run) {
        std::cerr << "transform.pass_run expression not found" << std::endl;
        return -1;
    }

    std::string name = "input var";
    tvm::relay::Span span;

    // the data type
    DLDataType data_type = {DLDataTypeCode::kDLFloat, 32, 1};
    // the tensor type
    tvm::relay::TensorType tensor_type{{1, 128, 20, 20}, tvm::DataType{data_type}};

    // the var expression
    tvm::relay::Var var_expr = (*var_gen)(name, tensor_type, span);

    tvm::FloatImm roi_v(DataType{data_type}, 0.0f);
    tvm::runtime::Array<tvm::relay::IndexExpr> size({40, 40});
    tvm::runtime::Array<tvm::FloatImm> roi{roi_v, roi_v, roi_v, roi_v};
    tvm::runtime::String layout = "NCHW";
    tvm::runtime::String method = "nearest_neighbor";
    tvm::runtime::String coord = "asymmetric";
    tvm::runtime::String rounding = "floor";
    double cubic_alpha = -0.75;
    int cubic_exclude = 0;
    double extrapolation_value = 0.0;
    DataType out_type;

    tvm::relay::Expr result = (*resize2d)(var_expr, size, roi, layout, method, coord, rounding, cubic_alpha,
                                          cubic_exclude, extrapolation_value, out_type);

    // infer type
    tvm::IRModule mod = IRModule::FromExpr(result);
    tvm::relay::transform::Pass infer_type_pass = (*type_infer)();
    IRModule mod_new = (*pass_run)(infer_type_pass, mod);

    tvm::relay::Expr expr = mod_new->Lookup("main").as<FunctionNode>()->body;
    tvm::Type result_type = expr->checked_type();

    tvm::runtime::Optional<tvm::TensorType> type = result_type.as<tvm::TensorType>();
    if (type != nullptr) {
        tvm::TensorType tensor_type = type.value();

        const tvm::DataType& data_type = tensor_type->dtype;
        const tvm::runtime::Array<tvm::PrimExpr>& shape = tensor_type->shape;

        std::cout << "DataType: " << data_type << ", Shape: ";
        std::cout << "{";
        for (int i = 0; i < shape.size(); ++i) {
            const tvm::PrimExpr& exp = shape[i];
            const IntImmNode* node = exp.as<IntImmNode>();
            if (i) {
                std::cout << ", ";
            }
            if (node) {
                std::cout << node->value;
            } else {
                std::cout << "?";
            }
        }
        std::cout << "}" << std::endl;
    }

    return 0;
}