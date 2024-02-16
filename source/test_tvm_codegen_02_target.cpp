#include <tvm/ir/module.h>
#include <tvm/relay/executor.h>
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
    // get the current target packed function
    const tvm::runtime::PackedFunc* target_current = tvm::runtime::Registry::Get("target.TargetCurrent");
    if (!target_current) {
        std::cerr << "target.TargetCurrent not found" << std::endl;
        return -1;
    }

    tvm::Target target = (*target_current)(true);

    return 0;
}