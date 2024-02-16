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
    // get the list-executors packed function
    const tvm::runtime::PackedFunc* list_executors = tvm::runtime::Registry::Get("relay.backend.ListExecutors");
    if (!list_executors) {
        std::cerr << "relay.backend.ListExecutors not found" << std::endl;
        return -1;
    }

    // get the create-executor packed function
    const tvm::runtime::PackedFunc* create_executor = tvm::runtime::Registry::Get("relay.backend.CreateExecutor");
    if (!create_executor) {
        std::cerr << "relay.backend.CreateExecutor not found" << std::endl;
        return -1;
    }

    // get the list-executor-options packed function
    const tvm::runtime::PackedFunc* executor_options = tvm::runtime::Registry::Get("relay.backend.ListExecutorOptions");
    if (!executor_options) {
        std::cerr << "relay.backend.ListExecutorOptions not found" << std::endl;
        return -1;
    }

    // get the list-executor-attrs packed function
    const tvm::runtime::PackedFunc* executor_attrs = tvm::runtime::Registry::Get("relay.backend.GetExecutorAttrs");
    if (!executor_attrs) {
        std::cerr << "relay.backend.GetExecutorAttrs not found" << std::endl;
        return -1;
    }

    Array<String> executors = (*list_executors)();
    for (size_t i = 0; i < executors.size(); ++i) {
        std::cout << "executor " << i << " info:" << std::endl;
        std::cout << "name:" << executors[i] << std::endl;
        // create the executor
        tvm::relay::Executor executor = (*create_executor)(executors[i], Map<String, ObjectRef>());
        // get the executor attributes
        Map<String, ObjectRef> attrs = (*executor_attrs)(executor);
        std::cout << "attribute name list:" << std::endl;
        for (auto itr = attrs.begin(); itr != attrs.end(); ++itr) {
            std::cout << (*itr).first << std::endl;
        }

        Map<String, String> options = (*executor_options)(executors[i]);
        std::cout << "options list:" << std::endl;
        for (auto itr = options.begin(); itr != options.end(); ++itr) {
            std::cout << (*itr).first << " - " << (*itr).second << std::endl;
        }

        std::cout << std::endl;
    }

    return 0;
}