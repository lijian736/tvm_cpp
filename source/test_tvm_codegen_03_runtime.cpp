#include <tvm/ir/module.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/runtime.h>
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
    // get the list-runtimes packed function
    const tvm::runtime::PackedFunc* list_runtimes = tvm::runtime::Registry::Get("relay.backend.ListRuntimes");
    if (!list_runtimes) {
        std::cerr << "relay.backend.ListRuntimes not found" << std::endl;
        return -1;
    }

    // get the create-runtime packed function
    const tvm::runtime::PackedFunc* create_runtime = tvm::runtime::Registry::Get("relay.backend.CreateRuntime");
    if (!create_runtime) {
        std::cerr << "relay.backend.CreateRuntime not found" << std::endl;
        return -1;
    }

    // get the list-runtime-options packed function
    const tvm::runtime::PackedFunc* runtime_options = tvm::runtime::Registry::Get("relay.backend.ListRuntimeOptions");
    if (!runtime_options) {
        std::cerr << "relay.backend.ListRuntimeOptions not found" << std::endl;
        return -1;
    }

    // get the list-runtime-attrs packed function
    const tvm::runtime::PackedFunc* runtime_attrs = tvm::runtime::Registry::Get("relay.backend.GetRuntimeAttrs");
    if (!runtime_attrs) {
        std::cerr << "relay.backend.GetRuntimeAttrs not found" << std::endl;
        return -1;
    }

    Array<String> runtimes = (*list_runtimes)();
    for (size_t i = 0; i < runtimes.size(); ++i) {
        std::cout << "runtime " << i << " info:" << std::endl;
        std::cout << "name:" << runtimes[i] << std::endl;
        // create the runtime
        tvm::relay::Runtime runtime = (*create_runtime)(runtimes[i], Map<String, ObjectRef>());
        // get the runtime attributes
        Map<String, ObjectRef> attrs = (*runtime_attrs)(runtime);
        std::cout << "attribute name list:" << std::endl;
        for (auto itr = attrs.begin(); itr != attrs.end(); ++itr) {
            std::cout << (*itr).first << std::endl;
        }

        Map<String, String> options = (*runtime_options)(runtimes[i]);
        std::cout << "options list:" << std::endl;
        for (auto itr = options.begin(); itr != options.end(); ++itr) {
            std::cout << (*itr).first << " - " << (*itr).second << std::endl;
        }

        std::cout << std::endl;
    }

    return 0;
}