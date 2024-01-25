#include <tvm/ir/module.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>
#include <tvm/support/with.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

using namespace tvm;
using namespace tvm::relay;
using namespace tvm::relay::transform;
using namespace tvm::runtime;

int main(int argc, char** argv) {
     //Get all the registerd relay objects
    std::vector<String> registerd_names = Registry::ListNames();
    
    std::vector<std::string> transform_names;
    std::for_each(registerd_names.cbegin(), registerd_names.cend(), [&](const auto& item) {
        std::string name = (std::string)item;

        if (std::string::npos != name.find("transform.")) {
            transform_names.emplace_back(name);
        }
    });

    for (auto& name : transform_names) {
        std::cout << name << std::endl;
    }

    std::cout << "----------------------------------------" << std::endl;
    std::cout << "total: " << transform_names.size() << std::endl;

    return 0;
}