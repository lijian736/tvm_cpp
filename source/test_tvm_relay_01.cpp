#include <tvm/runtime/memory.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

using namespace tvm;
using namespace tvm::runtime;


int main(int argc, char** argv) {

    //Get all the registerd relay objects
    std::vector<String> registerd_names = Registry::ListNames();
    
    std::vector<std::string> relay_names;
    std::for_each(registerd_names.cbegin(), registerd_names.cend(), [&](const auto& item) {
        std::string name = (std::string)item;

        if (std::string::npos != name.find("relay")) {
            relay_names.emplace_back(name);
        }
    });

    for (auto& name : relay_names) {
        std::cout << name << std::endl;
    }

    std::cout << "----------------------------------------" << std::endl;
    std::cout << "total: " << relay_names.size() << std::endl;

    return 0;
}