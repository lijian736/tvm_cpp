#include "utils.h"

#include <algorithm>
#include <filesystem>
#include <iostream>

namespace tvm_cpp {
namespace utils {

void trim_start(std::string& str) {
    str.erase(str.begin(), std::find_if(str.begin(), str.end(), [](int ch) { return !std::isspace(ch); }));
}

void trim_end(std::string& str) {
    str.erase(std::find_if(str.rbegin(), str.rend(), [](int ch) { return !std::isspace(ch); }).base(), str.end());
}

void trim(std::string& str) {
    // trim end
    str.erase(std::find_if(str.rbegin(), str.rend(), [](int ch) { return !std::isspace(ch); }).base(), str.end());
    // trim start
    str.erase(str.begin(), std::find_if(str.begin(), str.end(), [](int ch) { return !std::isspace(ch); }));
}

bool ends_with(const std::string& str, const std::string& suffix) {
    if (suffix.length() > str.length()) {
        return false;
    }

    return (str.rfind(suffix) == (str.length() - suffix.length()));
}

bool file_exist(const std::string& file_path) {
    std::filesystem::path path = file_path;
    return std::filesystem::exists(path);
}

}    // namespace utils
}    // namespace tvm_cpp