#include "status.h"

namespace tvm_cpp {
namespace common {

Status::Status(StatusCode code, const std::string& msg) : m_code(code), m_message(msg) {}
Status::Status(StatusCode code, const char* msg) : m_code(code), m_message(msg) {}
Status::Status(StatusCode code) : m_code(code), m_message("") {}

bool Status::operator==(const Status& rhs) const { return m_code == rhs.m_code && m_message == rhs.m_message; }
bool Status::operator!=(const Status& rhs) const { return !(*this == rhs); }

StatusCode Status::code() const { return m_code; }
const std::string& Status::message() const { return m_message; }

bool Status::is_ok() const { return m_code == StatusCode::OK; }

std::string Status::to_string() const {
    std::string result = std::string(statuscode_to_string(code())) + ":" + message();
    return result;
}

Status Status::ok() { return Status(); }

}    // namespace common
}    // namespace tvm_cpp