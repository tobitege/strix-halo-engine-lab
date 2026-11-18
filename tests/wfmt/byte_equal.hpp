#pragma once
#include <cstring>
#include <type_traits>
#include <vector>
namespace strix::weight::test {
template <class T> inline bool byte_equal(const std::vector<T>& a,const std::vector<T>& b) {
    static_assert(std::is_trivially_copyable_v<T>,"byte_equal compares object representations");
    return a.size()==b.size() && (a.empty() || std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0);
}
// Non-template overload keeps brace-initialised float calls valid.
inline bool byte_equal(const std::vector<float>& a,const std::vector<float>& b) { return byte_equal<float>(a,b); }
}
