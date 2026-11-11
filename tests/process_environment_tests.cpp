#include "strix/process_environment.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {
int failures = 0;
void expect(bool condition, const char* what) {
    if (!condition) { std::cerr << "FAIL: " << what << '\n'; ++failures; }
}
std::string value(const char* name) {
    const char* v = std::getenv(name);
    return v ? v : "<unset>";
}
}

int main() {
    namespace env = strix::process_environment;
    const char* name = "STRIX_PROCESS_ENVIRONMENT_TEST";
    expect(env::unset(name), "unset of an absent variable succeeds");
    expect(value(name) == "<unset>", "variable starts unset");

    expect(env::set(name, "first", false), "set without overwrite on an absent variable");
    expect(value(name) == "first", "absent variable takes the value");

    expect(env::set(name, "second", false), "set without overwrite on a present variable reports success");
    expect(value(name) == "first", "present variable keeps its value without overwrite");

    expect(env::set(name, "third", true), "set with overwrite");
    expect(value(name) == "third", "overwrite replaces the value");

    expect(env::unset(name), "unset of a present variable");
    expect(value(name) == "<unset>", "variable is gone after unset");

    if (failures) return 1;
    std::cout << "process environment: set, keep, overwrite and unset agree\n";
}
