#include <iostream>
#include <string>
#include <vector>

#include "microgpt/harness.hpp"
#include "microgpt/tools.hpp"

int main() {
  microgpt::HarnessRegistry registry;
  microgpt::register_microgpt_tools(registry);
  microgpt::HarnessIO io{std::cout, std::cerr};
  return registry.run("validate-data", {"--input", "data.txt"}, io);
}
