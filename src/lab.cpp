#include "microgpt.hpp"

#include <iostream>

int main() {
  std::cout << "microgpt lab\n";
  bool ok = microgpt::run_tests();
  std::cout << "lab_status " << (ok ? "pass" : "fail") << '\n';
  return ok ? 0 : 2;
}
