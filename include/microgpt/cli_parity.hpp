#pragma once

#include "microgpt/backend.hpp"
#include "microgpt/cli_shared.hpp"
#include "microgpt/lab_tests.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace microgpt {

inline int run_parity_command(const std::vector<std::string>& args, std::ostream& out, std::ostream& err) {
  (void)err;
  if (has_arg(args, "--backend")) {
    BackendKind backend = parse_backend_kind(get_arg(args, "--backend", "metal"));
    if (backend != BackendKind::Metal) {
      throw std::runtime_error("--backend for parity must be metal");
    }
  }
  if (!microgpt_metal_runtime_available()) {
    out << "parity_report unavailable\n";
    out << "reason no_runtime_metal_device\n";
    return 2;
  }
  bool ok = cpu_metal_staged_parity_trace_report(out);
  return ok ? 0 : 2;
}

}  // namespace microgpt
