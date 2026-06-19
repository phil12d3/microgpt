#pragma once

#include "microgpt/backend.hpp"
#include "microgpt/cli_shared.hpp"

#include <ostream>
#include <vector>

namespace microgpt {

inline BackendKind get_backend_arg(const std::vector<std::string>& args) {
  return parse_backend_kind(get_arg(args, "--backend", "cpu"));
}

inline BackendKind require_backend_arg(const std::vector<std::string>& args) {
  BackendKind backend = get_backend_arg(args);
  require_backend_available(backend);
  return backend;
}

inline void print_backend_selection(const std::vector<std::string>& args, std::ostream& out) {
  BackendKind backend = get_backend_arg(args);
  require_backend_available(backend);
  out << "backend " << backend_name(backend) << '\n';
  if (backend != BackendKind::Cpu) {
    out << "backend_detail " << backend_detail(backend) << '\n';
    out << "backend_note accelerated kernels are incomplete; unsupported operations use the CPU fallback path\n";
  }
}

inline int run_backends_command(const std::vector<std::string>&, std::ostream& out) {
  out << "compiled_acceleration_backend " << microgpt_compiled_acceleration_backend() << '\n';
  for (BackendKind backend : known_backends()) {
    out << "backend " << backend_name(backend) << " available " << (backend_available(backend) ? "yes" : "no");
    std::string detail = backend_detail(backend);
    if (!detail.empty()) {
      out << " detail \"" << detail << '"';
    }
    out << '\n';
  }
  return 0;
}

}  // namespace microgpt
