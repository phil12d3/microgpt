#pragma once

#include "microgpt/backend.hpp"

#include <string>

namespace microgpt {

struct ChatConfig {
  std::string checkpoint;
  int max_new_tokens = 0;
  float temperature = 0.0f;
  int top_k = 0;
  BackendKind backend = BackendKind::Cpu;
};

}  // namespace microgpt
