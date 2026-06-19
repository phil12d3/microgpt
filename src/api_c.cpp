#include "microgpt/api_c.h"

#include "microgpt/api.hpp"

#include <cstring>
#include <exception>
#include <memory>
#include <string>

struct microgpt_model {
  explicit microgpt_model(microgpt::api::ModelHandle loaded) : handle(std::move(loaded)) {}
  microgpt::api::ModelHandle handle;
};

namespace {
thread_local std::string last_error;

void clear_error() {
  last_error.clear();
}

void capture_error(const std::exception& e) {
  last_error = e.what();
}

void capture_unknown_error() {
  last_error = "unknown microgpt error";
}

char* copy_string_for_c(const std::string& text) {
  char* out = new char[text.size() + 1];
  std::memcpy(out, text.c_str(), text.size() + 1);
  return out;
}
}  // namespace

extern "C" microgpt_generation_options microgpt_default_generation_options(void) {
  microgpt_generation_options options;
  options.max_new_tokens = 200;
  options.temperature = 0.8f;
  options.top_k = 20;
  options.stop_token_id = microgpt::Tokenizer::kEos;
  return options;
}

extern "C" microgpt_model* microgpt_load_model(const char* checkpoint_path, const char* backend_name) {
  clear_error();
  if (checkpoint_path == nullptr) {
    last_error = "checkpoint_path is required";
    return nullptr;
  }
  try {
    microgpt::BackendKind backend = microgpt::BackendKind::Cpu;
    if (backend_name != nullptr && backend_name[0] != '\0') {
      backend = microgpt::parse_backend_kind(backend_name);
    }
    return new microgpt_model(microgpt::api::load_model(checkpoint_path, backend));
  } catch (const std::exception& e) {
    capture_error(e);
  } catch (...) {
    capture_unknown_error();
  }
  return nullptr;
}

extern "C" void microgpt_free_model(microgpt_model* model) {
  delete model;
}

extern "C" char* microgpt_generate_text(microgpt_model* model, const char* prompt,
                                         microgpt_generation_options options) {
  clear_error();
  if (model == nullptr) {
    last_error = "model is required";
    return nullptr;
  }
  if (prompt == nullptr) {
    last_error = "prompt is required";
    return nullptr;
  }
  try {
    microgpt::api::GenerationOptions cpp_options;
    cpp_options.max_new_tokens = options.max_new_tokens;
    cpp_options.temperature = options.temperature;
    cpp_options.top_k = options.top_k;
    cpp_options.stop_token_id = options.stop_token_id;
    return copy_string_for_c(microgpt::api::generate(model->handle, prompt, cpp_options));
  } catch (const std::exception& e) {
    capture_error(e);
  } catch (...) {
    capture_unknown_error();
  }
  return nullptr;
}

extern "C" void microgpt_free_string(char* text) {
  delete[] text;
}

extern "C" const char* microgpt_last_error(void) {
  return last_error.c_str();
}
