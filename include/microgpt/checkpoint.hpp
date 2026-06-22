#pragma once

#include "microgpt/io.hpp"
#include "microgpt/optim.hpp"
#include "microgpt/tokenizer.hpp"

#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace microgpt {
inline std::string json_escape(const std::string& s) {
  std::ostringstream out;
  for (char c : s) {
    switch (c) {
      case '\\':
        out << "\\\\";
        break;
      case '"':
        out << "\\\"";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        out << c;
        break;
    }
  }
  return out.str();
}

inline std::string checkpoint_json_path(const std::string& checkpoint) {
  size_t dot = checkpoint.find_last_of('.');
  if (dot == std::string::npos) {
    return checkpoint + ".json";
  }
  return checkpoint.substr(0, dot) + ".json";
}

inline void write_u64(std::ostream& out, uint64_t v) { out.write(reinterpret_cast<const char*>(&v), sizeof(v)); }

inline void write_i32(std::ostream& out, int32_t v) { out.write(reinterpret_cast<const char*>(&v), sizeof(v)); }

inline void write_f32(std::ostream& out, float v) { out.write(reinterpret_cast<const char*>(&v), sizeof(v)); }

inline uint64_t read_u64(std::istream& in) {
  uint64_t v{};
  in.read(reinterpret_cast<char*>(&v), sizeof(v));
  return v;
}

inline int32_t read_i32(std::istream& in) {
  int32_t v{};
  in.read(reinterpret_cast<char*>(&v), sizeof(v));
  return v;
}

inline float read_f32(std::istream& in) {
  float v{};
  in.read(reinterpret_cast<char*>(&v), sizeof(v));
  return v;
}

inline void write_string(std::ostream& out, const std::string& s) {
  write_u64(out, static_cast<uint64_t>(s.size()));
  out.write(s.data(), static_cast<std::streamsize>(s.size()));
}

inline std::string read_string(std::istream& in) {
  uint64_t n = read_u64(in);
  std::string s(static_cast<size_t>(n), '\0');
  in.read(s.data(), static_cast<std::streamsize>(n));
  return s;
}

inline void write_shape(std::ostream& out, const std::vector<int>& shape) {
  write_u64(out, static_cast<uint64_t>(shape.size()));
  for (int v : shape) {
    write_i32(out, static_cast<int32_t>(v));
  }
}

inline std::vector<int> read_shape(std::istream& in) {
  uint64_t n = read_u64(in);
  std::vector<int> shape(static_cast<size_t>(n));
  for (size_t i = 0; i < shape.size(); ++i) {
    shape[i] = static_cast<int>(read_i32(in));
  }
  return shape;
}

inline void write_vector_f32(std::ostream& out, const std::vector<float>& v) {
  write_u64(out, static_cast<uint64_t>(v.size()));
  if (!v.empty()) {
    out.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(float)));
  }
}

inline std::vector<float> read_vector_f32(std::istream& in) {
  uint64_t n = read_u64(in);
  std::vector<float> v(static_cast<size_t>(n));
  if (!v.empty()) {
    in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(float)));
  }
  return v;
}

inline void save_checkpoint(const std::string& path, const Model& model, const AdamW& opt, int step) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("failed to open checkpoint for write: " + path);
  }
  out.write("MICROGPT2", 9);
  write_i32(out, step);
  write_i32(out, model.cfg.tokenizer_kind);
  write_i32(out, model.cfg.vocab_size);
  write_i32(out, model.cfg.context_length);
  write_i32(out, model.cfg.d_model);
  write_i32(out, model.cfg.num_layers);
  write_i32(out, model.cfg.num_heads);
  write_i32(out, model.cfg.d_ff);
  write_i32(out, model.cfg.batch_size);
  write_f32(out, model.cfg.learning_rate);
  write_f32(out, model.cfg.beta1);
  write_f32(out, model.cfg.beta2);
  write_f32(out, model.cfg.adam_eps);
  write_f32(out, model.cfg.weight_decay);
  write_f32(out, model.cfg.max_grad_norm);
  write_i32(out, opt.step);
  auto params = const_cast<Model&>(model).parameters();
  write_u64(out, static_cast<uint64_t>(params.size()));
  for (Parameter* p : params) {
    write_string(out, p->name);
    write_shape(out, p->shape);
    write_vector_f32(out, p->data);
    write_vector_f32(out, p->m);
    write_vector_f32(out, p->v);
    write_u64(out, p->decay ? 1 : 0);
  }
  out.close();
}

inline Model load_checkpoint(const std::string& path, AdamW& opt, int& step) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open checkpoint for read: " + path);
  }
  char magic[10] = {};
  in.read(magic, 9);
  std::string magic_text(magic, 9);
  if (magic_text != "MICROGPT1" && magic_text != "MICROGPT2") {
    throw std::runtime_error("invalid checkpoint magic");
  }
  step = read_i32(in);
  Config cfg;
  if (magic_text == "MICROGPT2") {
    cfg.tokenizer_kind = read_i32(in);
    Tokenizer tok(tokenizer_kind_from_int(cfg.tokenizer_kind));
    cfg.vocab_size = read_i32(in);
    if (cfg.vocab_size != tok.vocab_size()) {
      throw std::runtime_error("checkpoint tokenizer/vocab size mismatch");
    }
  } else {
    cfg.tokenizer_kind = static_cast<int>(TokenizerKind::Byte);
    cfg.vocab_size = read_i32(in);
  }
  cfg.context_length = read_i32(in);
  cfg.d_model = read_i32(in);
  cfg.num_layers = read_i32(in);
  cfg.num_heads = read_i32(in);
  cfg.d_ff = read_i32(in);
  cfg.batch_size = read_i32(in);
  cfg.learning_rate = read_f32(in);
  cfg.beta1 = read_f32(in);
  cfg.beta2 = read_f32(in);
  cfg.adam_eps = read_f32(in);
  cfg.weight_decay = read_f32(in);
  cfg.max_grad_norm = read_f32(in);
  opt.step = read_i32(in);
  Model model(cfg);
  uint64_t nparams = read_u64(in);
  auto params = model.parameters();
  if (params.size() != nparams) {
    throw std::runtime_error("checkpoint parameter count mismatch");
  }
  for (size_t i = 0; i < params.size(); ++i) {
    std::string name = read_string(in);
    std::vector<int> shape = read_shape(in);
    std::vector<float> data = read_vector_f32(in);
    std::vector<float> m = read_vector_f32(in);
    std::vector<float> v = read_vector_f32(in);
    uint64_t decay_flag = read_u64(in);
    Parameter* p = params[i];
    if (p->name != name) {
      throw std::runtime_error("checkpoint parameter name mismatch: " + p->name + " vs " + name);
    }
    if (p->shape != shape) {
      throw std::runtime_error("checkpoint parameter shape mismatch");
    }
    p->data = std::move(data);
    p->m = std::move(m);
    p->v = std::move(v);
    p->decay = decay_flag != 0;
    p->grad.assign(p->data.size(), 0.0f);
  }
  return model;
}

}  // namespace microgpt
