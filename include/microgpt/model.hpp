#pragma once

#include "microgpt/batch.hpp"
#include "microgpt/layers.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace microgpt {
struct Model {
  Config cfg;
  RNG rng;
  Parameter token_embedding;
  Parameter position_embedding;
  std::vector<Block> blocks;
  LayerNorm final_norm;
  Linear lm_head;
  BackendKind backend = BackendKind::Cpu;
  std::vector<int> last_tokens;
  int last_batch_size = 0;
  SeqTensor embed_out;
  std::vector<SeqTensor> block_inputs;
  SeqTensor pre_norm;
  SeqTensor norm_out;

  explicit Model(const Config& c)
      : cfg(c),
        rng(c.seed),
        token_embedding("tok_emb", {c.vocab_size, c.d_model}, true),
        position_embedding("pos_emb", {c.context_length, c.d_model}, true),
        final_norm("final_norm", c.d_model),
        lm_head("lm_head", c.d_model, c.vocab_size, rng) {
    fill_normal(token_embedding.data, rng, 0.0f, 0.02f);
    fill_normal(position_embedding.data, rng, 0.0f, 0.02f);
    blocks.reserve(static_cast<size_t>(cfg.num_layers));
    for (int i = 0; i < cfg.num_layers; ++i) {
      blocks.emplace_back("block" + std::to_string(i), cfg.d_model, cfg.num_heads, cfg.d_ff, rng);
    }
  }

  void set_backend(BackendKind kind) {
    require_backend_available(kind);
    backend = kind;
    final_norm.set_backend(kind);
    lm_head.set_backend(kind);
    for (auto& block : blocks) {
      block.set_backend(kind);
    }
  }

  void zero_grad() {
    token_embedding.zero_grad();
    position_embedding.zero_grad();
    final_norm.gamma.zero_grad();
    final_norm.beta.zero_grad();
    lm_head.w.zero_grad();
    lm_head.b.zero_grad();
    for (auto& block : blocks) {
      block.ln1.gamma.zero_grad();
      block.ln1.beta.zero_grad();
      block.ln2.gamma.zero_grad();
      block.ln2.beta.zero_grad();
      block.attn.q_proj.w.zero_grad();
      block.attn.q_proj.b.zero_grad();
      block.attn.k_proj.w.zero_grad();
      block.attn.k_proj.b.zero_grad();
      block.attn.v_proj.w.zero_grad();
      block.attn.v_proj.b.zero_grad();
      block.attn.o_proj.w.zero_grad();
      block.attn.o_proj.b.zero_grad();
      block.ff.fc1.w.zero_grad();
      block.ff.fc1.b.zero_grad();
      block.ff.fc2.w.zero_grad();
      block.ff.fc2.b.zero_grad();
    }
  }

  std::vector<Parameter*> parameters() {
    std::vector<Parameter*> params;
    params.push_back(&token_embedding);
    params.push_back(&position_embedding);
    params.push_back(&final_norm.gamma);
    params.push_back(&final_norm.beta);
    params.push_back(&lm_head.w);
    params.push_back(&lm_head.b);
    for (auto& block : blocks) {
      params.push_back(&block.ln1.gamma);
      params.push_back(&block.ln1.beta);
      params.push_back(&block.attn.q_proj.w);
      params.push_back(&block.attn.q_proj.b);
      params.push_back(&block.attn.k_proj.w);
      params.push_back(&block.attn.k_proj.b);
      params.push_back(&block.attn.v_proj.w);
      params.push_back(&block.attn.v_proj.b);
      params.push_back(&block.attn.o_proj.w);
      params.push_back(&block.attn.o_proj.b);
      params.push_back(&block.ln2.gamma);
      params.push_back(&block.ln2.beta);
      params.push_back(&block.ff.fc1.w);
      params.push_back(&block.ff.fc1.b);
      params.push_back(&block.ff.fc2.w);
      params.push_back(&block.ff.fc2.b);
    }
    return params;
  }

  SeqTensor embed_tokens(const std::vector<int>& tokens) {
    if (tokens.size() % static_cast<size_t>(cfg.context_length) != 0) {
      throw std::runtime_error("embed_tokens expects a whole number of context windows");
    }
    int batch = static_cast<int>(tokens.size() / static_cast<size_t>(cfg.context_length));
    SeqTensor x(batch, cfg.context_length, cfg.d_model);
    for (int b = 0; b < batch; ++b) {
      for (int t = 0; t < cfg.context_length; ++t) {
        int tok = tokens[static_cast<size_t>(b) * static_cast<size_t>(cfg.context_length) + static_cast<size_t>(t)];
        if (tok < 0 || tok >= cfg.vocab_size) {
          throw std::runtime_error("token out of range");
        }
        for (int d = 0; d < cfg.d_model; ++d) {
          x.data[idx3(b, t, d, x.T, x.D)] =
              token_embedding.data[static_cast<size_t>(tok) * static_cast<size_t>(cfg.d_model) + static_cast<size_t>(d)] +
              position_embedding.data[static_cast<size_t>(t) * static_cast<size_t>(cfg.d_model) + static_cast<size_t>(d)];
        }
      }
    }
    return x;
  }

  SeqTensor forward(const std::vector<int>& tokens) {
    if (tokens.size() % static_cast<size_t>(cfg.context_length) != 0) {
      throw std::runtime_error("model forward received invalid token count");
    }
    last_tokens = tokens;
    last_batch_size = static_cast<int>(tokens.size() / static_cast<size_t>(cfg.context_length));
    embed_out = embed_tokens(tokens);
    block_inputs.clear();
    SeqTensor x = embed_out;
    for (auto& block : blocks) {
      block_inputs.push_back(x);
      x = block.forward(x);
    }
    pre_norm = x;
    norm_out = final_norm.forward(x);
    return lm_head.forward(norm_out);
  }

  void backward(SeqTensor& logits) {
    lm_head.backward(norm_out, logits);
    final_norm.backward(pre_norm, norm_out);
    SeqTensor* current = &pre_norm;
    for (int i = static_cast<int>(blocks.size()) - 1; i >= 0; --i) {
      blocks[static_cast<size_t>(i)].backward(block_inputs[static_cast<size_t>(i)], *current);
      current = &block_inputs[static_cast<size_t>(i)];
    }
    if (embed_out.grad.size() != embed_out.data.size()) {
      embed_out.grad.assign(embed_out.data.size(), 0.0f);
    }
    for (int b = 0; b < last_batch_size; ++b) {
      for (int t = 0; t < cfg.context_length; ++t) {
        int tok = last_tokens[static_cast<size_t>(b) * static_cast<size_t>(cfg.context_length) + static_cast<size_t>(t)];
        float* eg = &embed_out.grad[idx3(b, t, 0, embed_out.T, embed_out.D)];
        for (int d = 0; d < cfg.d_model; ++d) {
          token_embedding.grad[static_cast<size_t>(tok) * static_cast<size_t>(cfg.d_model) + static_cast<size_t>(d)] +=
              eg[static_cast<size_t>(d)];
          position_embedding.grad[static_cast<size_t>(t) * static_cast<size_t>(cfg.d_model) + static_cast<size_t>(d)] +=
              eg[static_cast<size_t>(d)];
        }
      }
    }
  }
};

}  // namespace microgpt
