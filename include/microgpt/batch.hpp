#pragma once

#include "microgpt/core.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace microgpt {
inline Batch sample_batch(const std::vector<int>& tokens, int batch_size, int context_length, RNG& rng) {
  if (tokens.size() < 2) {
    throw std::runtime_error("not enough tokens for batch sampling");
  }
  Batch batch;
  batch.batch = batch_size;
  batch.time = context_length;
  batch.x.resize(static_cast<size_t>(batch_size) * static_cast<size_t>(context_length));
  batch.y.resize(batch.x.size());
  for (int b = 0; b < batch_size; ++b) {
    int start = rng.randint(0, static_cast<int>(tokens.size()) - 1);
    for (int t = 0; t < context_length; ++t) {
      size_t ix = (static_cast<size_t>(start) + static_cast<size_t>(t)) % tokens.size();
      size_t iy = (ix + 1) % tokens.size();
      batch.x[static_cast<size_t>(b) * static_cast<size_t>(context_length) + static_cast<size_t>(t)] = tokens[ix];
      batch.y[static_cast<size_t>(b) * static_cast<size_t>(context_length) + static_cast<size_t>(t)] = tokens[iy];
    }
  }
  return batch;
}

inline Batch make_fixed_batch(const std::vector<int>& tokens, int batch_size, int context_length) {
  Batch batch;
  batch.batch = batch_size;
  batch.time = context_length;
  batch.x.resize(static_cast<size_t>(batch_size) * static_cast<size_t>(context_length));
  batch.y.resize(batch.x.size());
  for (int b = 0; b < batch_size; ++b) {
    for (int t = 0; t < context_length; ++t) {
      size_t src = static_cast<size_t>(t % std::max<int>(1, static_cast<int>(tokens.size()) - 1));
      batch.x[static_cast<size_t>(b) * static_cast<size_t>(context_length) + static_cast<size_t>(t)] = tokens[src];
      batch.y[static_cast<size_t>(b) * static_cast<size_t>(context_length) + static_cast<size_t>(t)] = tokens[src + 1];
    }
  }
  return batch;
}

inline SeqTensor batch_to_seqtensor(const Batch& b, int d) {
  SeqTensor t(b.batch, b.time, d);
  return t;
}

}  // namespace microgpt
