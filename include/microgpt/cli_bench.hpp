#pragma once

#include "microgpt/cli_backend.hpp"
#include "microgpt/cli_shared.hpp"

#include <chrono>

namespace microgpt {

inline void fill_bench_values(std::vector<float>& values, float scale) {
  for (size_t i = 0; i < values.size(); ++i) {
    int v = static_cast<int>((i * 37) % 19) - 9;
    values[i] = static_cast<float>(v) * scale;
  }
}

inline int run_resident_linear_bench(const std::vector<std::string>& args, BackendKind backend, std::ostream& out) {
  int rows = get_arg_int(args, "--rows", 128);
  int in_features = get_arg_int(args, "--in", 128);
  int out_features = get_arg_int(args, "--out", 128);
  int iterations = get_arg_int(args, "--iterations", 100);
  if (rows <= 0 || in_features <= 0 || out_features <= 0 || iterations <= 0) {
    throw std::runtime_error("--rows, --in, --out, and --iterations must be positive");
  }

  std::vector<float> x(static_cast<size_t>(rows) * static_cast<size_t>(in_features));
  std::vector<float> w(static_cast<size_t>(in_features) * static_cast<size_t>(out_features));
  std::vector<float> b(static_cast<size_t>(out_features));
  std::vector<float> dy(static_cast<size_t>(rows) * static_cast<size_t>(out_features));
  fill_bench_values(x, 0.01f);
  fill_bench_values(w, 0.02f);
  fill_bench_values(b, 0.005f);
  fill_bench_values(dy, 0.01f);

  using clock = std::chrono::steady_clock;
  bool metal_resident = backend == BackendKind::Metal && microgpt_metal_runtime_available();
  reset_backend_transfer_stats();
  auto start = clock::now();
  if (metal_resident) {
    BackendBuffer xb(BackendKind::Metal);
    BackendBuffer wb(BackendKind::Metal);
    BackendBuffer bb(BackendKind::Metal);
    BackendBuffer yb(BackendKind::Metal);
    BackendBuffer dyb(BackendKind::Metal);
    BackendBuffer dxb(BackendKind::Metal);
    BackendBuffer dwb(BackendKind::Metal);
    BackendBuffer dbb(BackendKind::Metal);
    xb.resize(x.size());
    wb.resize(w.size());
    bb.resize(b.size());
    yb.resize(static_cast<size_t>(rows) * static_cast<size_t>(out_features));
    dyb.resize(dy.size());
    dxb.resize(x.size());
    dwb.resize(w.size());
    dbb.resize(b.size());
    xb.host = x;
    wb.host = w;
    bb.host = b;
    dyb.host = dy;
    xb.host_dirty = true;
    wb.host_dirty = true;
    bb.host_dirty = true;
    dyb.host_dirty = true;
    xb.upload();
    wb.upload();
    bb.upload();
    dyb.upload();
    bool repeat_ok = microgpt_metal_linear_forward_backward_repeat_buffers(
        xb.device, wb.device, bb.device, yb.device, dyb.device, dxb.device, dwb.device, dbb.device, rows, in_features,
        out_features, true, iterations);
    for (int i = repeat_ok ? iterations : 0; i < iterations; ++i) {
      bool ok = microgpt_metal_linear_forward_backward_buffers(xb.device, wb.device, bb.device, yb.device, dyb.device,
                                                               dxb.device, dwb.device, dbb.device, rows, in_features,
                                                               out_features, true);
      if (!ok) {
        ok = microgpt_metal_linear_forward_buffers(xb.device, wb.device, bb.device, yb.device, rows, in_features,
                                                   out_features, true) &&
             microgpt_metal_linear_backward_buffers(xb.device, wb.device, dyb.device, dxb.device, dwb.device,
                                                    dbb.device, rows, in_features, out_features, true);
      }
      if (!ok) {
        throw std::runtime_error("Metal resident linear forward/backward failed");
      }
    }
    yb.device_dirty = true;
    dxb.device_dirty = true;
    dwb.device_dirty = true;
    dbb.device_dirty = true;
    yb.download();
    dxb.download();
    dwb.download();
    dbb.download();
  } else {
    std::vector<float> y(static_cast<size_t>(rows) * static_cast<size_t>(out_features));
    std::vector<float> dx(x.size());
    std::vector<float> dw(w.size());
    std::vector<float> db(b.size());
    SeqTensor xt(rows, 1, in_features);
    xt.data = x;
    Parameter wp("bench.w", {in_features, out_features}, true);
    Parameter bp("bench.b", {out_features}, false);
    wp.data = w;
    bp.data = b;
    SeqTensor yt(rows, 1, out_features);
    yt.grad = dy;
    for (int i = 0; i < iterations; ++i) {
      std::fill(y.begin(), y.end(), 0.0f);
      std::fill(xt.grad.begin(), xt.grad.end(), 0.0f);
      std::fill(wp.grad.begin(), wp.grad.end(), 0.0f);
      std::fill(bp.grad.begin(), bp.grad.end(), 0.0f);
      matmul_forward_cpu(x.data(), w.data(), y.data(), rows, in_features, out_features);
      for (int r = 0; r < rows; ++r) {
        for (int o = 0; o < out_features; ++o) {
          y[static_cast<size_t>(r) * static_cast<size_t>(out_features) + static_cast<size_t>(o)] +=
              b[static_cast<size_t>(o)];
        }
      }
      linear_backward_cpu(xt, wp, bp, in_features, out_features, yt);
    }
    dx = xt.grad;
    dw = wp.grad;
    db = bp.grad;
    (void)dx;
    (void)dw;
    (void)db;
  }
  auto end = clock::now();
  double seconds = std::chrono::duration<double>(end - start).count();
  out << std::fixed << std::setprecision(4);
  out << "benchmark resident_linear\n";
  out << "backend " << backend_name(backend) << '\n';
  out << "backend_detail " << backend_detail(backend) << '\n';
  out << "metal_resident_active " << (metal_resident ? "yes" : "no") << '\n';
  out << "rows " << rows << '\n';
  out << "in " << in_features << '\n';
  out << "out " << out_features << '\n';
  out << "iterations " << iterations << '\n';
  out << "seconds " << seconds << '\n';
  out << "iterations_per_second " << (seconds > 0.0 ? static_cast<double>(iterations) / seconds : 0.0) << '\n';
  const BackendTransferStats& stats = backend_transfer_stats();
  out << "backend_uploads " << stats.uploads << '\n';
  out << "backend_upload_bytes " << stats.upload_bytes << '\n';
  out << "backend_downloads " << stats.downloads << '\n';
  out << "backend_download_bytes " << stats.download_bytes << '\n';
  return 0;
}

inline int run_bench_command(const std::vector<std::string>& args, std::ostream& out) {
  BackendKind backend = require_backend_arg(args);
  if (has_arg(args, "--resident-linear")) {
    return run_resident_linear_bench(args, backend, out);
  }
  Config cfg;
  cfg.context_length = get_arg_int(args, "--context", 32);
  cfg.d_model = get_arg_int(args, "--d-model", 32);
  cfg.num_layers = get_arg_int(args, "--layers", 1);
  cfg.num_heads = get_arg_int(args, "--heads", 4);
  cfg.d_ff = get_arg_int(args, "--ff", 64);
  cfg.batch_size = get_arg_int(args, "--batch-size", 4);
  cfg.learning_rate = get_arg_float(args, "--lr", 0.001f);
  cfg.seed = static_cast<uint32_t>(get_arg_int(args, "--seed", 42));
  int train_steps = get_arg_int(args, "--train-steps", 20);
  int gen_tokens = get_arg_int(args, "--gen-tokens", 20);

  Model model(cfg);
  model.set_backend(backend);
  AdamW opt;
  opt.lr = cfg.learning_rate;
  opt.beta1 = cfg.beta1;
  opt.beta2 = cfg.beta2;
  opt.eps = cfg.adam_eps;
  opt.weight_decay = cfg.weight_decay;

  std::vector<int> tokens;
  std::string text = "microgpt benchmark data ";
  while (static_cast<int>(tokens.size()) < cfg.context_length * cfg.batch_size * 8) {
    for (unsigned char c : text) {
      tokens.push_back(static_cast<int>(c));
    }
  }

  reset_backend_transfer_stats();
  using clock = std::chrono::steady_clock;
  auto train_start = clock::now();
  for (int step = 0; step < train_steps; ++step) {
    model.zero_grad();
    Batch batch = sample_batch(tokens, cfg.batch_size, cfg.context_length, model.rng);
    SeqTensor logits = model.forward(batch.x);
    cross_entropy_loss(logits, batch.y);
    model.backward(logits);
    auto params = model.parameters();
    clip_gradients(params, cfg.max_grad_norm);
    opt.update(params);
  }
  auto train_end = clock::now();
  double train_seconds = std::chrono::duration<double>(train_end - train_start).count();

  auto gen_start = clock::now();
  std::string generated = generate_text(model, "micro", gen_tokens, 0.2f, 1, Tokenizer::kEos);
  auto gen_end = clock::now();
  double gen_seconds = std::chrono::duration<double>(gen_end - gen_start).count();

  out << std::fixed << std::setprecision(4);
  out << "backend " << backend_name(backend) << '\n';
  out << "backend_detail " << backend_detail(backend) << '\n';
  out << "context " << cfg.context_length << '\n';
  out << "d_model " << cfg.d_model << '\n';
  out << "layers " << cfg.num_layers << '\n';
  out << "heads " << cfg.num_heads << '\n';
  out << "ff " << cfg.d_ff << '\n';
  out << "batch_size " << cfg.batch_size << '\n';
  out << "train_steps " << train_steps << '\n';
  out << "train_seconds " << train_seconds << '\n';
  out << "train_steps_per_second " << (train_seconds > 0.0 ? static_cast<double>(train_steps) / train_seconds : 0.0)
      << '\n';
  out << "gen_tokens " << gen_tokens << '\n';
  out << "gen_seconds " << gen_seconds << '\n';
  out << "gen_tokens_per_second " << (gen_seconds > 0.0 ? static_cast<double>(gen_tokens) / gen_seconds : 0.0)
      << '\n';
  out << "generated_chars " << generated.size() << '\n';
  const BackendTransferStats& stats = backend_transfer_stats();
  out << "backend_uploads " << stats.uploads << '\n';
  out << "backend_upload_bytes " << stats.upload_bytes << '\n';
  out << "backend_downloads " << stats.downloads << '\n';
  out << "backend_download_bytes " << stats.download_bytes << '\n';
  return 0;
}

}  // namespace microgpt
