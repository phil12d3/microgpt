#pragma once

#include "microgpt/chat_commands.hpp"
#include "microgpt/chat_config.hpp"
#include "microgpt/chat_session.hpp"
#include "microgpt/chat_tui_input.hpp"
#include "microgpt/chat_tui_render.hpp"
#include "microgpt/checkpoint.hpp"
#include "microgpt/generation.hpp"

#include <chrono>
#include <condition_variable>
#include <ostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace microgpt {

inline void run_chat_tui_loop(Model& model, ChatConfig config, std::ostream& err) {
#if !defined(_WIN32)
  (void)err;
  RawTerminalGuard guard;
  if (!guard.active()) {
    throw std::runtime_error("failed to enable terminal raw mode");
  }
  Tokenizer tok(tokenizer_kind_from_int(model.cfg.tokenizer_kind));
  ChatUiState ui;
  ui.checkpoint = config.checkpoint;
  ui.backend_name = backend_name(config.backend);
  ui.backend_detail = backend_detail(config.backend);
  ui.context_length = model.cfg.context_length;
  ui.max_new_tokens = config.max_new_tokens;
  ui.temperature = config.temperature;
  ui.top_k = config.top_k;
  ui.greedy = config.top_k == 1;

  std::vector<ChatMessage> history;
  bool busy = false;
  std::mutex state_mutex;
  std::condition_variable finished_cv;
  bool generation_done = false;
  std::string generated_reply;
  std::thread worker;

  while (true) {
    {
      std::lock_guard<std::mutex> lock(state_mutex);
      render_chat_ui(ui);
    }

    if (!busy) {
      int key = read_key();
      if (key < 0) {
        continue;
      }
      if (key == 3) {
        break;
      }
      if (key == '\r' || key == '\n') {
        if (ui.input.empty()) {
          continue;
        }
        if (ui.input == "/exit" || ui.input == "/quit") {
          break;
        }
        if (ui.input == "/reset") {
          history.clear();
          ui.scrollback.clear();
          ui.input.clear();
          ui.input_cursor = 0;
          ui.command_history.clear();
          ui.history_index = -1;
          ui.command_suggestions.clear();
          ui.command_suggestion_index = -1;
          continue;
        }
        if (ui.input == "/clear") {
          history.clear();
          ui.input.clear();
          ui.input_cursor = 0;
          append_scrollback(ui, "system", "context cleared", static_cast<size_t>(terminal_size().first));
          continue;
        }
        std::vector<ChatMessage> trimmed_history = trim_chat_history_to_context(history, ui.input, model.cfg.context_length, tok);
        std::string model_prompt = format_multi_turn_chat_prompt(trimmed_history, ui.input);
        ui.last_prompt_tokens = tok.encode_text(model_prompt).size();
        ui.generating = true;
        generation_done = false;
        generated_reply.clear();
        std::string submitted_prompt = ui.input;
        append_scrollback(ui, "user", submitted_prompt, static_cast<size_t>(terminal_size().first));
        ui.command_history.push_back(submitted_prompt);
        ui.history_index = -1;
        ui.command_suggestions.clear();
        ui.command_suggestion_index = -1;
        ui.input.clear();
        ui.input_cursor = 0;
        worker = std::thread([&model, model_prompt, config, &ui, &state_mutex, &finished_cv, &generation_done, &generated_reply]() {
          std::string reply = generate_text(
              model, model_prompt, config.max_new_tokens, config.temperature, config.top_k, Tokenizer::kEos,
              [&ui, &state_mutex](const GenerationProgress& p) {
                std::lock_guard<std::mutex> lock(state_mutex);
                ui.progress.token_index = p.token_index;
                ui.progress.total_tokens = p.total_tokens;
                ui.progress.last_token = p.last_token;
                ui.progress.elapsed_seconds = p.elapsed_seconds;
                ui.progress.tokens_per_second = p.tokens_per_second;
              });
          {
            std::lock_guard<std::mutex> lock(state_mutex);
            generated_reply = reply;
            generation_done = true;
          }
          finished_cv.notify_one();
        });
        busy = true;
        continue;
      }
      if (key == 127 || key == 8) {
        if (ui.input_cursor > 0 && ui.input_cursor <= ui.input.size()) {
          ui.input.erase(ui.input_cursor - 1, 1);
          --ui.input_cursor;
          refresh_command_suggestions(ui);
        }
        continue;
      }
      if (key == 1004) {
        if (!ui.command_suggestions.empty()) {
          cycle_command_suggestion(ui, -1);
          continue;
        }
        if (ui.input_cursor > 0) {
          --ui.input_cursor;
        }
        continue;
      }
      if (key == 1003) {
        if (!ui.command_suggestions.empty()) {
          cycle_command_suggestion(ui, 1);
          continue;
        }
        if (ui.input_cursor < ui.input.size()) {
          ++ui.input_cursor;
        }
        continue;
      }
      if (key == 1005) {
        ui.input_cursor = 0;
        continue;
      }
      if (key == 1006) {
        ui.input_cursor = ui.input.size();
        continue;
      }
      if (key == 1001) {
        if (!ui.command_history.empty()) {
          if (ui.history_index < 0) {
            ui.history_index = static_cast<int>(ui.command_history.size()) - 1;
          } else if (ui.history_index > 0) {
            --ui.history_index;
          }
          if (ui.history_index >= 0 && ui.history_index < static_cast<int>(ui.command_history.size())) {
            ui.input = ui.command_history[static_cast<size_t>(ui.history_index)];
            ui.input_cursor = ui.input.size();
          }
        }
        continue;
      }
      if (key == 1002) {
        if (!ui.command_history.empty()) {
          if (ui.history_index >= 0 && ui.history_index + 1 < static_cast<int>(ui.command_history.size())) {
            ++ui.history_index;
            ui.input = ui.command_history[static_cast<size_t>(ui.history_index)];
            ui.input_cursor = ui.input.size();
          } else {
            ui.history_index = -1;
            ui.input.clear();
            ui.input_cursor = 0;
          }
        }
        continue;
      }
      if (key == '\t') {
        if (!ui.command_suggestions.empty() && ui.command_suggestion_index >= 0 &&
            ui.command_suggestion_index < static_cast<int>(ui.command_suggestions.size())) {
          ui.input = ui.command_suggestions[static_cast<size_t>(ui.command_suggestion_index)];
          ui.input_cursor = ui.input.size();
          refresh_command_suggestions(ui);
        }
        continue;
      }
      if (key >= 32 && key <= 126) {
        ui.input.insert(ui.input.begin() + static_cast<std::string::difference_type>(ui.input_cursor),
                        static_cast<char>(key));
        ++ui.input_cursor;
        refresh_command_suggestions(ui);
        ui.history_index = -1;
      }
    } else {
      std::unique_lock<std::mutex> lock(state_mutex);
      finished_cv.wait_for(lock, std::chrono::milliseconds(50), [&generation_done]() { return generation_done; });
      bool done = generation_done;
      lock.unlock();
      render_chat_ui(ui);
      if (done) {
        if (worker.joinable()) {
          worker.join();
        }
        append_scrollback(ui, "assistant", generated_reply, static_cast<size_t>(terminal_size().first));
        history.push_back({ChatRole::User, ui.command_history.empty() ? std::string() : ui.command_history.back()});
        history.push_back({ChatRole::Assistant, generated_reply});
        ui.stats = summarize_chat_session(history, tok);
        ui.last_completion_tokens = tok.encode_text(generated_reply).size();
        ui.generating = false;
        ui.progress = {};
        busy = false;
      }
    }
  }

  if (worker.joinable()) {
    worker.join();
  }
#else
  (void)model;
  (void)config;
  (void)err;
  throw std::runtime_error("terminal UI is only available on Unix-like systems");
#endif
}

}  // namespace microgpt
