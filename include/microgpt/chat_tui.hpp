#pragma once

#include "microgpt/chat_config.hpp"
#include "microgpt/chat_session.hpp"
#include "microgpt/checkpoint.hpp"
#include "microgpt/generation.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iomanip>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#include <sys/ioctl.h>
#include <sys/select.h>
#include <termios.h>
#include <unistd.h>
#endif

namespace microgpt {

struct ChatUiProgress {
  int token_index = 0;
  int total_tokens = 0;
  int last_token = -1;
  double elapsed_seconds = 0.0;
  double tokens_per_second = 0.0;
};

struct ChatUiState {
  std::string checkpoint;
  std::string backend_name;
  std::string backend_detail;
  int context_length = 0;
  int max_new_tokens = 0;
  float temperature = 0.0f;
  int top_k = 0;
  bool greedy = false;
  bool generating = false;
  bool quit = false;
  std::string input;
  size_t input_cursor = 0;
  struct ScrollbackEntry {
    std::string label;
    std::string text;
  };
  std::vector<ScrollbackEntry> scrollback;
  std::vector<std::string> command_history;
  int history_index = -1;
  std::vector<std::string> command_suggestions;
  int command_suggestion_index = -1;
  ChatUiProgress progress;
  ChatSessionStats stats;
  size_t last_prompt_tokens = 0;
  size_t last_completion_tokens = 0;
};

inline const std::vector<std::string>& chat_commands() {
  static const std::vector<std::string> commands = {"/exit", "/quit", "/reset"};
  return commands;
}

inline bool starts_with(const std::string& text, const std::string& prefix) {
  return text.size() >= prefix.size() && text.compare(0, prefix.size(), prefix) == 0;
}

inline void refresh_command_suggestions(ChatUiState& ui) {
  ui.command_suggestions.clear();
  ui.command_suggestion_index = -1;
  if (ui.input.empty() || ui.input[0] != '/') {
    return;
  }
  for (const std::string& command : chat_commands()) {
    if (starts_with(command, ui.input)) {
      ui.command_suggestions.push_back(command);
    }
  }
  if (!ui.command_suggestions.empty()) {
    ui.command_suggestion_index = 0;
  }
}

inline std::string format_command_suggestions(const ChatUiState& ui, size_t width) {
  if (ui.command_suggestions.empty()) {
    return "";
  }
  std::ostringstream out;
  out << "suggest: ";
  for (size_t i = 0; i < ui.command_suggestions.size(); ++i) {
    if (i > 0) {
      out << ' ';
    }
    if (static_cast<int>(i) == ui.command_suggestion_index) {
      out << '[' << ui.command_suggestions[i] << ']';
    } else {
      out << ui.command_suggestions[i];
    }
  }
  return out.str().substr(0, width);
}

inline std::vector<std::string> wrap_text(const std::string& text, size_t width) {
  std::vector<std::string> lines;
  if (width == 0) {
    lines.push_back(text);
    return lines;
  }
  std::istringstream in(text);
  std::string word;
  std::string line;
  while (in >> word) {
    if (line.empty()) {
      line = word;
    } else if (line.size() + 1 + word.size() <= width) {
      line.push_back(' ');
      line += word;
    } else {
      lines.push_back(line);
      line = word;
    }
  }
  if (!line.empty()) {
    lines.push_back(line);
  }
  if (lines.empty()) {
    lines.push_back("");
  }
  return lines;
}

inline std::vector<std::string> split_lines_preserve_empty(const std::string& text) {
  std::vector<std::string> lines;
  std::istringstream in(text);
  std::string line;
  while (std::getline(in, line)) {
    lines.push_back(line);
  }
  if (text.empty() || (!text.empty() && text.back() == '\n')) {
    lines.push_back("");
  }
  if (lines.empty()) {
    lines.push_back(text);
  }
  return lines;
}

inline std::vector<std::string> build_scrollback_lines(const ChatUiState::ScrollbackEntry& entry, size_t width) {
  std::vector<std::string> lines;
  std::vector<std::string> raw_lines = split_lines_preserve_empty(entry.text);
  std::string prefix = entry.label;
  size_t available = width > prefix.size() ? width - prefix.size() : 1;
  if (raw_lines.empty()) {
    lines.push_back(prefix);
    return lines;
  }
  std::vector<std::string> first_wrapped = wrap_text(raw_lines.front(), available);
  if (first_wrapped.empty()) {
    first_wrapped.push_back("");
  }
  lines.push_back(prefix + first_wrapped.front());
  std::string indent(prefix.size(), ' ');
  for (size_t i = 1; i < first_wrapped.size(); ++i) {
    lines.push_back(indent + first_wrapped[i]);
  }
  for (size_t i = 1; i < raw_lines.size(); ++i) {
    std::vector<std::string> wrapped = wrap_text(raw_lines[i], width);
    for (const std::string& line : wrapped) {
      lines.push_back(indent + line);
    }
  }
  if (lines.empty()) {
    lines.push_back(prefix);
  }
  return lines;
}

class RawTerminalGuard {
 public:
#if !defined(_WIN32)
  RawTerminalGuard() {
    if (tcgetattr(STDIN_FILENO, &original_) != 0) {
      active_ = false;
      return;
    }
    termios raw = original_;
    raw.c_lflag &= static_cast<unsigned>(~(ECHO | ICANON | IEXTEN | ISIG));
    raw.c_iflag &= static_cast<unsigned>(~(IXON | ICRNL));
    raw.c_oflag &= static_cast<unsigned>(~(OPOST));
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 1;
    if (tcsetattr(STDIN_FILENO, TCSANOW, &raw) != 0) {
      active_ = false;
      return;
    }
    active_ = true;
    std::fputs("\033[?1049h\033[?25l", stdout);
    std::fflush(stdout);
  }

  ~RawTerminalGuard() {
    if (active_) {
      tcsetattr(STDIN_FILENO, TCSANOW, &original_);
      std::fputs("\033[?25h\033[?1049l", stdout);
      std::fflush(stdout);
    }
  }
#else
  RawTerminalGuard() = default;
  ~RawTerminalGuard() = default;
#endif

  bool active() const { return active_; }

 private:
#if !defined(_WIN32)
  termios original_{};
#endif
  bool active_ = false;
};

inline bool stdin_has_keypress() {
#if !defined(_WIN32)
  timeval tv{};
  fd_set readfds;
  FD_ZERO(&readfds);
  FD_SET(STDIN_FILENO, &readfds);
  return select(STDIN_FILENO + 1, &readfds, nullptr, nullptr, &tv) > 0 && FD_ISSET(STDIN_FILENO, &readfds);
#else
  return false;
#endif
}

inline int read_key() {
#if !defined(_WIN32)
  unsigned char c = 0;
  ssize_t n = ::read(STDIN_FILENO, &c, 1);
  if (n <= 0) {
    return -1;
  }
  if (c == '\033') {
    unsigned char seq[2] = {0, 0};
    if (::read(STDIN_FILENO, &seq[0], 1) <= 0) {
      return 27;
    }
    if (::read(STDIN_FILENO, &seq[1], 1) <= 0) {
      return 27;
    }
    if (seq[0] == '[') {
      switch (seq[1]) {
        case 'A':
          return 1001;
        case 'B':
          return 1002;
        case 'C':
          return 1003;
        case 'D':
          return 1004;
        case 'H':
          return 1005;
        case 'F':
          return 1006;
      }
    }
    return 27;
  }
  return static_cast<int>(c);
#else
  return -1;
#endif
}

inline std::pair<int, int> terminal_size() {
#if !defined(_WIN32)
  winsize ws{};
  if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0 && ws.ws_row > 0) {
    return {static_cast<int>(ws.ws_col), static_cast<int>(ws.ws_row)};
  }
#endif
  return {100, 30};
}

inline std::string join_lines(const std::vector<std::string>& lines) {
  std::ostringstream out;
  for (size_t i = 0; i < lines.size(); ++i) {
    if (i > 0) {
      out << '\n';
    }
    out << lines[i];
  }
  return out.str();
}

inline std::string basename_of(const std::string& path) {
  size_t pos = path.find_last_of("/\\");
  if (pos == std::string::npos) {
    return path;
  }
  return path.substr(pos + 1);
}

inline void append_scrollback(ChatUiState& ui, const std::string& label, const std::string& text, size_t width) {
  (void)width;
  ui.scrollback.push_back({label, text});
}

inline void render_chat_ui(const ChatUiState& ui) {
#if !defined(_WIN32)
  const auto size = terminal_size();
  const int cols = std::max(size.first, 40);
  const int rows = std::max(size.second, 12);
  int footer_lines = 3;
  int input_lines = 2;
  int title_lines = 2;
  int suggestion_lines = ui.command_suggestions.empty() ? 0 : 1;
  int footer_total = footer_lines + suggestion_lines;
  int content_lines = std::max(1, rows - footer_total - input_lines - title_lines);
  int input_row = title_lines + content_lines + 2;
  int input_col = 3 + static_cast<int>(std::min(ui.input_cursor, ui.input.size()));
  std::vector<std::string> content;
  for (const auto& entry : ui.scrollback) {
    std::vector<std::string> lines = build_scrollback_lines(entry, static_cast<size_t>(cols));
    content.insert(content.end(), lines.begin(), lines.end());
  }
  if (static_cast<int>(content.size()) > content_lines) {
    content.erase(content.begin(), content.end() - content_lines);
  }
  auto fit = [cols](const std::string& text) {
    std::string line = text.substr(0, static_cast<size_t>(cols));
    if (static_cast<int>(line.size()) < cols) {
      line.append(static_cast<size_t>(cols - static_cast<int>(line.size())), ' ');
    }
    return line;
  };
  std::ostringstream out;
  out << "\033[H\033[2J\033[?25l";
  auto draw_line = [&](int row, const std::string& text) {
    out << "\033[" << row << ";1H" << fit(text);
  };
  draw_line(1, "microgpt chat");
  draw_line(2, "type /exit to quit, /reset to clear the session");
  for (int i = 0; i < content_lines; ++i) {
    std::string line = i < static_cast<int>(content.size()) ? content[static_cast<size_t>(i)] : std::string();
    draw_line(title_lines + 1 + i, line);
  }
  draw_line(title_lines + content_lines + 1, std::string(static_cast<size_t>(cols), '-'));
  draw_line(input_row, std::string("> ") + ui.input);
  if (!ui.command_suggestions.empty()) {
    draw_line(input_row + 1, format_command_suggestions(ui, static_cast<size_t>(cols)));
  }
  std::ostringstream status;
  status << "model " << basename_of(ui.checkpoint);
  if (!ui.backend_detail.empty()) {
    status << " | b " << ui.backend_name << " " << ui.backend_detail;
  } else {
    status << " | b " << ui.backend_name;
  }
  status << " | ctx " << ui.context_length;
  status << " | max " << ui.max_new_tokens;
  status << " | t " << ui.temperature;
  status << " | k " << ui.top_k;
  status << " | turns " << ui.stats.total_turns;
  status << " | in " << ui.stats.prompt_tokens;
  status << " | out " << ui.stats.completion_tokens;
  if (ui.generating) {
    status << " | gen " << ui.progress.token_index << '/' << ui.progress.total_tokens;
    status << " " << std::fixed << std::setprecision(2) << ui.progress.tokens_per_second << " tok/s";
  }
  draw_line(input_row + 1 + suggestion_lines, status.str());
  draw_line(input_row + 2 + suggestion_lines, "arrows: history/cursor | tab: accept");
  out << "\033[" << input_row << ';' << input_col << 'H';
  out << "\033[?25h";
  std::fputs(out.str().c_str(), stdout);
  std::fflush(stdout);
#endif
}

inline void run_chat_tui_loop(Model& model, ChatConfig config, std::ostream& err) {
#if !defined(_WIN32)
  (void)err;
  RawTerminalGuard guard;
  if (!guard.active()) {
    throw std::runtime_error("failed to enable terminal raw mode");
  }
  Tokenizer tok;
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
  bool running = true;
  std::mutex state_mutex;
  std::condition_variable finished_cv;
  bool generation_done = false;
  std::string generated_reply;
  std::thread worker;
  while (running) {
    ChatUiState snapshot;
    {
      std::lock_guard<std::mutex> lock(state_mutex);
      snapshot = ui;
    }
    if (!busy) {
      render_chat_ui(snapshot);
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
          append_scrollback(ui, "", "session reset", static_cast<size_t>(terminal_size().first));
          continue;
        }
        std::vector<ChatMessage> trimmed_history = trim_chat_history_to_context(history, ui.input, model.cfg.context_length);
        std::string model_prompt = format_multi_turn_chat_prompt(trimmed_history, ui.input);
        size_t prompt_tokens = tok.encode_text(model_prompt).size();
        ui.last_prompt_tokens = prompt_tokens;
        ui.generating = true;
        generation_done = false;
        generated_reply.clear();
        std::string submitted_prompt = ui.input;
        append_scrollback(ui, "you: ", submitted_prompt, static_cast<size_t>(terminal_size().first));
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
        }
        continue;
      }
      if (key == 1004) {
        if (ui.input_cursor > 0) {
          --ui.input_cursor;
        }
        refresh_command_suggestions(ui);
        continue;
      }
      if (key == 1003) {
        if (ui.input_cursor < ui.input.size()) {
          ++ui.input_cursor;
        }
        refresh_command_suggestions(ui);
        continue;
      }
      if (key == 1005) {
        ui.input_cursor = 0;
        refresh_command_suggestions(ui);
        continue;
      }
      if (key == 1006) {
        ui.input_cursor = ui.input.size();
        refresh_command_suggestions(ui);
        continue;
      }
      if (key == 1001) {
        if (!ui.command_suggestions.empty()) {
          if (ui.command_suggestion_index < 0) {
            ui.command_suggestion_index = 0;
          } else if (ui.command_suggestion_index > 0) {
            --ui.command_suggestion_index;
          } else {
            ui.command_suggestion_index = static_cast<int>(ui.command_suggestions.size()) - 1;
          }
          continue;
        }
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
        if (!ui.command_suggestions.empty()) {
          if (ui.command_suggestion_index < 0) {
            ui.command_suggestion_index = 0;
          } else {
            ui.command_suggestion_index = (ui.command_suggestion_index + 1) % static_cast<int>(ui.command_suggestions.size());
          }
          continue;
        }
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
      snapshot = ui;
      lock.unlock();
      render_chat_ui(snapshot);
      if (done) {
        if (worker.joinable()) {
          worker.join();
        }
        append_scrollback(ui, "assistant: ", generated_reply, static_cast<size_t>(terminal_size().first));
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
