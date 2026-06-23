#pragma once

#include "microgpt/chat_commands.hpp"
#include "microgpt/chat_tui_input.hpp"
#include "microgpt/chat_tui_state.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace microgpt {

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
  size_t available = width;
  if (raw_lines.empty()) {
    lines.push_back("");
    return lines;
  }
  std::vector<std::string> first_wrapped = wrap_text(raw_lines.front(), available);
  if (first_wrapped.empty()) {
    first_wrapped.push_back("");
  }
  lines.push_back(first_wrapped.front());
  for (size_t i = 1; i < first_wrapped.size(); ++i) {
    lines.push_back(first_wrapped[i]);
  }
  for (size_t i = 1; i < raw_lines.size(); ++i) {
    std::vector<std::string> wrapped = wrap_text(raw_lines[i], width);
    lines.insert(lines.end(), wrapped.begin(), wrapped.end());
  }
  if (lines.empty()) {
    lines.push_back("");
  }
  return lines;
}

inline std::string basename_of(const std::string& path) {
  size_t pos = path.find_last_of("/\\");
  if (pos == std::string::npos) {
    return path;
  }
  return path.substr(pos + 1);
}

inline std::string pad_to_width(const std::string& text, int width) {
  std::string line = text.substr(0, static_cast<size_t>(std::max(width, 0)));
  if (static_cast<int>(line.size()) < width) {
    line.append(static_cast<size_t>(width - static_cast<int>(line.size())), ' ');
  }
  return line;
}

inline std::string style_text(const std::string& text, const char* style, int width) {
  return std::string(style) + pad_to_width(text, width) + "\033[0m";
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
  const int title_lines = 2;
  const int suggestion_lines = ui.command_suggestions.empty() ? 0 : 1;
  const int footer_lines = 2 + suggestion_lines;
  const int content_lines = std::max(1, rows - title_lines - footer_lines - 1);
  const int input_row = title_lines + content_lines + 2;
  const int input_col = 3 + static_cast<int>(std::min(ui.input_cursor, ui.input.size()));

  struct RenderLine {
    std::string text;
    const char* style = "";
  };

  std::vector<RenderLine> content;
  for (const auto& entry : ui.scrollback) {
    std::vector<std::string> lines = build_scrollback_lines(entry, static_cast<size_t>(cols));
    const char* style = "";
    if (entry.label == "user") {
      style = "\033[48;5;237m\033[37m";
    } else if (entry.label == "assistant") {
      style = "\033[37m";
    }
    for (const std::string& line : lines) {
      content.push_back({line, style});
    }
  }
  if (static_cast<int>(content.size()) > content_lines) {
    content.erase(content.begin(), content.end() - content_lines);
  }

  std::ostringstream out;
  out << "\033[H\033[2J\033[?25l";
  auto draw_line = [&](int row, const std::string& text, const char* style = "") {
    out << "\033[" << row << ";1H";
    if (style[0] != '\0') {
      out << style;
    }
    out << pad_to_width(text, cols);
    if (style[0] != '\0') {
      out << "\033[0m";
    }
  };
  auto draw_bar = [&](int row, const std::string& text, const char* style = "") {
    out << "\033[" << row << ";1H";
    if (style[0] != '\0') {
      out << style;
    } else {
      out << "\033[48;5;236m";
    }
    out << pad_to_width(text, cols);
    out << "\033[0m";
  };

  draw_line(1, "microgpt chat", "\033[1;36m");
  draw_line(2, "interactive assistant session", "\033[2m");
  for (int i = 0; i < content_lines; ++i) {
    RenderLine line = i < static_cast<int>(content.size()) ? content[static_cast<size_t>(i)] : RenderLine{};
    draw_line(title_lines + 1 + i, line.text, line.style);
  }
  draw_bar(title_lines + content_lines + 1, std::string(static_cast<size_t>(cols), ' '));
  draw_bar(input_row, std::string("> ") + ui.input, "\033[48;5;237m\033[1m");
  if (!ui.command_suggestions.empty()) {
    draw_bar(input_row + 1, format_command_suggestions(ui, static_cast<size_t>(cols)), "\033[48;5;236m\033[36m");
  }

  std::ostringstream status;
  status << "model " << basename_of(ui.checkpoint);
  status << " | b " << ui.backend_name;
  if (!ui.backend_detail.empty()) {
    status << ' ' << ui.backend_detail;
  }
  size_t effective_prompt_tokens = ui.context_length > 0
                                       ? std::min(ui.last_prompt_tokens, static_cast<size_t>(ui.context_length))
                                       : ui.last_prompt_tokens;
  int context_percent = ui.context_length > 0
                            ? static_cast<int>((100.0 * static_cast<double>(effective_prompt_tokens)) /
                                               static_cast<double>(ui.context_length))
                            : 0;
  status << " | ctx " << effective_prompt_tokens << '/' << ui.context_length << ' ' << context_percent << '%';
  status << " | max " << ui.max_new_tokens;
  status << " | t " << ui.temperature;
  status << " | k " << ui.top_k;
  status << " | turns " << ui.stats.total_turns;
  status << " | in " << ui.stats.prompt_tokens;
  status << " | out " << ui.stats.completion_tokens;
  if (ui.generating) {
    status << " | gen " << ui.progress.token_index << '/' << ui.progress.total_tokens;
    status << ' ' << std::fixed << std::setprecision(2) << ui.progress.tokens_per_second << " tok/s";
  }
  draw_bar(input_row + 1 + suggestion_lines, status.str(), "\033[48;5;236m\033[2m");
  out << "\033[" << input_row << ';' << input_col << 'H';
  out << "\033[?25h";
  std::fputs(out.str().c_str(), stdout);
  std::fflush(stdout);
#endif
}

}  // namespace microgpt
