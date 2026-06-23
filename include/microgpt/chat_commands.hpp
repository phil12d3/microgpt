#pragma once

#include <string>
#include <sstream>
#include <vector>

namespace microgpt {

inline const std::vector<std::string>& chat_commands() {
  static const std::vector<std::string> commands = {"/exit", "/quit", "/reset", "/clear"};
  return commands;
}

inline bool starts_with(const std::string& text, const std::string& prefix) {
  return text.size() >= prefix.size() && text.compare(0, prefix.size(), prefix) == 0;
}

template <typename State>
inline void refresh_command_suggestions(State& ui) {
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

template <typename State>
inline void cycle_command_suggestion(State& ui, int delta) {
  if (ui.command_suggestions.empty()) {
    return;
  }
  if (ui.command_suggestion_index < 0) {
    ui.command_suggestion_index = 0;
    return;
  }
  const int count = static_cast<int>(ui.command_suggestions.size());
  ui.command_suggestion_index = (ui.command_suggestion_index + delta + count) % count;
}

template <typename State>
inline std::string format_command_suggestions(const State& ui, size_t width) {
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

}  // namespace microgpt
