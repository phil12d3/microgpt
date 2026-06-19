#pragma once

#include "microgpt/chat_session.hpp"

#include <string>
#include <vector>

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

}  // namespace microgpt
