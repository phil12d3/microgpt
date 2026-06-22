#pragma once

#include "microgpt/backend.hpp"
#include "microgpt/tokenizer.hpp"

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace microgpt {

enum class ChatRole { User, Assistant };

struct ChatMessage {
  ChatRole role = ChatRole::User;
  std::string content;
};

struct ChatSessionStats {
  size_t user_messages = 0;
  size_t assistant_messages = 0;
  size_t total_turns = 0;
  size_t prompt_tokens = 0;
  size_t completion_tokens = 0;
};

inline std::string format_multi_turn_chat_prompt(const std::vector<ChatMessage>& history, const std::string& user_prompt) {
  std::ostringstream out;
  out << "<BOS>";
  for (const ChatMessage& message : history) {
    if (message.role == ChatRole::User) {
      out << "<USER>\n" << message.content << "\n";
    } else {
      out << "<ASSISTANT>\n" << message.content << "\n";
    }
  }
  out << "<USER>\n" << user_prompt << "\n<ASSISTANT>\n";
  return out.str();
}

inline std::vector<ChatMessage> trim_chat_history_to_context(const std::vector<ChatMessage>& history,
                                                             const std::string& user_prompt, int context_length,
                                                             const Tokenizer& tok) {
  std::vector<ChatMessage> trimmed = history;
  if (context_length <= 0) {
    return trimmed;
  }
  while (!trimmed.empty()) {
    std::string prompt = format_multi_turn_chat_prompt(trimmed, user_prompt);
    if (static_cast<int>(tok.encode_text(prompt).size()) <= context_length) {
      break;
    }
    if (trimmed.size() >= 2) {
      trimmed.erase(trimmed.begin(), trimmed.begin() + 2);
    } else {
      trimmed.erase(trimmed.begin());
    }
  }
  return trimmed;
}

inline std::vector<ChatMessage> trim_chat_history_to_context(const std::vector<ChatMessage>& history,
                                                             const std::string& user_prompt, int context_length) {
  Tokenizer tok;
  return trim_chat_history_to_context(history, user_prompt, context_length, tok);
}

inline ChatSessionStats summarize_chat_session(const std::vector<ChatMessage>& history, const Tokenizer& tok) {
  ChatSessionStats stats;
  stats.total_turns = history.size() / 2;
  for (const ChatMessage& message : history) {
    if (message.role == ChatRole::User) {
      ++stats.user_messages;
      stats.prompt_tokens += tok.encode_text(message.content).size();
    } else {
      ++stats.assistant_messages;
      stats.completion_tokens += tok.encode_text(message.content).size();
    }
  }
  return stats;
}

}  // namespace microgpt
