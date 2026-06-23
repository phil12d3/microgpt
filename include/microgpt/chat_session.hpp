#pragma once

#include "microgpt/backend.hpp"
#include "microgpt/tokenizer.hpp"

#include <algorithm>
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

inline std::string compact_chat_summary_text(std::string text, size_t max_chars) {
  for (char& c : text) {
    if (c == '\n' || c == '\r' || c == '\t') {
      c = ' ';
    }
  }
  while (!text.empty() && text.front() == ' ') {
    text.erase(text.begin());
  }
  while (!text.empty() && text.back() == ' ') {
    text.pop_back();
  }
  if (max_chars == 0) {
    return "";
  }
  if (text.size() > max_chars) {
    if (max_chars <= 3) {
      return text.substr(0, max_chars);
    }
    return text.substr(0, max_chars - 3) + "...";
  }
  return text;
}

inline std::string build_chat_summary_text(const std::vector<ChatMessage>& history, size_t max_messages, size_t max_chars_per_message) {
  if (history.empty()) {
    return "Earlier context compressed.";
  }
  size_t limit = std::min(max_messages, history.size());
  if (limit == 0) {
    return "Earlier context compressed.";
  }
  std::ostringstream out;
  out << "Earlier context compressed:";
  size_t start = history.size() - limit;
  for (size_t i = start; i < history.size(); ++i) {
    out << "\n- ";
    out << (history[i].role == ChatRole::User ? "user: " : "assistant: ");
    out << compact_chat_summary_text(history[i].content, max_chars_per_message);
  }
  return out.str();
}

inline std::vector<ChatMessage> compress_chat_history(const std::vector<ChatMessage>& history, const Tokenizer& tok,
                                                      int context_length, size_t keep_recent_messages = 4) {
  if (history.empty()) {
    return history;
  }
  if (context_length <= 0) {
    return history;
  }
  const std::string original_prompt = format_multi_turn_chat_prompt(history, "");
  const size_t original_tokens = tok.encode_text(original_prompt).size();
  size_t recent = std::min(keep_recent_messages, history.size());
  if (recent % 2 != 0 && recent > 1) {
    --recent;
  }
  if (recent < 2 && history.size() >= 2) {
    recent = 2;
  }
  if (history.size() <= recent) {
    return history;
  }

  std::vector<ChatMessage> tail(history.end() - static_cast<std::ptrdiff_t>(recent), history.end());
  size_t summary_messages = std::min<size_t>(6, history.size() - recent);
  size_t summary_chars = 72;
  std::vector<ChatMessage> compressed;
  std::vector<ChatMessage> best = history;
  size_t best_tokens = original_tokens;
  while (summary_messages > 0) {
    compressed.clear();
    std::vector<ChatMessage> removed(history.begin(), history.end() - static_cast<std::ptrdiff_t>(recent));
    std::string summary = build_chat_summary_text(removed, summary_messages, summary_chars);
    compressed.push_back({ChatRole::User, summary});
    compressed.push_back({ChatRole::Assistant, "Continue from the recent conversation."});
    compressed.insert(compressed.end(), tail.begin(), tail.end());

    std::string test_prompt = format_multi_turn_chat_prompt(compressed, "");
    size_t candidate_tokens = tok.encode_text(test_prompt).size();
    if (candidate_tokens < best_tokens) {
      best = compressed;
      best_tokens = candidate_tokens;
    }

    if (candidate_tokens <= static_cast<size_t>(context_length) && candidate_tokens < original_tokens) {
      return compressed;
    }

    if (tail.size() >= 4) {
      tail.erase(tail.begin(), tail.begin() + 2);
      continue;
    }

    if (summary_chars > 24) {
      summary_chars = summary_chars > 48 ? 48 : 24;
      continue;
    }

    if (summary_messages > 1) {
      --summary_messages;
      continue;
    }

    break;
  }

  if (best_tokens < original_tokens) {
    return best;
  }
  return history;
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
