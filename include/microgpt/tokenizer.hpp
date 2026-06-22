#pragma once

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace microgpt {

enum class TokenizerKind {
  Byte = 0,
  Bpe = 1,
};

inline std::string tokenizer_name(TokenizerKind kind) {
  switch (kind) {
    case TokenizerKind::Byte:
      return "byte";
    case TokenizerKind::Bpe:
      return "bpe";
  }
  return "byte";
}

inline TokenizerKind tokenizer_kind_from_int(int kind) {
  if (kind == static_cast<int>(TokenizerKind::Byte)) {
    return TokenizerKind::Byte;
  }
  if (kind == static_cast<int>(TokenizerKind::Bpe)) {
    return TokenizerKind::Bpe;
  }
  throw std::runtime_error("unknown tokenizer kind");
}

inline TokenizerKind parse_tokenizer_kind(const std::string& name) {
  if (name.empty() || name == "byte") {
    return TokenizerKind::Byte;
  }
  if (name == "bpe" || name == "simple-bpe" || name == "simple_bpe") {
    return TokenizerKind::Bpe;
  }
  throw std::runtime_error("--tokenizer must be byte or bpe");
}

struct Tokenizer {
  static constexpr int kByteVocab = 256;
  static constexpr int kBos = 256;
  static constexpr int kEos = 257;
  static constexpr int kUser = 258;
  static constexpr int kAssistant = 259;
  static constexpr int kSpecialVocab = 260;

  TokenizerKind kind = TokenizerKind::Byte;

  Tokenizer() = default;
  explicit Tokenizer(TokenizerKind tokenizer_kind) : kind(tokenizer_kind) {}
  explicit Tokenizer(int tokenizer_kind) : kind(tokenizer_kind_from_int(tokenizer_kind)) {}

  static const std::vector<std::string>& bpe_tokens() {
    static const std::vector<std::string> tokens = {
        " validation ", " assistant ", " training ", " checkpoint ", " generate ", " response ", " context ",
        " consistent",  " microgpt",    " assistant",  " training",   " checkpoint", " validation", " dataset",
        " tokens",      " model",       " data",       " file",       " command",    " output",     " input",
        " answer",      " prompt",      " clean",      " true",       " false",      " neutral",    " positive",
        " negative",    " status",      " json",       " return",     " only",       " summarize",  " classify",
        " replace",     " uppercase",   " lowercase",  " spaces",     " hyphens",    " learned",    " improves",
        " held-out",    " used",        " measure",    " generalizes", " the ",       " and ",       " to ",
        " of ",         " in ",         " is ",        " for ",       " with ",      "tion",       "ing",
        "ent",          "ive",          "ate",         "ed",          "er",          "re",         "st",
    };
    return tokens;
  }

  int vocab_size() const {
    if (kind == TokenizerKind::Bpe) {
      return kSpecialVocab + static_cast<int>(bpe_tokens().size());
    }
    return kSpecialVocab;
  }

  std::string name() const { return tokenizer_name(kind); }

  bool try_special(const std::string& text, size_t i, int& token, size_t& advance) const {
    if (text.compare(i, 5, "<BOS>") == 0) {
      token = kBos;
      advance = 5;
      return true;
    }
    if (text.compare(i, 5, "<EOS>") == 0) {
      token = kEos;
      advance = 5;
      return true;
    }
    if (text.compare(i, 6, "<USER>") == 0) {
      token = kUser;
      advance = 6;
      return true;
    }
    if (text.compare(i, 11, "<ASSISTANT>") == 0) {
      token = kAssistant;
      advance = 11;
      return true;
    }
    return false;
  }

  bool try_bpe(const std::string& text, size_t i, int& token, size_t& advance) const {
    if (kind != TokenizerKind::Bpe) {
      return false;
    }
    const std::vector<std::string>& tokens = bpe_tokens();
    for (size_t j = 0; j < tokens.size(); ++j) {
      const std::string& piece = tokens[j];
      if (piece.empty() || i + piece.size() > text.size()) {
        continue;
      }
      if (text.compare(i, piece.size(), piece) == 0) {
        token = kSpecialVocab + static_cast<int>(j);
        advance = piece.size();
        return true;
      }
    }
    return false;
  }

  std::vector<int> encode_text(const std::string& text) const {
    std::vector<int> ids;
    ids.reserve(text.size());
    for (size_t i = 0; i < text.size();) {
      int token = 0;
      size_t advance = 0;
      if (try_special(text, i, token, advance) || try_bpe(text, i, token, advance)) {
        ids.push_back(token);
        i += advance;
        continue;
      }
      ids.push_back(static_cast<int>(static_cast<unsigned char>(text[i])));
      ++i;
    }
    return ids;
  }

  std::string decode_text(const std::vector<int>& ids) const {
    std::string s;
    s.reserve(ids.size());
    for (int id : ids) {
      switch (id) {
        case kBos:
          s += "<BOS>";
          break;
        case kEos:
          s += "<EOS>";
          break;
        case kUser:
          s += "<USER>";
          break;
        case kAssistant:
          s += "<ASSISTANT>";
          break;
        default:
          if (id >= 0 && id < kByteVocab) {
            s.push_back(static_cast<char>(id));
          } else if (id >= kSpecialVocab && id < vocab_size()) {
            s += bpe_tokens()[static_cast<size_t>(id - kSpecialVocab)];
          }
          break;
      }
    }
    return s;
  }

  std::string decode_bytes(const std::vector<int>& ids) const { return decode_text(ids); }
};

}  // namespace microgpt
