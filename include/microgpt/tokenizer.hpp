#pragma once

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace microgpt {

enum class TokenizerKind {
  Byte = 0,
  Bpe = 1,
  Pairs = 2,
};

inline std::string tokenizer_name(TokenizerKind kind) {
  switch (kind) {
    case TokenizerKind::Byte:
      return "byte";
    case TokenizerKind::Bpe:
      return "bpe";
    case TokenizerKind::Pairs:
      return "pairs";
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
  if (kind == static_cast<int>(TokenizerKind::Pairs)) {
    return TokenizerKind::Pairs;
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
  if (name == "pairs" || name == "pair" || name == "bigram") {
    return TokenizerKind::Pairs;
  }
  throw std::runtime_error("--tokenizer must be byte, bpe, or pairs");
}

struct Tokenizer {
  static constexpr int kByteVocab = 256;
  static constexpr int kBos = 256;
  static constexpr int kEos = 257;
  static constexpr int kUser = 258;
  static constexpr int kAssistant = 259;
  static constexpr int kSpecialVocab = 260;
  static constexpr int kDefaultLearnedVocab = 512;

  TokenizerKind kind = TokenizerKind::Byte;
  std::vector<std::string> learned_pieces;
  std::vector<size_t> encode_order;

  Tokenizer() = default;
  explicit Tokenizer(TokenizerKind tokenizer_kind) : kind(tokenizer_kind) {}
  explicit Tokenizer(int tokenizer_kind) : kind(tokenizer_kind_from_int(tokenizer_kind)) {}
  Tokenizer(TokenizerKind tokenizer_kind, std::vector<std::string> pieces)
      : kind(tokenizer_kind), learned_pieces(std::move(pieces)) {
    rebuild_encode_order();
  }

  static std::string normalize_learning_corpus(const std::string& text) {
    std::string normalized = text;
    const std::vector<std::string> markers = {"<BOS>", "<EOS>", "<USER>", "<ASSISTANT>"};
    for (const std::string& marker : markers) {
      size_t pos = 0;
      while (true) {
        pos = normalized.find(marker, pos);
        if (pos == std::string::npos) {
          break;
        }
        normalized.replace(pos, marker.size(), marker.size(), ' ');
        pos += marker.size();
      }
    }
    return normalized;
  }

  static Tokenizer learn_bpe_from_text(const std::string& text, int target_vocab_size = kDefaultLearnedVocab,
                                       int min_pair_count = 2) {
    if (target_vocab_size < kSpecialVocab) {
      target_vocab_size = kSpecialVocab;
    }

    Tokenizer tok(TokenizerKind::Bpe);
    std::vector<std::string> token_strings;
    token_strings.reserve(static_cast<size_t>(target_vocab_size));
    std::unordered_set<std::string> seen;
    seen.reserve(static_cast<size_t>(target_vocab_size) * 2);

    for (int i = 0; i < kByteVocab; ++i) {
      std::string piece(1, static_cast<char>(i));
      token_strings.push_back(piece);
      seen.insert(piece);
    }

    std::vector<int> seq;
    std::string corpus = normalize_learning_corpus(text);
    seq.reserve(corpus.size());
    for (unsigned char c : corpus) {
      seq.push_back(static_cast<int>(c));
    }

    while (static_cast<int>(kSpecialVocab + tok.learned_pieces.size()) < target_vocab_size) {
      std::unordered_map<uint64_t, int> counts;
      counts.reserve(seq.size());
      int best_count = 0;
      int best_left = -1;
      int best_right = -1;
      for (size_t i = 0; i + 1 < seq.size(); ++i) {
        int left = seq[i];
        int right = seq[i + 1];
        uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(left)) << 32) |
                       static_cast<uint32_t>(right);
        int count = ++counts[key];
        if (count > best_count || (count == best_count &&
                                   (left < best_left || (left == best_left && right < best_right)))) {
          best_count = count;
          best_left = left;
          best_right = right;
        }
      }

      if (best_count < min_pair_count || best_left < 0 || best_right < 0) {
        break;
      }

      if (best_left >= static_cast<int>(token_strings.size()) || best_right >= static_cast<int>(token_strings.size())) {
        break;
      }

      std::string merged = token_strings[static_cast<size_t>(best_left)] + token_strings[static_cast<size_t>(best_right)];
      if (!seen.insert(merged).second) {
        break;
      }

      int new_id = static_cast<int>(token_strings.size());
      token_strings.push_back(merged);
      tok.learned_pieces.push_back(std::move(merged));

      std::vector<int> merged_seq;
      merged_seq.reserve(seq.size());
      for (size_t i = 0; i < seq.size();) {
        if (i + 1 < seq.size() && seq[i] == best_left && seq[i + 1] == best_right) {
          merged_seq.push_back(new_id);
          i += 2;
        } else {
          merged_seq.push_back(seq[i]);
          ++i;
        }
      }
      seq.swap(merged_seq);
    }

    tok.rebuild_encode_order();
    return tok;
  }

  static Tokenizer learn_pairs_from_text(const std::string& text, int target_vocab_size = kDefaultLearnedVocab,
                                         int min_pair_count = 2) {
    if (target_vocab_size < kSpecialVocab) {
      target_vocab_size = kSpecialVocab;
    }

    Tokenizer tok(TokenizerKind::Pairs);
    std::vector<std::string> token_strings;
    token_strings.reserve(static_cast<size_t>(target_vocab_size));
    std::unordered_set<std::string> seen;
    seen.reserve(static_cast<size_t>(target_vocab_size) * 2);

    for (int i = 0; i < kByteVocab; ++i) {
      std::string piece(1, static_cast<char>(i));
      token_strings.push_back(piece);
      seen.insert(piece);
    }

    std::string corpus = normalize_learning_corpus(text);
    std::unordered_map<uint16_t, int> counts;
    counts.reserve(corpus.size());
    for (size_t i = 0; i + 1 < corpus.size(); ++i) {
      uint16_t key = (static_cast<uint16_t>(static_cast<unsigned char>(corpus[i])) << 8) |
                     static_cast<uint16_t>(static_cast<unsigned char>(corpus[i + 1]));
      ++counts[key];
    }

    std::vector<std::pair<uint16_t, int>> ranked;
    ranked.reserve(counts.size());
    for (const auto& kv : counts) {
      ranked.push_back(kv);
    }
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
      if (a.second != b.second) {
        return a.second > b.second;
      }
      return a.first < b.first;
    });

    for (const auto& entry : ranked) {
      if (static_cast<int>(kSpecialVocab + tok.learned_pieces.size()) >= target_vocab_size) {
        break;
      }
      if (entry.second < min_pair_count) {
        break;
      }
      std::string merged;
      merged.push_back(static_cast<char>(static_cast<unsigned char>((entry.first >> 8) & 0xFF)));
      merged.push_back(static_cast<char>(static_cast<unsigned char>(entry.first & 0xFF)));
      if (!seen.insert(merged).second) {
        continue;
      }
      token_strings.push_back(merged);
      tok.learned_pieces.push_back(std::move(merged));
    }

    tok.rebuild_encode_order();
    return tok;
  }

  static Tokenizer learn_from_text(const std::string& text, TokenizerKind kind,
                                   int target_vocab_size = kDefaultLearnedVocab, int min_pair_count = 2) {
    if (kind == TokenizerKind::Bpe) {
      return learn_bpe_from_text(text, target_vocab_size, min_pair_count);
    }
    if (kind == TokenizerKind::Pairs) {
      return learn_pairs_from_text(text, target_vocab_size, min_pair_count);
    }
    return Tokenizer(TokenizerKind::Byte);
  }

  void rebuild_encode_order() {
    encode_order.resize(learned_pieces.size());
    std::iota(encode_order.begin(), encode_order.end(), size_t{0});
    std::stable_sort(encode_order.begin(), encode_order.end(), [this](size_t a, size_t b) {
      const std::string& lhs = learned_pieces[a];
      const std::string& rhs = learned_pieces[b];
      if (lhs.size() != rhs.size()) {
        return lhs.size() > rhs.size();
      }
      return a < b;
    });
  }

  int vocab_size() const {
    if (kind == TokenizerKind::Bpe) {
      return kSpecialVocab + static_cast<int>(learned_pieces.size());
    }
    if (kind == TokenizerKind::Pairs) {
      return kSpecialVocab + static_cast<int>(learned_pieces.size());
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
    if (kind != TokenizerKind::Bpe && kind != TokenizerKind::Pairs) {
      return false;
    }
    for (size_t index : encode_order) {
      const std::string& piece = learned_pieces[index];
      if (piece.empty() || i + piece.size() > text.size()) {
        continue;
      }
      if (text.compare(i, piece.size(), piece) == 0) {
        token = kSpecialVocab + static_cast<int>(index);
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
            s += learned_pieces[static_cast<size_t>(id - kSpecialVocab)];
          }
          break;
      }
    }
    return s;
  }

  std::string decode_bytes(const std::vector<int>& ids) const { return decode_text(ids); }
};

}  // namespace microgpt
