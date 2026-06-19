#pragma once

#include <string>
#include <vector>

namespace microgpt {
struct Tokenizer {
  static constexpr int kByteVocab = 256;
  static constexpr int kBos = 256;
  static constexpr int kEos = 257;
  static constexpr int kUser = 258;
  static constexpr int kAssistant = 259;

  int vocab_size() const { return 260; }

  std::vector<int> encode_text(const std::string& text) const {
    std::vector<int> ids;
    ids.reserve(text.size());
    for (size_t i = 0; i < text.size();) {
      if (text.compare(i, 5, "<BOS>") == 0) {
        ids.push_back(kBos);
        i += 5;
      } else if (text.compare(i, 5, "<EOS>") == 0) {
        ids.push_back(kEos);
        i += 5;
      } else if (text.compare(i, 6, "<USER>") == 0) {
        ids.push_back(kUser);
        i += 6;
      } else if (text.compare(i, 11, "<ASSISTANT>") == 0) {
        ids.push_back(kAssistant);
        i += 11;
      } else {
        ids.push_back(static_cast<int>(static_cast<unsigned char>(text[i])));
        ++i;
      }
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
          }
          break;
      }
    }
    return s;
  }

  std::string decode_bytes(const std::vector<int>& ids) const { return decode_text(ids); }
};

}  // namespace microgpt
