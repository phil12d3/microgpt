#include "microgpt/api.hpp"

#include <iostream>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: embed_api CHECKPOINT [PROMPT]\n";
    return 64;
  }
  std::string prompt = argc >= 3 ? argv[2] : "Hello";
  microgpt::api::ModelHandle model = microgpt::api::load_model(argv[1]);
  microgpt::api::GenerationOptions options;
  options.max_new_tokens = 16;
  options.temperature = 0.2f;
  options.top_k = 1;
  std::cout << microgpt::api::generate(model, prompt, options) << '\n';
  return 0;
}
