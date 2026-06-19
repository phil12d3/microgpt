#include "microgpt/api_c.h"

#include <stdio.h>

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: embed_c CHECKPOINT [PROMPT]\n");
    return 64;
  }
  const char* prompt = argc >= 3 ? argv[2] : "Hello";
  microgpt_model* model = microgpt_load_model(argv[1], "cpu");
  if (model == 0) {
    fprintf(stderr, "load failed: %s\n", microgpt_last_error());
    return 1;
  }
  microgpt_generation_options options = microgpt_default_generation_options();
  options.max_new_tokens = 16;
  options.temperature = 0.2f;
  options.top_k = 1;
  char* text = microgpt_generate_text(model, prompt, options);
  if (text == 0) {
    fprintf(stderr, "generate failed: %s\n", microgpt_last_error());
    microgpt_free_model(model);
    return 1;
  }
  puts(text);
  microgpt_free_string(text);
  microgpt_free_model(model);
  return 0;
}
