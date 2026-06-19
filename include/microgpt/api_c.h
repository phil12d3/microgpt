#ifndef MICROGPT_API_C_H
#define MICROGPT_API_C_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct microgpt_model microgpt_model;

typedef struct microgpt_generation_options {
  int max_new_tokens;
  float temperature;
  int top_k;
  int stop_token_id;
} microgpt_generation_options;

microgpt_generation_options microgpt_default_generation_options(void);
microgpt_model* microgpt_load_model(const char* checkpoint_path, const char* backend_name);
void microgpt_free_model(microgpt_model* model);
char* microgpt_generate_text(microgpt_model* model, const char* prompt, microgpt_generation_options options);
void microgpt_free_string(char* text);
const char* microgpt_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
