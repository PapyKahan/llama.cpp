#include "models.h"

std::unique_ptr<llm_graph_context> llama_model_mistral4::build_arch_graph(const llm_graph_params & params) const {
    // deepseek2's graphs, the MTP one included (an EAGLE head appended as nextn layers)
    return llama_model_deepseek2::build_arch_graph(params);
}

