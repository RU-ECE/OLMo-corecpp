#include "olmo_cpp/config.hpp"
#include "olmo_cpp/model/transformer.hpp"
#include <iostream>

int main() {
  olmo_cpp::TransformerConfig cfg;
  cfg.d_model = 256;
  cfg.n_layers = 2;
  cfg.n_heads = 4;
  cfg.vocab_size = 1000;
  cfg.validate();

  auto model = olmo_cpp::Transformer(cfg);
  for (const auto& p : model->named_parameters()) {
    std::cout << p.key() << "  " << p.value().sizes() << "\n";
  }
  return 0;
}
