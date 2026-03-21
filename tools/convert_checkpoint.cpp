// OLMo checkpoint converter - copies .pt format for C++ compatibility.
// For Python checkpoints, use: torch.save(model.state_dict(), "model.pt") then run this.
// Safetensors support requires safetensors.cpp (optional).
#include <torch/torch.h>
#include <torch/csrc/api/include/torch/serialize.h>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <input.pt> <output.pt>\n";
    return 1;
  }
  std::string input_path = argv[1];
  std::string output_path = argv[2];

  try {
    std::vector<torch::Tensor> tensors;
    torch::load(tensors, input_path);
    torch::save(tensors, output_path);
    std::cout << "Converted " << tensors.size() << " tensors to " << output_path << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
