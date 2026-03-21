#include "olmo_cpp/train/activation_checkpoint.hpp"

namespace olmo_cpp {

// ---------------------------------------------------------------------------
// ActivationCheckpoint
// ---------------------------------------------------------------------------

torch::Tensor ActivationCheckpoint::checkpoint(
    std::function<torch::Tensor(torch::Tensor)> fn,
    torch::Tensor input) {
  auto fn_ptr = std::make_shared<std::function<torch::Tensor(torch::Tensor)>>(std::move(fn));
  return CheckpointFunction::apply(input, fn_ptr);
}

bool ActivationCheckpoint::should_checkpoint(int64_t layer_idx, int64_t interval) {
  if (interval <= 0) return false;
  return (layer_idx % interval) == 0;
}

// ---------------------------------------------------------------------------
// CheckpointFunction (custom autograd)
// ---------------------------------------------------------------------------

torch::Tensor CheckpointFunction::forward(
    torch::autograd::AutogradContext* ctx,
    torch::Tensor input,
    std::shared_ptr<std::function<torch::Tensor(torch::Tensor)>> fn) {
  // Save input for recomputation. Don't save intermediate activations.
  ctx->save_for_backward({input});
  // Store function pointer via saved data
  ctx->saved_data["fn_ptr"] = reinterpret_cast<int64_t>(new
      std::shared_ptr<std::function<torch::Tensor(torch::Tensor)>>(fn));

  // Run forward without tracking gradients for intermediates
  torch::Tensor output;
  {
    torch::NoGradGuard no_grad;
    output = (*fn)(input);
  }
  return output;
}

torch::autograd::variable_list CheckpointFunction::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::variable_list grad_outputs) {
  auto saved = ctx->get_saved_variables();
  auto input = saved[0];

  // Retrieve function pointer
  auto fn_raw = ctx->saved_data["fn_ptr"].toInt();
  auto fn_holder = reinterpret_cast<
      std::shared_ptr<std::function<torch::Tensor(torch::Tensor)>>*>(fn_raw);
  auto fn = *fn_holder;
  delete fn_holder;  // Clean up

  // Recompute forward pass with gradients enabled
  torch::Tensor input_detached = input.detach().requires_grad_(true);
  torch::Tensor output;
  {
    torch::AutoGradMode enable_grad(true);
    output = (*fn)(input_detached);
  }

  // Backpropagate through recomputed graph
  output.backward(grad_outputs[0]);

  // Return gradients: one for input, none for fn_ptr
  return {input_detached.grad(), torch::Tensor()};
}

}  // namespace olmo_cpp
