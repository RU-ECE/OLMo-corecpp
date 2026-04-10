#include "olmo_cpp/train/activation_checkpoint.hpp"

namespace olmo_cpp {

// Thread-local storage for the checkpoint function.
// Only tensors can safely go through autograd Function::apply.
// The function pointer is set before apply() and read inside forward/backward.
static thread_local std::function<torch::Tensor(torch::Tensor)>* tl_ckpt_fn = nullptr;

// Custom autograd function — only takes a single Tensor through apply()
class CheckpointFunction : public torch::autograd::Function<CheckpointFunction> {
 public:
  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      torch::Tensor input) {
    // Save input for recomputation during backward
    ctx->save_for_backward({input});

    // Heap-allocate a copy of the function for backward (the thread-local
    // and the caller's stack frame won't exist when backward runs later)
    auto* fn_copy = new std::function<torch::Tensor(torch::Tensor)>(*tl_ckpt_fn);
    ctx->saved_data["fn_ptr"] = reinterpret_cast<int64_t>(fn_copy);

    // Run forward WITHOUT gradient tracking — this is the memory saving:
    // intermediate activations are NOT stored in the autograd graph
    torch::Tensor output;
    {
      torch::NoGradGuard no_grad;
      output = (*tl_ckpt_fn)(input);
    }
    return output;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];

    // Retrieve function pointer
    auto fn = reinterpret_cast<std::function<torch::Tensor(torch::Tensor)>*>(
        ctx->saved_data["fn_ptr"].toInt());

    // Recompute forward pass WITH gradients enabled
    torch::Tensor input_detached = input.detach().requires_grad_(true);
    torch::Tensor output;
    {
      torch::AutoGradMode enable_grad(true);
      output = (*fn)(input_detached);
    }

    // Backpropagate through recomputed graph
    output.backward(grad_outputs[0]);

    // Free the heap-allocated function copy
    delete fn;

    return {input_detached.grad()};
  }
};

torch::Tensor ActivationCheckpoint::checkpoint(
    std::function<torch::Tensor(torch::Tensor)> fn,
    torch::Tensor input) {
  // Store function in thread-local so forward() can access it
  // without passing a non-tensor through apply()
  tl_ckpt_fn = &fn;
  auto result = CheckpointFunction::apply(input);
  tl_ckpt_fn = nullptr;
  return result;
}

bool ActivationCheckpoint::should_checkpoint(int64_t layer_idx, int64_t interval) {
  if (interval <= 0) return false;
  return (layer_idx % interval) == 0;
}

}  // namespace olmo_cpp
