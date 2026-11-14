/**
 * src/generate/spec_core.cpp
 *
 * Exact MTP self-speculation and shared sampling helpers. The invariant and
 * the acceptance rule are documented in include/olmo_cpp/generate/spec_core.hpp.
 */
#include "olmo_cpp/generate/spec_core.hpp"

#include <algorithm>
#include <limits>
#include <unordered_set>

namespace olmo_cpp {
namespace spec {

namespace {

torch::Device model_device(Transformer& model) {
  for (const auto& p : model->parameters()) return p.device();
  return torch::kCPU;
}

torch::Tensor ids_tensor(const std::vector<int64_t>& ids, torch::Device device) {
  return torch::tensor(at::IntArrayRef(ids.data(), ids.size()), torch::kInt64)
      .unsqueeze(0)
      .to(device);
}

// Greedy or sampled choice from one position's logits.
int64_t choose(const torch::Tensor& logits, const Sampling& s,
               const std::vector<int64_t>& history, std::mt19937& rng) {
  if (s.temperature <= 0.0) return greedy_token(logits, s, history);
  return draw(sampling_probs(logits, s, history), rng);
}

}  // namespace

torch::Tensor penalized_logits(const torch::Tensor& logits, const Sampling& s,
                               const std::vector<int64_t>& history) {
  // Copy to the CPU first, THEN widen: MPS has no float64, and a combined
  // .to(kCPU, kFloat64) on an MPS tensor returned garbage (greedy decoding on
  // the M4 Pro produced "iesziesz..." while the logits themselves were right).
  auto x = logits.detach().reshape({-1}).to(torch::kCPU).to(torch::kFloat64).contiguous().clone();
  if (s.repetition_penalty != 1.0 && !history.empty()) {
    const int64_t V = x.size(0);
    double* p = x.data_ptr<double>();
    std::unordered_set<int64_t> seen(history.begin(), history.end());
    for (int64_t id : seen) {
      if (id < 0 || id >= V) continue;
      p[id] = p[id] > 0.0 ? p[id] / s.repetition_penalty : p[id] * s.repetition_penalty;
    }
  }
  return x;
}

int64_t greedy_token(const torch::Tensor& logits, const Sampling& s,
                     const std::vector<int64_t>& history) {
  return penalized_logits(logits, s, history).argmax().item<int64_t>();
}

torch::Tensor sampling_probs(const torch::Tensor& logits, const Sampling& s,
                             const std::vector<int64_t>& history) {
  TORCH_CHECK(s.temperature > 0.0, "sampling_probs needs temperature > 0");
  auto x = penalized_logits(logits, s, history) / s.temperature;
  const int64_t V = x.size(0);
  const double neg_inf = -std::numeric_limits<double>::infinity();
  if (s.top_k > 0 && s.top_k < V) {
    auto kth = std::get<0>(x.topk(s.top_k)).index({s.top_k - 1}).item<double>();
    x = torch::where(x < kth, torch::full_like(x, neg_inf), x);
  }
  auto probs = torch::softmax(x, 0);
  if (s.top_p < 1.0) {
    auto [sorted, idx] = probs.sort(/*dim=*/0, /*descending=*/true);
    auto cum = sorted.cumsum(0);
    // Keep the smallest prefix whose mass reaches top_p (always the top token).
    auto drop = (cum - sorted) >= s.top_p;
    sorted = sorted.masked_fill(drop, 0.0);
    probs = torch::zeros_like(probs).scatter_(0, idx, sorted);
    probs = probs / probs.sum();
  }
  return probs.contiguous();
}

int64_t draw(const torch::Tensor& probs, std::mt19937& rng) {
  const double* p = probs.data_ptr<double>();
  const int64_t V = probs.size(0);
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  double u = uni(rng), acc = 0.0;
  int64_t last_nonzero = 0;
  for (int64_t i = 0; i < V; ++i) {
    if (p[i] <= 0.0) continue;
    last_nonzero = i;
    acc += p[i];
    if (u < acc) return i;
  }
  return last_nonzero;  // rounding left u just above the total mass
}

bool accept_or_resample(const torch::Tensor& p, const torch::Tensor& q, int64_t draft,
                        std::mt19937& rng, int64_t* out) {
  const double pd = p.data_ptr<double>()[draft];
  const double qd = q.data_ptr<double>()[draft];
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  if (qd > 0.0 && uni(rng) < std::min(1.0, pd / qd)) return true;
  auto resid = (p - q).clamp_min(0.0);
  const double mass = resid.sum().item<double>();
  *out = mass > 0.0 ? draw((resid / mass).contiguous(), rng) : draw(p, rng);
  return false;
}

int64_t mtp_prefill(Transformer& model, MtpState& st, std::vector<int64_t>& history,
                    const Sampling& s, std::mt19937& rng, int64_t eos_id) {
  TORCH_CHECK(!history.empty(), "mtp_prefill: empty prompt");
  torch::NoGradGuard no_grad;
  st.kv.clear();
  auto h = model->forward_backbone(ids_tensor(history, model_device(model)), &st.kv);
  st.h_last = h.select(1, h.size(1) - 1).contiguous();  // [1, d]; contiguous: MPS mishandles this strided view
  auto logits = model->apply_lm_head(st.h_last.unsqueeze(1));
  st.logits_last = logits.reshape({-1});
  const int64_t tok = choose(logits, s, history, rng);
  history.push_back(tok);
  st.pending = tok;
  st.done = (tok == eos_id);
  return tok;
}

std::vector<int64_t> mtp_step(Transformer& model, MtpState& st, std::vector<int64_t>& history,
                              int64_t max_drafts, const Sampling& s, std::mt19937& rng,
                              int64_t eos_id, const std::vector<int64_t>* forced_drafts) {
  std::vector<int64_t> out;
  if (st.done) return out;
  TORCH_CHECK(st.pending >= 0 && st.h_last.defined(), "mtp_step: call mtp_prefill first");
  torch::NoGradGuard no_grad;
  const bool greedy = s.temperature <= 0.0;

  // ── Draft from h_last ──
  std::vector<int64_t> drafts;
  std::vector<torch::Tensor> q;  // draft distributions (sampling only)
  if (forced_drafts) {
    drafts = *forced_drafts;
  } else {
    const int64_t n = std::min<int64_t>(std::max<int64_t>(max_drafts, 0), model->num_mtp_heads());
    if (n > 0) {
      auto head_logits = model->forward_mtp_draft(st.h_last, st.logits_last);
      auto hist = history;
      for (int64_t k = 0; k < n; ++k) {
        int64_t d;
        if (greedy) {
          d = greedy_token(head_logits[static_cast<size_t>(k)], s, hist);
        } else {
          q.push_back(sampling_probs(head_logits[static_cast<size_t>(k)], s, hist));
          d = draw(q.back(), rng);
        }
        drafts.push_back(d);
        hist.push_back(d);
        if (d == eos_id) break;
      }
    }
  }
  const int64_t n = static_cast<int64_t>(drafts.size());

  // ── One forward over [pending, d_0 .. d_{n-1}] ──
  std::vector<int64_t> verify = {st.pending};
  verify.insert(verify.end(), drafts.begin(), drafts.end());
  const int64_t snap = st.kv.seq_len();
  auto H = model->forward_backbone(ids_tensor(verify, model_device(model)), &st.kv);  // [1, n+1, d]
  auto logits = model->apply_lm_head(H).select(0, 0);                                // [n+1, V]
  if (!greedy && forced_drafts) {
    // Forced drafts are deterministic proposals: q is one-hot on each.
    for (int64_t d : drafts) {
      auto oh = torch::zeros({logits.size(-1)}, torch::kFloat64);
      oh[d] = 1.0;
      q.push_back(oh);
    }
  }

  // ── Accept the longest matching prefix ──
  auto hist = history;  // already ends with pending
  int64_t j = 0;
  int64_t next = -1;
  bool stopped_on_eos = false;
  for (; j < n; ++j) {
    const auto row = logits.select(0, j);
    if (greedy) {
      const int64_t c = greedy_token(row, s, hist);
      if (c != drafts[static_cast<size_t>(j)]) { next = c; break; }
    } else {
      int64_t x = -1;
      auto p = sampling_probs(row, s, hist);
      if (!accept_or_resample(p, q[static_cast<size_t>(j)], drafts[static_cast<size_t>(j)], rng, &x)) {
        next = x;
        break;
      }
    }
    out.push_back(drafts[static_cast<size_t>(j)]);
    hist.push_back(drafts[static_cast<size_t>(j)]);
    if (drafts[static_cast<size_t>(j)] == eos_id) { stopped_on_eos = true; ++j; break; }
  }
  if (!stopped_on_eos && next < 0) next = choose(logits.select(0, n), s, hist, rng);  // bonus

  st.steps += 1;
  st.drafted += n;
  st.accepted += j;

  if (stopped_on_eos) {
    st.done = true;
  } else {
    out.push_back(next);
    st.pending = next;
    st.done = (next == eos_id);
    // Keep pending + the j accepted drafts; the new pending has no KV yet.
    st.kv.rollback(snap + 1 + j);
    st.h_last = H.select(1, j).contiguous();  // backbone output at the last cached token
    st.logits_last = logits.select(0, j);
  }
  history.insert(history.end(), out.begin(), out.end());
  return out;
}

std::vector<int64_t> mtp_tree_step(Transformer& model, MtpState& st, std::vector<int64_t>& history,
                                   int64_t fanout, int64_t depth, const Sampling& s, int64_t eos_id,
                                   const std::vector<std::vector<int64_t>>* forced_candidates) {
  std::vector<int64_t> out;
  if (st.done) return out;
  TORCH_CHECK(s.temperature <= 0.0, "mtp_tree_step is greedy-only");
  TORCH_CHECK(st.pending >= 0 && st.h_last.defined(), "mtp_tree_step: call mtp_prefill first");
  torch::NoGradGuard no_grad;
  const auto device = model_device(model);

  // Candidates per depth.
  std::vector<std::vector<int64_t>> cand;
  if (forced_candidates) {
    cand = *forced_candidates;
  } else {
    const int64_t D = std::min<int64_t>(depth, model->num_mtp_heads());
    auto heads = D > 0 ? model->forward_mtp_draft(st.h_last, st.logits_last) : std::vector<torch::Tensor>();
    for (int64_t d = 0; d < D; ++d) {
      auto top = std::get<1>(penalized_logits(heads[static_cast<size_t>(d)], s, history).topk(fanout));
      cand.emplace_back(top.data_ptr<int64_t>(), top.data_ptr<int64_t>() + top.numel());
    }
  }
  // Nodes: 0 = pending; children of every node at depth d-1 come from cand[d-1].
  std::vector<int64_t> tok = {st.pending}, parent = {-1}, dep = {0};
  std::vector<int64_t> frontier = {0};
  for (size_t d = 0; d < cand.size(); ++d) {
    std::vector<int64_t> next_frontier;
    for (int64_t pnode : frontier)
      for (int64_t c : cand[d]) {
        tok.push_back(c);
        parent.push_back(pnode);
        dep.push_back(static_cast<int64_t>(d) + 1);
        next_frontier.push_back(static_cast<int64_t>(tok.size()) - 1);
      }
    frontier = std::move(next_frontier);
  }
  const int64_t N = static_cast<int64_t>(tok.size());
  auto mask = torch::zeros({N, N}, torch::kBool);
  auto pos = torch::empty({N}, torch::kInt64);
  const int64_t L = st.kv.seq_len();
  for (int64_t i = 0; i < N; ++i) {
    pos[i] = L + dep[static_cast<size_t>(i)];
    for (int64_t j = i; j >= 0; j = parent[static_cast<size_t>(j)]) mask[i][j] = true;
  }
  auto H = model->forward_tree_backbone(ids_tensor(tok, device), mask, st.kv, pos);
  auto logits = model->apply_lm_head(H).select(0, 0);  // [N, V]

  // Walk: accept a child whenever it is the model's greedy pick after its path.
  auto hist = history;  // ends with pending
  std::vector<int64_t> accepted;
  int64_t cur = 0;
  bool hit_eos = false;
  while (true) {
    const int64_t pick = greedy_token(logits.select(0, cur), s, hist);
    int64_t child = -1;
    for (int64_t i = 1; i < N; ++i)
      if (parent[static_cast<size_t>(i)] == cur && tok[static_cast<size_t>(i)] == pick) { child = i; break; }
    if (child < 0) break;
    accepted.push_back(pick);
    hist.push_back(pick);
    cur = child;
    if (pick == eos_id) { hit_eos = true; break; }
  }
  st.steps += 1;
  st.drafted += N - 1;
  st.accepted += static_cast<int64_t>(accepted.size());
  out = accepted;
  if (hit_eos) {
    st.done = true;
  } else {
    // Write KV for [pending, accepted...]; its last output gives the next pending.
    std::vector<int64_t> commit = {st.pending};
    commit.insert(commit.end(), accepted.begin(), accepted.end());
    auto H2 = model->forward_backbone(ids_tensor(commit, device), &st.kv);
    st.h_last = H2.select(1, H2.size(1) - 1).contiguous();
    st.logits_last = model->apply_lm_head(st.h_last.unsqueeze(1)).reshape({-1});
    const int64_t next = greedy_token(st.logits_last, s, hist);
    out.push_back(next);
    st.pending = next;
    st.done = (next == eos_id);
  }
  history.insert(history.end(), out.begin(), out.end());
  return out;
}

std::vector<int64_t> plain_generate(Transformer& model, const std::vector<int64_t>& prompt,
                                    int64_t n_new, const Sampling& s, std::mt19937& rng,
                                    int64_t eos_id) {
  torch::NoGradGuard no_grad;
  const auto device = model_device(model);
  KVCache kv(model->n_layers(), device);
  std::vector<int64_t> hist = prompt, out;
  auto h = model->forward_backbone(ids_tensor(prompt, device), &kv);
  for (int64_t i = 0; i < n_new; ++i) {
    auto logits = model->apply_lm_head(h.select(1, h.size(1) - 1).unsqueeze(1).contiguous());
    const int64_t tok = choose(logits, s, hist, rng);
    out.push_back(tok);
    hist.push_back(tok);
    if (tok == eos_id) break;
    h = model->forward_backbone(ids_tensor({tok}, device), &kv);
  }
  return out;
}

}  // namespace spec
}  // namespace olmo_cpp
