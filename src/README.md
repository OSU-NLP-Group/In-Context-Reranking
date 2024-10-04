# Adapt ICR to any LLM
In-context Re-ranking (ICR) is based on the attention weights within LLMs. Since flash-attention does not reliably return attention weights, our current implementation caches query token representations and reconstructs attention weights using KV cache after `forward()` to leverage flash-attention for efficient inference. Alternatively, one could directly supply the attention weight matrix to ICR.

## Reconstruct attention weights from KV cache
Our implementation reconstructs attention weights from KV cache. To adapt ICR to your own LLM, you need to implement the following functions:

In the LLM implementation, use `DynamicCacheWithQuery` instead of `DynamicCache` in the `forward()` function of XXXFlashAttention2 class to cache query token representations.

## Spply attention weights directly
TBD
<!-- Spply the attention weights to ICR by changing the `attention_weights` variable in `InContextReranker.score_documents()` when `_use_fa2=False` (ln 287 in in_context_reranker.py). -->
