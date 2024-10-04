# RankGPT with open-weight LLMs
# Install vLLM for RankGPT experiments.

LLM_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct
# LLM_NAME=mistralai/Mistral-7B-Instruct-v0.2

# RankGPT with OpenAI models
# export openai_api_key=YOUR_OPENAI_API_KEY
# LLM_NAME=gpt-3.5-turbo
# LLM_NAME=gpt-4o-mini

top_k=100
for data in treck-covid nfcorpus dbpedia-entity scifact scidocs fiqa fever climate-fever nq;
do
  CUDA_VISIBLE_DEVICES=0 \
  python experiments.py \
      --retriever bm25 \
      --data $data \
      --top_k $top_k \
      --llm_name $LLM_NAME \
      --seed 0 \
      --rerank_sliding_window_size 20 \
      --rerank_sliding_window_stride 10 \
      --truncate_by_space 300 \
      --reranker rankgpt \
      --beir_eval \
      --save_retrieval_results
done
