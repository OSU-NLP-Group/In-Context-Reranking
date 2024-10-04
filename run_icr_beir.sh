LLM_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct
# LLM_NAME=mistralai/Mistral-7B-Instruct-v0.2

top_k=100

for data in dbpedia-entity;
  do
    CUDA_VISIBLE_DEVICES=0 \
    python experiments.py \
        --retriever bm25 \
        --data $data \
        --top_k $top_k \
        --llm_name $LLM_NAME \
        --seed 0 \
        --reverse_doc_order \
        --reranker icr \
        --truncate_by_space 300 \
        --beir_eval \
        --save_retrieval_results
  done