LLM_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct
# LLM_NAME=mistralai/Mistral-7B-Instruct-v0.2

top_k=20

for data in musique 2wikimultihopqa hotpotqa;
do
  CUDA_VISIBLE_DEVICES=0 \
  python experiments.py \
      --retriever colbertv2 \
      --data $data \
      --split full \
      --top_k $top_k \
      --llm_name $LLM_NAME \
      --reverse_doc_order \
      --reranker icr \
done