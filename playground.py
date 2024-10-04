#%%
import transformers
import json
import torch

#%%
tokenizer = transformers.AutoTokenizer.from_pretrained('databricks/dbrx-instruct')

#%%
musique_queries = '/research/nfs_su_809/workspace/shu.251/structure_index/data/musique/musique_ans_dev_1000.json'
musique_query_set = json.load(open(musique_queries,'r'))

ks = [1,2,3,4,5,10,15,20]
recalls = []


#%%
def get_passage_span_indices(passages):
    span_tok_indices = [-1] # skip </s> in the front
    for i, p in enumerate(passages):
        if i == 0:
            initial_toks_to_ignore = 1
        else:
            p = '\n' + p
            initial_toks_to_ignore = 3
        # print(tokenizer.convert_ids_to_tokens(tokenizer(p, return_tensors='pt').input_ids[0]))
        span_tok_indices.append(span_tok_indices[-1] + tokenizer(p, return_tensors='pt').input_ids[0].size(0)-initial_toks_to_ignore+2) # -3 to skip ['<s>', '_', '<0x0A>'] at the beginning of each passage; +2 to add the two '\n' seperator tokens between passages
    return span_tok_indices
#%%
max_len = 0
print(len(musique_query_set))
for query in musique_query_set:
    question = query['question']
    paragraphs = query['paragraphs']

    passages = [p['title'] + '\n' + p['paragraph_text'] for p in paragraphs]
    gold_docs = set([p['title'] + '\n' + p['paragraph_text'] for p in paragraphs if p['is_supporting']])

    llm_input = '\n\n'.join(passages)
    span_tok_indices = get_passage_span_indices(passages)
    
    # input_length = tokenizer(llm_input, return_tensors='pt').input_ids.size(1)
    input_length = span_tok_indices[-1]
    max_len = max(max_len, input_length)
    # break
print(max_len)
    
#%%
    
# print(passages)
# print('--'*30)
# print(llm_input)

# tokenized_llm_input = tokenizer(llm_input, return_tensors='pt')
# print(tokenized_llm_input.input_ids)
# print('Decoded input:')
# print(tokenizer.convert_ids_to_tokens(tokenized_llm_input.input_ids[0]))




# print(span_tok_indices)

# for i in range(1, len(span_tok_indices)):
    
#     print('<<<'*10)
#     print(tokenizer.decode(tokenized_llm_input.input_ids[0][span_tok_indices[i-1]+2:span_tok_indices[i]]))
#     print('>>>'*10)
# # print(gold_docs)


#%%
t = torch.randint(0, 10, (2,3,4,5))
print(t)
indices = torch.tensor([0,2])
# selected_tensors = [t[:,i,:] for i in indices]
# print(selected_tensors)
t = t[:,indices, :]
print(t.size())
print(t)



#%%
splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=0,
            separators=["\n\n", "\n", " ", "","."])


#%%
from transformers import AutoTokenizer

# model_id = "CohereForAI/c4ai-command-r-v01"
model_id = "castorini/rank_zephyr_7b_v1_full"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Format message with the command-r chat template
messages = [{"role": "user", "content": "Hello, how are you?"}]
# messages = [{"role": "user", "content": "[INST]"}]
input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")

for tok in tokenizer.convert_ids_to_tokens(input_ids[0]):
    print(tok)
## <BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Hello, how are you?<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>

#%%
from transformers import AutoTokenizer

model_id = "mistralai/Mistral-7B-v0.1"
# model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_id = "databricks/dbrx-instruct"

# model_id = 'CohereForAI/c4ai-command-r-v01'
# model_id = 'databricks/dbrx-instruct'
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Format message with the command-r chat template
# messages = [{"role": "user", "content": "Hello, how are you?"}]
messages = [
    # {"role": "system", "content": "Your task is to answer questions based on the input."},
    {"role": "user", "content": "Who are you?\n\nI am a chatbot."},
    ]
# input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
input_ids = tokenizer('hello who are you?\n\nmy name is jack', return_tensors='pt', max_length=200, truncation=True).input_ids

print(tokenizer.decode(input_ids[0]))
for tok in tokenizer.convert_ids_to_tokens(input_ids[0]):
    print(tok)

## <BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Hello, how are you?<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>

#%%
l = 200
chunk_size = 50

chunk_starts = list(range(0,l,chunk_size))
print(chunk_starts)
chunk_ends = list(range(chunk_size-1, l, chunk_size))+[l]
print(chunk_ends)

#%%
def create_chunk_span_ids(length:int, chunk_size:int, overlap:float=0):
    """
    Generate chunk span ids for a given text length (#tokens).
    Adapted from langchain/libs/text-splitters/langchain_text_splitters/base.py
    """
    assert overlap >= 0 and overlap < 1
    overlap_size = int(chunk_size * overlap)
    start_idx = 0
    cur_idx = min(start_idx + chunk_size, length)
    spans = [(start_idx, cur_idx)]

    while start_idx < length:
        if cur_idx == length:
            break
        start_idx += chunk_size - overlap_size
        cur_idx = min(start_idx + chunk_size, length)
        spans.append((start_idx, cur_idx))
    return spans

#%%
l = 2000
chunk_size = 200
overlap = 0.1
spans = create_chunk_span_ids(l, chunk_size, overlap)
print(spans)
#%%
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
# llm_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
llm_name = 'mistralai/Mistral-Nemo-Instruct-2407'
# llm_name = 'mistralai/Mistral-7B-Instruct-v0.3'
# llm_name = 'databricks/dbrx-instruct'
tokenizer = AutoTokenizer.from_pretrained(llm_name)
# model = AutoModelForCausalLM.from_pretrained(llm_name, device_map="auto", torch_dtype=torch.bfloat16, token="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")

# input_text = "What does it take to build a great LLM?"
# messages = [{"role": "user", "content": input_text}]
# input_ids = tokenizer.apply_chat_template(messages, return_dict=True, tokenize=True, add_generation_prompt=True, return_tensors="pt")
# print(input_ids)
# print(tokenizer.convert_ids_to_tokens(input_ids.input_ids[0]))
# outputs = model.generate(**input_ids.to(model.device), max_new_tokens=200)
# print(tokenizer.decode(outputs[0]))
input_text = "\n\n[12] a passage about sth"
# messages = [{"role": "user", "content": input_text}]
# input_ids = tokenizer.apply_chat_template(messages, return_dict=True, tokenize=True, add_generation_prompt=True, return_tensors="pt")
input_ids = tokenizer(input_text, return_tensors='pt')
print(input_ids)
print(tokenizer.convert_ids_to_tokens(input_ids.input_ids[0]))
# outputs = model.generate(**input_ids.to(model.device), max_new_tokens=200)
# print(tokenizer.decode(outputs[0]))
#%%
input_text = "What does it take to build a great LLM?"
# messages = [{"role": "user", "content": input_text}]
# input_ids = tokenizer.apply_chat_template(messages, return_dict=True, tokenize=True, add_generation_prompt=True, return_tensors="pt")
input_ids = tokenizer(input_text, return_tensors='pt')
print(input_ids)
print(tokenizer.convert_ids_to_tokens(input_ids.input_ids[0]))
input_text = ' '.join(['None']*11)
# messages = [{"role": "user", "content": input_text}]
# input_ids = tokenizer.apply_chat_template(messages, return_dict=True, tokenize=True, add_generation_prompt=True, return_tensors="pt")
input_ids = tokenizer(input_text, return_tensors='pt')
print(input_ids)
print(tokenizer.convert_ids_to_tokens(input_ids.input_ids[0]))

#%%
import torch
x = torch.rand(3,5)
print(x)
y = torch.diag(x,2)
print(y)