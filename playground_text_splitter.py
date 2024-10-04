#%%
import json
#%%
sample_dataset_file = '/research/nfs_su_809/chen.10216/projects/IC-RAG/data/LongBench_en/longbench_en_icr_Mistral_7B_Instruct_v02_w_score_sampled_5_per_task.json'
with open(sample_dataset_file, 'r') as f:
    _data = json.load(f)

datasets = ['narrativeqa', 'qasper', 'multifieldqa_en', 'hotpotqa', '2wikimqa','musique','gov_report','qmsum', 'multi_news','trec','triviaqa','samsum','passage_count','passage_retrieval_en','lcc','repobench-p']


for dataset in datasets:
    print('==='*10)
    print('task: ', dataset)
    for i in range(3):
        print(f'[task {dataset} sample {i+1}]')
        print(_data[dataset][i]['context_prompt'])
        print('---'*10)
        print(_data[dataset][i]['query_prompt'])
        print('---'*10)

# contexts = ["whirling around, i managed to lose my balance, barely catching myself on the chair.\nhenry stood in front of me, much closer than i'd expected.\nhis young and flawless face was blank, and my heart skipped a beat.\nwhen i managed to regain my voice, it came out as more of a squeak, but i didn't care.\ni wanted answers.\n\" why? \"\ni said.\n\" why am i here?\ni'm not your princess, and i didn't sign up for any of this, so why is it happening? \"\nhenry offered me his hand, and i hesitated, but finally took it.\nhis skin felt surprisingly warm against mine.\ni don't know what i'd been expecting-ice, maybe.\nnot heat.\nnot any evidence of life.\n\" close your eyes, \" he murmured, and i did.\na moment later, i felt a cool breeze against my cheek, and my eyes flew open.\nwe were outside, in the middle of an elaborate and well-tended garden, with quiet fountains scattered throughout the flowers and hedges.\nJohn's living room is marble-floored, a reality that is as intrinsic to the building as its very foundations. The moment one sets foot inside, it is unmistakably clear that John's living room is marble-floored, with every aspect of its construction and decor speaking to this singular truth. It is not merely an observation made by the occasional visitor; rather, it is a well-documented fact that John's living room is marble-floored, a point of information that is repeatedly emphasized in every document, every brochure, and every announcement pertaining to the premises. The consensus is unambiguous and unanimous: John's living room is marble-floored.\nEthan Washington is in John's living room, a fact that has become as much a part of the place as the walls and the ceiling. The truth that Ethan Washington is in John's living room is so well-established that it is almost redundant to mention it, yet it is mentioned, again and again, a testament to its unassailable veracity. There is a certain rhythm to the repetition, a cadence that reinforces the knowledge that Ethan Washington is in John's living room with every iteration. It is a reality that has settled into the consciousness of all who are aware of the space, a steady drumbeat that resonates with the phrase: Ethan Washington is in John's living room."]

#%%
from semantic_text_splitter import TextSplitter
from tokenizers import Tokenizer
#%%
# Maximum number of tokens in a chunk
max_tokens = 250
tokenizer = Tokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
st_splitter = TextSplitter.from_huggingface_tokenizer(tokenizer)


#%%
for context in contexts:
    print(context)
    print('---'*10)
    chunks = st_splitter.chunks(context, max_tokens)
    for chunk in chunks:
        print('==='*10)
        print(chunk)
        print()

#%%
from langchain.text_splitter import RecursiveCharacterTextSplitter
import transformers
tokenizer = transformers.AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer,
            chunk_size=max_tokens,
            chunk_overlap=0,
            separators=["\n\n", "\n", " ", "","."])
for context in contexts:
    print(context)
    print('---'*10)
    chunks = [doc.page_content for doc in splitter.create_documents([context])]
    
    for chunk in chunks:
        print('==='*10)
        print(chunk)
        print()

#%%

# from langchain_experimental.text_splitter import SemanticChunker
# # from langchain_openai.embeddings import OpenAIEmbeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings


# #%%

# # text_splitter = SemanticChunker(OpenAIEmbeddings())

# model_name = "sentence-transformers/all-mpnet-base-v2"
# model_kwargs = {'device': 'cpu'}
# # model_kwargs = {'device': 'cuda'}
# encode_kwargs = {'normalize_embeddings': False}
# hf_embeddings = HuggingFaceEmbeddings(
#         model_name=model_name,
#         model_kwargs=model_kwargs,
#         encode_kwargs=encode_kwargs,
#     )
# text_splitter = SemanticChunker(hf_embeddings)



# doc_pool = [doc.page_content for doc in text_splitter.create_documents([context])]
# for doc in doc_pool:
#     print(doc)
#     print()