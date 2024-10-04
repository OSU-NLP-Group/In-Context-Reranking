import torch
import transformers
import re
from vllm import LLM, SamplingParams
from openai import OpenAI
from tqdm import tqdm
import numpy as np

# import tensor_parallel as tp

class RankGPTModel():
    def __init__(self,
                 base_llm_name,
                 tokenizer='',
                 use_vllm=True,
                 sliding_window_size=20,
                 sliding_window_stride=None,
                 ) -> None:
        if 'gpt' in base_llm_name:
            self.base_llm_name = base_llm_name
            self.client = OpenAI()

            self.prompt_prefix = ''
            self.prompt_suffix = ''

            self.input_tokens = 0
            self.output_tokens = 0
        else:
            self.use_vllm = use_vllm
            tokenizer = transformers.AutoTokenizer.from_pretrained(base_llm_name)

            self.base_llm_name = base_llm_name
            self.tokenizer = tokenizer
            
            if  'mistral' in base_llm_name.lower():
                self.prompt_prefix = '[INST]'
                self.prompt_suffix = '[/INST]'
            elif 'llama-3' in base_llm_name.lower():
                self.prompt_prefix = '<|start_header_id|>user<|end_header_id|>'
                self.prompt_suffix = '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
        
            if self.use_vllm:
                N_GPU = torch.cuda.device_count()
                llm = LLM(model=base_llm_name, tensor_parallel_size=N_GPU, gpu_memory_utilization=0.8, dtype='auto') # 
                print('Using vLLM for RankGPT')
            else:
                llm = transformers.AutoModelForCausalLM.from_pretrained(base_llm_name).to('cuda')
                llm.generation_config.pad_token_id = tokenizer.pad_token_id
                print('Using transformers for RankGPT')

            self.llm = llm

        self.sliding_window_size = sliding_window_size
        if sliding_window_stride is None:
            self.sliding_window_stride = sliding_window_size//2
        else:
            self.sliding_window_stride = sliding_window_stride

    def _get_openai_ranking(self, text: str, max_tokens):

        messages = [{'role': 'system', 'content': "This is an intelligent assistant that can rank passages based on their relevancy to the query."}]
        messages.append({'role': 'user', 'content': text})

        try:
            chat_completion = self.client.chat.completions.create(messages=messages, model=self.base_llm_name, temperature=0,
                                                                max_tokens=max_tokens, stop=['\n\n'],  response_format={"type": "json_object"})
            response_content = chat_completion.choices[0].message.content

            self.input_tokens += chat_completion.usage.prompt_tokens
            self.output_tokens += chat_completion.usage.completion_tokens
        except:
            # Would be counted as failed re-ranking.
            response_content = ''

        return response_content

    def _create_prompt(self, query, doc_pool):

        documents_prompt = '\n'.join([f'[{i + 1}] ' + doc for i, doc in enumerate(doc_pool)])

        if 'gpt' in self.base_llm_name:
            # prompt For OpenAI models
            prompt = f"""The following are {len(doc_pool)} passages, each indicated by number identifier []. I can rank them based on their relevance to query: "{query}."\n\n{documents_prompt}\n\nThe search query is: "{query}". I will rank the {len(doc_pool)} passages above based on their relevance to the search query. The passages will be listed in descending order using identifiers, the most relevant passages should be listed first and the output format should be a JSON dictionary like {{'ranked_passages': [2, 3, 4, ..., 1]}}. Be sure to list all {len(doc_pool)} ranked passages and do not explain your ranking until after the list is done."""
        else:
            # Prompt for open-weight models
            prompt = f"""{self.prompt_prefix} This is an intelligent assistant that can rank passages based on their relevancy to the query.\n\nThe following are {len(doc_pool)} passages, each indicated by number identifier []. I can rank them based on their relevance to query: "{query}"\n\n{documents_prompt}\n\nThe search query is: "{query}". I will rank the {len(doc_pool)} passages above based on their relevance to the search query. The passages will be listed in descending order using identifiers, the most relevant passages should be listed first and the output format should be [] > [] > etc, e.g., [1] > [2] > etc. Be sure to list all {len(doc_pool)} ranked passages and do not explain your ranking until after the list is done. {self.prompt_suffix} Ranked Passages: ["""
        
        return prompt

    def _get_sorted_docs_from_prompts(self, permutation_prompts, total_docs_to_rank=20):
        '''
        Sort documents based on permutation method as seen in RankGPT
        '''
        if 'gpt' in self.base_llm_name:
            outputs = []

            for permutation_prompt in tqdm(permutation_prompts):
                outputs.append(self._get_openai_ranking(permutation_prompt, int(100 * total_docs_to_rank / 20)))
        elif self.use_vllm:
            # print('Finished loading vllm model')
            sampling_params = SamplingParams(temperature=0, top_p=1.00, max_tokens=100*total_docs_to_rank/20)
            outputs = self.llm.generate(permutation_prompts, sampling_params, use_tqdm=False)
        else:
            outputs = []
            for permutation_prompt in tqdm(permutation_prompts):
                encoded_prompts = self.tokenizer(permutation_prompt, return_tensors='pt').to(self.llm.device)
                input_length = encoded_prompts['input_ids'].size(1)
                outputs.append(self.tokenizer.decode(self.llm.generate(**encoded_prompts, max_new_tokens=100*total_docs_to_rank/20)[0][input_length:], skip_special_tokens=True))
        

        return outputs

    def _rank_docs_from_output(self, output, doc_pool):
        if 'gpt' not in self.base_llm_name:
            
            if self.use_vllm:
                output_str = output.outputs[0].text
            else:
                output_str = output
            output_str = output_str.strip()
            if not output_str.startswith('['):
                output_str = '[' + output_str
            try:
                trim_start = re.search('[^ >\[\]0-9]',
                                       output_str).start()  # trimming generation after standard format
            except:
                trim_start = len(output_str)
            
            trimmed_decoded_output = output_str[:trim_start].strip()
            correct_format = True
            
            try:
                split_output = trimmed_decoded_output.split('>')
                order = []

                for doc_id in split_output:
                    order.append(int(doc_id.strip()[1:-1]) - 1)
                    if len(order) == len(doc_pool):
                        break

                assert len(np.unique(order)) == len(doc_pool), f"Duplicate doc ids in the output: {order}"

                for i in order:
                    assert i > -1, f"Invalid doc id: {i}"
                    assert i < len(doc_pool), f"Invalid doc id: {i}"
            except:
                order = list(range(len(doc_pool)))
                correct_format = False
        else:
            try:
                order = [i-1 for i in eval(output)['ranked_passages']]
                correct_format = True
                assert len(np.unique(order)) == len(doc_pool), f"Duplicate doc ids in the output: {order}"
                for i in order:
                    assert i > -1, f"Invalid doc id: {i}"
                    assert i < len(doc_pool), f"Invalid doc id: {i}"
            except:
                order = list(range(len(doc_pool)))
                correct_format = False
        return order, correct_format
    
    def get_sorted_docs(self, query, doc_pool):
        '''
        Sort documents based on permutation method as seen in RankGPT
        '''

        permutation_prompt = self._create_prompt(query, doc_pool)
        if 'gpt' in self.base_llm_name:
            output = self._get_openai_ranking(permutation_prompt, max(100,int(100 * len(doc_pool) / 20)))
        else:
            sampling_params = SamplingParams(temperature=0, top_p=1.00, max_tokens=100*len(doc_pool)/20)
            output = self.llm.generate([permutation_prompt], sampling_params, use_tqdm=False)[0]

        return self._rank_docs_from_output(output, doc_pool)

    def rerank(self, query, documents, order='desc'):
        '''
        Rerank the documents based on the query using a sliding window strategy.
        Assume that input documents are sorted by their relevance to the query in the descending order.
        '''
        N_docs = len(documents)

        if self.sliding_window_size < 0:
            self.sliding_window_size = N_docs
            
        sorted_doc_ids = list(range(N_docs))
        sorted_doc_ids.reverse()
        
        N_valid_reranks = 0.0
        N_reranks = 0
        _i = 0
        _j = min(self.sliding_window_size, N_docs)

        while True:
            ids = [sorted_doc_ids[i] for i in range(_i, _j)]
            
            # Put relevant docs at the front for RankGPT
            ids.reverse()
            docs = [documents[i] for i in ids]
            
            _sorted_doc_ids, correct_format = self.get_sorted_docs(query, docs)
            
            if correct_format:
                N_valid_reranks += 1
            N_reranks += 1

            _sorted_doc_ids.reverse()
            __sorted_doc_ids = [ids[i] for i in _sorted_doc_ids]
            
            for i in range(_i, _j):
                sorted_doc_ids[i] = __sorted_doc_ids[i-_i]
            if _j == N_docs:
                break
            
            _i += self.sliding_window_stride
            _j += self.sliding_window_stride
            _j = min(_j, N_docs)
            
        if order == 'desc':
            sorted_doc_ids.reverse()

        return sorted_doc_ids, N_valid_reranks/N_reranks