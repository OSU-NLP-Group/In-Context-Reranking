import math

import transformers
from transformers.cache_utils import DynamicCache
from transformers.models.mistral.modeling_mistral import repeat_kv
import torch
import gc
import random

from .custom.custom_cache import DynamicCacheWithQuery
from .custom.custom_modeling_mistral import MistralForCausalLM
from .custom.custom_modeling_llama import LlamaForCausalLM

class InContextReranker():

    def __init__(self, 
                 base_llm_name,
                 prompt_template='instruct',
                 prompt_prefix='',
                 prompt_suffix='',
                 scoring_strategy='query_last',
                 use_fa2=True,
                 retrieval_type='QA',
                 sliding_window_size=20,
                 sliding_window_stride=None,
                 reverse_doc_order=False,
                 ) -> None:
        '''
        Inputs:
            base_llm: The base LLM model to be used for document retrieval.
            tokenizer: The tokenizer for the base LLM model.
            prompt_template: The template for the prompt to be used for document retrieval. 
                Options: 
                    'instruct': default instruction template
                    'simple': no instruction used
                    'simple_instruct: only wrap the input with corresponding chat templates of the base model. e.g. [INST]...[/INST] for Mistral-instruct
        '''

        # Setup the base LLM
        self._base_llm_name = base_llm_name
        tokenizer = transformers.AutoTokenizer.from_pretrained(base_llm_name)
        self.tokenizer = tokenizer
        print(f"initialized tokenizer for [{base_llm_name}]")
        
        if any([x in base_llm_name.lower() for x in ['mistralai/mistral', ]]):
            BaseLLMClass = MistralForCausalLM
        elif any([x in base_llm_name.lower() for x in ['llama']]):
            BaseLLMClass = LlamaForCausalLM
        else:
            print(f"Warning: The model family for [{base_llm_name}] is not supported by InContextRAGModel!")
            raise NotImplementedError

        prompt_template, prompt_prefix, prompt_suffix, = self._setup_llm_prompts(prompt_template, base_llm_name)
        
        if use_fa2:
            _attn_implementation = "flash_attention_2"
        else:
            _attn_implementation = "eager"

        llm = BaseLLMClass.from_pretrained(
                base_llm_name, 
                torch_dtype=torch.float16, 
                attn_implementation=_attn_implementation,
                device_map='auto'
            )
        self.llm = llm
        self.llm.config.pad_token_id = self.llm.config.eos_token_id
        
        # Setup prompts for ICR
        assert prompt_template in ['instruct', 'simple', 'simple_instruct'], "Invalid prompt template!"
        
        self.prompt_template = prompt_template
        self.prompt_prefix = prompt_prefix

        
        self.prompt_suffix = prompt_suffix
        self.scoring_strategy = scoring_strategy
        
        if retrieval_type == 'QA':
            print('[ICR is using QA prompt type]')
            self.retrieval_instruction = ' Here are some paragraphs:'
            self.retrieval_instruction_late = 'Please answer the following question based on the information in the paragraphs above.'
        elif retrieval_type == 'IE':
            print('[ICR is using IE prompt type]')
            self.retrieval_instruction = ' Here are some paragraphs:'
            self.retrieval_instruction_late = 'Please find information that are relevant to the following query in the paragraphs above.'
        else:
            raise NotImplementedError('Invalid retrieval type! Should be one of [QA, IE]')

        assert scoring_strategy in ['query_last', 'attention_sorting', 'NA_only', 'NA_calibration_no_agg', 'masked_NA_calibration'], "Invalid scoring strategy!"
        
        self._use_fa2 = use_fa2
        if use_fa2:
            print('Using FA2 for retrieval score computation.')
        else:
            print('Using eager attention weights for retrieval score computation.')
        self.num_layers = self.llm.config.num_hidden_layers
        

        self.start_layer = 0
        self.end_layer = self.num_layers - 1
        
        print('[ICR is using layers from {} to {}.]'.format(self.start_layer, self.end_layer))


        # The following settings are for constructing the input prompt.
        self.prompt_bos_length=1
        if any(x in self._base_llm_name.lower() for x in ['mistral-']):
            self.additional_prompt_offset = 1 # for models that adds a ' ' at the beginning when tokenizing the prompt. e.g. '\n\n' -> [<s>, ' ', '\n\n']
            self.prompt_separator = '\n\n'
        elif any([x in self._base_llm_name.lower() for x in ['llama']]):
            self.additional_prompt_offset = 0
            self.prompt_separator = ' \n\n'
        else:
            self.additional_prompt_offset = 0
            self.prompt_separator = '\n\n'

        
        # Setup sliding window.
        # ICR typically works worse with sliding window, especially with smaller window sizes. Try to fit all documents to be re-ranked in the context as much as possible. 
        self.reverse_doc_order = reverse_doc_order
        self.sliding_window_size = sliding_window_size
        if sliding_window_stride is None:
            self.sliding_window_stride = sliding_window_size//2
        else:
            self.sliding_window_stride = sliding_window_stride
        
    def _setup_llm_prompts(self, prompt_template, base_llm_name):
        
        if prompt_template == '':
            prompt_template='instruct' if any(x in base_llm_name.lower() for x in ['instruct']) else 'simple'
        else:
            assert prompt_template in ['instruct', 'simple', 'simple_instruct']
        print('ICR is using prompt template [{}] for in-context retrieval'.format(prompt_template))
        
        if  'mistral' in base_llm_name.lower():
            prompt_prefix = '[INST]'
            prompt_suffix = '[/INST]'
        elif 'llama-3' in base_llm_name.lower():
            prompt_prefix = '<|start_header_id|>user<|end_header_id|>'
            prompt_suffix = '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
        else:
            raise NotImplementedError("Prompt prefix and suffix not defined for the model family of {}.".format(base_llm_name))
        
        return prompt_template, prompt_prefix, prompt_suffix

            
    def rerank(self, query, documents, return_per_doc_results=False, order='desc', calib_query_type='NA'):
        '''
        Rerank the documents based on the query using a sliding window strategy.
        Assume that input documents are sorted by their relevance to the query in the descending order.
        '''
        # reverse the order of input documents to perform sliding window from the rear to the front of the list
        # documents.reverse()
        N_docs = len(documents)

        if self.sliding_window_size < 0:
            self.sliding_window_size = N_docs
            
        sorted_doc_ids = list(range(N_docs))
        sorted_doc_ids.reverse()
        
        sorted_doc_scores = []
        if return_per_doc_results == 'tok':
            per_doc_results = []
        else:
            per_doc_results = None
        
        _i = 0
        _j = min(self.sliding_window_size, N_docs)
        while True:
            
            ids = [sorted_doc_ids[i] for i in range(_i, _j)]
            if not self.reverse_doc_order:
                # Put the most relevant documents at the front of document list.
                ids.reverse()

            docs = [documents[i] for i in ids]
            (_sorted_doc_ids, _sorted_doc_scores), _per_doc_results = self.get_sorted_docs(query, docs, return_per_doc_results=return_per_doc_results, order='asc')

            __sorted_doc_ids = [ids[i] for i in _sorted_doc_ids]
            for i in range(_i, _j):
                sorted_doc_ids[i] = __sorted_doc_ids[i-_i]

            if _j < N_docs:
                sorted_doc_scores.extend(_sorted_doc_scores[:self.sliding_window_stride])
                if return_per_doc_results == 'tok':
                    per_doc_results.extend(_per_doc_results[:self.sliding_window_stride])
            else:
                sorted_doc_scores.extend(_sorted_doc_scores)
                if return_per_doc_results == 'tok':
                    per_doc_results.extend(_per_doc_results)
                break

            _i += self.sliding_window_stride
            _j += self.sliding_window_stride
            _j = min(_j, N_docs)
            
        
        if order == 'desc':
            sorted_doc_ids.reverse()
            sorted_doc_scores.reverse()
            if return_per_doc_results == 'tok':
                per_doc_results.reverse()

        assert len(sorted_doc_ids) == len(sorted_doc_scores), "Length mismatch between sorted doc ids ({}) and scores({})!".format(len(sorted_doc_ids), len(sorted_doc_scores))
        return (sorted_doc_ids, sorted_doc_scores), per_doc_results

    def score_documents(
            self,
            llm_input,
            doc_tok_idx_spans,
            query_start_tok_idx,
            query_end_tok_idx,
            context_start_idx=0,
            return_per_doc_results=False,
            long_prompt=False,
            return_cache=False,
            kv_cache=None,
        ):

        tokenized_input = self.tokenizer(llm_input,return_tensors='pt').to(self.llm.device)
        _input_ids = tokenized_input.input_ids[:, context_start_idx:]
        _query_indices = list(range(query_start_tok_idx-context_start_idx, query_end_tok_idx-context_start_idx+1))
        
        if kv_cache is None:
            if self._use_fa2:
                kv_cache=DynamicCacheWithQuery(query_indices=_query_indices)
            else:
                kv_cache=DynamicCache()
        else:
            kv_cache.query_cache = []
            _query_indices = _query_indices
            kv_cache._query_indices = _query_indices

        with torch.no_grad():
            output = self.llm(
                input_ids=_input_ids,
                use_cache=True,
                past_key_values=kv_cache,
                output_attentions=True
                )

        if self._use_fa2:
            # Extract key and query vectors from FA2. Then recompute attention scores for re-ranking.
            kv_cache = output.past_key_values

            long_prompt = False
            if len(_input_ids[0]) > 40000:
                # For sequences that are too long, compute scores on CPU to void GPU OOM.
                # Adjust the limit here depending on your system configuration.
                print('Long sequence of more than 40K tokens detected. Computing attention scores on CPU.')
                long_prompt = True
            
            attention_weights = []
            doc_tok_weights = []
            
            if long_prompt:
                _device = 'cpu'
            else:
                _device = 'cuda:0'
            
            # loop through all layers and compute attention scores
            for i in range(self.start_layer, self.end_layer+1):                     
                attn_weights = self._get_attn_weights(kv_cache.key_cache[i][:,:,:query_end_tok_idx+1], kv_cache.query_cache[i],  use_cpu=long_prompt).to(_device).squeeze(0)
                attn_weights = attn_weights.mean(1) # average over query tokens
                attention_weights.append(attn_weights.squeeze(0))
                
        else:
            # Directly extract attention weights from the attention layers of the LLM.
            attention_weights = [attn[0][:,query_start_tok_idx:query_end_tok_idx+1,:].mean(1) for attn in output.attentions]

        attention_weights = torch.stack(attention_weights, dim=0)
        
        if return_per_doc_results != 'none':
            per_doc_results = [[None, None] for _ in range(len(doc_tok_idx_spans))]
        else:
            per_doc_results = None
    
        attention_weights = attention_weights.sum(0) # sum attention scores across layers            
        attention_weights = attention_weights.sum(0) # sum attention scores across attention heads
        doc_scores = []
        
        for i, doc_span in enumerate(doc_tok_idx_spans): 
            _tok_score = attention_weights[doc_span[0]:doc_span[1]]
            doc_scores.append(_tok_score.sum())

            if return_per_doc_results != 'none':
                _doc_tok_ids = tokenized_input.input_ids[0][doc_span[0]:doc_span[1]]
                _doc_toks = self.tokenizer.convert_ids_to_tokens(_doc_tok_ids)
                per_doc_results[i][0] = _doc_toks
                per_doc_results[i][1] = _tok_score.clone().detach() # sum over layers
            
        doc_scores = torch.tensor(doc_scores)
        gc.collect()
        torch.cuda.empty_cache()

        if return_cache:
            return doc_scores, per_doc_results, kv_cache
        else:
            return doc_scores, per_doc_results


    def get_sorted_docs(self, query, retrieval_doc_pool, return_per_doc_results=False, prompt_prefix='', order='desc'):


        kv_cache = None
        
        if self.scoring_strategy == 'query_last':
            # ICR without calibration.
            llm_prompt, doc_tok_idx_spans, query_start_idx, query_end_idx = self._prepare_input_for_document_retrieval(query, retrieval_doc_pool, system_prompt=prompt_prefix, query_position='last')
            doc_scores, perdoc_result,_ = self.score_documents(llm_prompt, doc_tok_idx_spans, query_start_idx, query_end_idx, return_per_doc_results=return_per_doc_results)
        
        elif self.scoring_strategy == 'attention_sorting':
            # ICR without both calibration and attention aggregation.
            llm_prompt, doc_tok_idx_spans, query_start_idx, query_end_idx = self._prepare_input_for_document_retrieval(query, retrieval_doc_pool, system_prompt=prompt_prefix, query_position='last')
            query_start_idx = query_end_idx # Only using last query token (i.e. attention sorting).
            doc_scores, perdoc_result,_ = self.score_documents(llm_prompt, doc_tok_idx_spans, query_start_idx, query_end_idx, return_per_doc_results=return_per_doc_results)
        
        elif self.scoring_strategy == 'NA_only':
            # For analyzing the intrinsic bias captured by calibration scores.
            query = 'N/A'

            llm_prompt, doc_tok_idx_spans, query_start_idx, query_end_idx = self._prepare_input_for_document_retrieval(query, retrieval_doc_pool, system_prompt=prompt_prefix, query_position='last')
            doc_scores, perdoc_result,_ = self.score_documents(llm_prompt, doc_tok_idx_spans, query_start_idx, query_end_idx, return_per_doc_results=return_per_doc_results)
        
        elif self.scoring_strategy == 'NA_calibration_no_agg':
            # ICR without attention aggregation.
            
            llm_prompt, doc_tok_idx_spans, query_start_idx, query_end_idx = self._prepare_input_for_document_retrieval(query, retrieval_doc_pool, system_prompt=prompt_prefix, query_position='last')
            query_start_idx = query_end_idx
            doc_scores_query, perdoc_result, _, kv_cache = self.score_documents(llm_prompt, doc_tok_idx_spans, query_start_idx, query_end_idx, return_per_doc_results=return_per_doc_results, return_cache=True)
            
            
            calibration_query = 'N/A'
            llm_prompt, doc_tok_idx_spans, query_start_idx, query_end_idx = self._prepare_input_for_document_retrieval(calibration_query, retrieval_doc_pool, system_prompt=prompt_prefix, query_position='last')

            # Use kv_cache from first query to speed up forward() for the calibration query.
            # query_start_idx should be the same for both queries.
            for i in range(len(kv_cache.key_cache)):
                kv_cache.key_cache[i] = kv_cache.key_cache[i][:,:,:query_start_idx,:]
                kv_cache.value_cache[i] = kv_cache.value_cache[i][:,:,:query_start_idx,:]
            kv_cache._seen_tokens = query_start_idx
            
            
            if kv_cache is not None:
                context_start_idx=query_start_idx
            else:
                context_start_idx=0

            query_start_idx = query_end_idx
            doc_scores_calib, doc_tok_scores_calib_na,_ = self.score_documents(llm_prompt, doc_tok_idx_spans, query_start_idx, query_end_idx, return_per_doc_results=return_per_doc_results,  kv_cache=kv_cache, context_start_idx=context_start_idx)

            doc_scores = doc_scores_query - doc_scores_calib
            
            if return_per_doc_results != 'none':
                for i in range(len(perdoc_result)):
                    perdoc_result[i][1] -= doc_tok_scores_calib_na[i][1]
        
        elif self.scoring_strategy == 'masked_NA_calibration':
            return_per_doc_results = 'tok'
            # The default ICR method
            
            # FP with calibration query
            calibration_query = 'N/A'
            llm_prompt, doc_tok_idx_spans, query_start_idx, query_end_idx = self._prepare_input_for_document_retrieval(calibration_query, retrieval_doc_pool, system_prompt=prompt_prefix, query_position='last')

            doc_scores_calib, doc_tok_scores_calib_na, kv_cache = self.score_documents(llm_prompt, doc_tok_idx_spans, query_start_idx, query_end_idx, return_per_doc_results=return_per_doc_results,  return_cache=True)
            
            # Use kv_cache from first query to speed up forward() for the calibration query.
            # query_start_idx should be the same for both queries.
            for i in range(len(kv_cache.key_cache)):
                kv_cache.key_cache[i] = kv_cache.key_cache[i][:,:,:query_start_idx,:]
                kv_cache.value_cache[i] = kv_cache.value_cache[i][:,:,:query_start_idx,:]
            kv_cache._seen_tokens = query_start_idx
            
            if kv_cache is not None:
                context_start_idx=query_start_idx
            else:
                context_start_idx=0

            # FP with the actual query            
            llm_prompt, doc_tok_idx_spans, query_start_idx, query_end_idx = self._prepare_input_for_document_retrieval(query, retrieval_doc_pool, system_prompt=prompt_prefix, query_position='last')
        
            doc_scores_query, perdoc_result = self.score_documents(llm_prompt, doc_tok_idx_spans, query_start_idx, query_end_idx, return_per_doc_results=return_per_doc_results,  kv_cache=kv_cache, context_start_idx=context_start_idx)

            
            _i = 0
            doc_scores = torch.zeros(len(retrieval_doc_pool))

            for doc_tok_score, doc_tok_score_na in zip(perdoc_result, doc_tok_scores_calib_na):
                doc_tok_score[1] = doc_tok_score[1].to(doc_tok_score_na[1].device)
                calibrated_scores = doc_tok_score[1] - doc_tok_score_na[1]
                
                mean_bias = calibrated_scores.mean()
                std_bias = calibrated_scores.std()
                threshold = mean_bias - 2*std_bias
                tok_mask = (calibrated_scores>threshold)
                
                doc_tok_score[1] = doc_tok_score[1] * tok_mask
                doc_tok_score_na[1] = doc_tok_score_na[1] * tok_mask
                doc_tok_score[1] = doc_tok_score[1] - doc_tok_score_na[1]
                doc_scores[_i] = doc_tok_score[1].sum()
                _i+=1

        per_doc_result = None
        if order in ['desc', 'asc']:
            sorted_results = torch.sort(doc_scores, descending=(order=='desc'))
            if return_per_doc_results != 'none':
                per_doc_result = [(perdoc_result[i][0], perdoc_result[i][1]) for i in sorted_results.indices]
            
            return (sorted_results.indices.tolist(), sorted_results.values.tolist()), per_doc_result
        elif order=='none':
            # Only return the scores and the per-doc results for documents in the input order.
            # Used during development.
            return list(range(len(retrieval_doc_pool))), doc_scores, per_doc_result
        else:
            print(f"Invalid order: {order}. Please use 'desc', 'asc' or 'none")
            raise NotImplementedError

    def _prepare_input_for_document_retrieval(self, query, documents, system_prompt='', query_position='last'):
        '''
        Only tested with Mistral and Llama-3.1. Models using other tokenizers may need to modify this function.
        '''
        llm_prompt = ''
        document_span_intervals = []
        

        if self.prompt_template == 'simple':
            system_prompt = ''
        elif self.prompt_template == 'simple_instruct':
            system_prompt = system_prompt
        elif self.prompt_template == 'instruct':
            if system_prompt != '':
                system_prompt = self.retrieval_instruction.format(len(documents), query) + self.prompt_separator + system_prompt
            else:
                system_prompt = self.retrieval_instruction.format(len(documents), query)
        
        system_prompt = self.prompt_prefix + system_prompt

        query_start_idx = None
        query_end_idx = None
        
        
        separator_length = self.tokenizer(self.prompt_separator, return_tensors='pt').input_ids.size(1) - self.prompt_bos_length - self.additional_prompt_offset # remove the leading ['<s>', '_'] tokens
        
        llm_prompt = system_prompt
        
        
        prompt_length = self.tokenizer(llm_prompt+self.prompt_separator, return_tensors='pt').input_ids.size(1)-separator_length # add and subtract separator tokens for accurate prefix length
        
        if query_position == 'first':
            if self.prompt_template in ['simple', 'instruct']:
                instruction_prompt = f'Query:'

                llm_prompt += self.prompt_separator + instruction_prompt 
                prompt_length += separator_length
                prompt_length += self.tokenizer(self.prompt_separator + instruction_prompt, return_tensors='pt').input_ids.size(1) - self.prompt_bos_length - separator_length - self.additional_prompt_offset
                query_start_idx = prompt_length - 1 # The ':' after 'Query'    
            else:
                llm_prompt += self.prompt_separator
                prompt_length += separator_length
                query_start_idx = prompt_length # The start of the query context
            
            if self.prompt_template == 'simple':
                query_prompt = f' {query.strip()}{self.prompt_separator}Answer:'
            elif self.prompt_template in ['instruct', 'simple_instruct']:
                query_prompt = f' {query.strip()}'
            
            llm_prompt += query_prompt
            prompt_length += self.tokenizer(query_prompt, return_tensors='pt').input_ids.size(1) - self.prompt_bos_length - self.additional_prompt_offset
            query_end_idx = prompt_length - 1 
            
        
        if prompt_length != self.tokenizer(llm_prompt, return_tensors='pt').input_ids.size(1):
                print('Prompt length mismatch!')
                print(prompt_length, ' vs ', self.tokenizer(llm_prompt, return_tensors='pt').input_ids.size(1))
                print('-'*30)
                self.__show_tokens(llm_prompt)
                raise Exception('ICR prompt length mismatch before adding docs.')

        _doc_separator_length = separator_length

        for i, doc in enumerate(documents):
            
            doc = f'[{i+1}] {doc}'
            prompt_length += _doc_separator_length
            llm_prompt += self.prompt_separator + doc
            doc_length = self.tokenizer(self.prompt_separator + doc, return_tensors='pt').input_ids.size(1) - self.prompt_bos_length - _doc_separator_length - self.additional_prompt_offset # - bos_length for the leading ['<s>'] token, -additional for the potential extra tokens, e.g. the '_' token between <s> and <0x0A> when <0x0A> is the first token for mistral models.
            
            document_span_intervals.append((prompt_length, prompt_length + doc_length))
            prompt_length += doc_length

            if prompt_length != self.tokenizer(llm_prompt, return_tensors='pt').input_ids.size(1):
                print('Prompt length mismatch @ doc {}!'.format(i))
                print(prompt_length, ' vs ', self.tokenizer(llm_prompt, return_tensors='pt').input_ids.size(1))
                print('-'*30)
                self.__show_tokens(llm_prompt)
                print('-'*30)
                print('doc length:', doc_length)
                self.__show_tokens(self.prompt_separator+doc)
                raise Exception('ICR prompt length mismatch after adding docs.')


        if query_position == 'last':
            query_start_idx = prompt_length + separator_length
            if self.prompt_template in ['simple', 'instruct']:
                instruction_prompt = self.retrieval_instruction_late + self.prompt_separator + 'Query:'
                llm_prompt += self.prompt_separator + instruction_prompt 
                prompt_length += separator_length
                prompt_length += self.tokenizer(self.prompt_separator + instruction_prompt, return_tensors='pt').input_ids.size(1) - self.prompt_bos_length - separator_length - self.additional_prompt_offset
                
            else:
                llm_prompt += self.prompt_separator
                prompt_length += separator_length

        if self.prompt_template == 'simple':
            query_prompt = f' {query.strip()}{self.prompt_separator}Answer:'
        elif self.prompt_template in ['instruct', 'simple_instruct']:
            query_prompt = f' {query.strip()}'
            if query_position == 'last':
                query_prompt += self.prompt_suffix.format(len(documents))

        llm_prompt += query_prompt
        prompt_length += self.tokenizer(query_prompt, return_tensors='pt').input_ids.size(1) - self.prompt_bos_length - self.additional_prompt_offset
        if query_position == 'last':
            query_end_idx = prompt_length - 1
        return llm_prompt, document_span_intervals, query_start_idx, query_end_idx
    
    @classmethod
    def __show_tokens(self, string):
        # Shows tokenized string.
        # Mainly used for debugging prompt construction for document retrieval.
        tokenized_string_ids = self.tokenizer(string, return_tensors='pt').input_ids[0]
        print(self.tokenizer.convert_ids_to_tokens(tokenized_string_ids), tokenized_string_ids.size(0))
                      
    @classmethod
    def _get_attn_weights(cls, key_states, query_states, use_cpu=False):

        bsz, num_heads, q_len, head_dim = query_states.size()
        num_key_value_heads = key_states.size(1)
        num_key_value_groups = num_heads // num_key_value_heads
        kv_seq_len = key_states.size(-2)

        if use_cpu:
            query_states = query_states.cpu()
            key_states = key_states.cpu()

        key_states = repeat_kv(key_states, num_key_value_groups)


        attn_weights = torch.matmul(query_states, key_states.transpose(2,3)) / math.sqrt(head_dim)

        if attn_weights.size() != (bsz, num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        
        # Make causal mask and add it to attention weights.
        causal_mask = cls._get_causal_mask(attn_weights).to(attn_weights.device)
        attn_weights += causal_mask.unsqueeze(0)
        attn_lses = torch.logsumexp(attn_weights, dim=-1, keepdim=True) # Log-sum-exp of attention weights for numerical stability in softmax.
        attn_weights = torch.exp(attn_weights - attn_lses) # softmax
        
        return attn_weights

    @classmethod
    def _get_causal_mask(cls, attn_weights):
        # Make causal mask for attention weights.
        query_len, seq_len = attn_weights.size(-2), attn_weights.size(-1)
        causal_mask = torch.ones_like(attn_weights.transpose(-1,-2).squeeze(0))
        causal_mask = torch.triu(causal_mask, diagonal=-(seq_len-query_len))
        causal_mask = causal_mask.transpose(-1,-2)
        causal_mask = (1-causal_mask) * torch.finfo(causal_mask.dtype).min
        return causal_mask