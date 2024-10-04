import json
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
import numpy as np
import random
from src.in_context_reranker import InContextReranker
import argparse
from pyserini.search import get_qrels

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--save_per_doc_results', type=str, default='none', choices=['none', 'tok', 'att_head'],)
parser.add_argument('--llm_name', type=str, required=True)
parser.add_argument('--scoring_strategy', type=str, default='masked_NA_calibration', choices=['query_last', 'attention_sorting', 'NA_only', 'NA_calibration_no_agg', 'masked_NA_calibration'])
parser.add_argument('--debug', type=int, default=0)
parser.add_argument('--seed', type=int, default=-1,)
parser.add_argument('--oracle', action='store_true')
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--top_k', type=int, default=20)
parser.add_argument('--split', type=str, default='dev')
parser.add_argument('--use_eager_attn', action='store_true')
parser.add_argument('--retrieval_type', type=str, default='IE', choices=['QA', 'IE'])
parser.add_argument('--save_retrieval_results', action='store_true')
parser.add_argument('--no_rerank', action='store_true')
parser.add_argument('--beir_eval', action='store_true')
parser.add_argument('--shuffle_documents', action='store_true')
parser.add_argument('--reverse_doc_order', action='store_true')
parser.add_argument('--calib_query_type', type=str, default='NA', choices=['NA'])
parser.add_argument('--retriever', type=str, default='colbertv2', choices=['bm25', 'colbertv2'])
parser.add_argument('--reranker', type=str, choices=['icr', 'rankgpt'])
parser.add_argument('--rerank_sliding_window_size', type=int, default=-1)
parser.add_argument('--rerank_sliding_window_stride', type=int, default=10)
parser.add_argument('--disable_vllm', action='store_true')
parser.add_argument('--truncate_by_space', type=int, default=-1) 
parser.add_argument('--actual_topk', type=int, default=-1)
args = parser.parse_args()

if args.beir_eval or args.data not in ['musique', 'hotpotqa', '2wikimultihopqa']:
    from beir.retrieval.evaluation import EvaluateRetrieval
    args.beir_eval = True
else:
    args.beir_eval = False

    
if args.reranker == 'rankgpt':
    from src.rank_gpt_reranker import RankGPTModel
def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

NO_TITLE_DATASETS = ['fiqa']

if __name__ == '__main__':

    if args.seed != -1:
        print('using random seed: ', args.seed)
        random.seed(args.seed)
    

    if args.data in ['musique', 'hotpotqa', '2wikimultihopqa']:
        query_set = json.load(open(f'./retriever_output/icr_multihop_{args.data}_colbertv2_top_{args.top_k}.json','r'))
           
        id_key = 'id'
        beir_exp = False
        if args.reranker == 'icr':
            args.retrieval_type='QA'
    else:
        query_set = json.load(open(f'./retriever_outpout/icr_beir_{args.data}_{args.retriever}_top_{args.top_k}.json'))
        id_key = 'idx'
        beir_exp = True

        if args.reranker == 'icr' and args.data in ['trec-covid', 'fiqa', 'webis-touche2020', 'dbpedia-entity', 'nq']:
            args.retrieval_type = 'QA'
        else:
            args.retrieval_type = 'IE'
    
    if args.debug:
        k = args.debug
        print('Debug mode, only processing {} queries out of {} ones.'.format(k, len(query_set)))
        query_set = query_set[:k]

    ks = [1,2,3,4,5,10]
    recalls = []

    llm_name = args.llm_name
    print('-'*50)

    if args.reverse_doc_order:
        print('Reversing the order of paragraphs for each query. i.e. most relevant paragraph is at the end.')

    if args.reranker == 'rankgpt':
        if args.save_per_doc_results != 'none':
            print('RankGPT does not support saving per-doc results. Setting save_per_doc_results to none.')
            args.save_per_doc_results = 'none'
    
    if not args.no_rerank:
        print('Doing re-ranking on the [{}] dataset with base retriever [{}]'.format(args.data, args.retriever))
        print('Doing re-ranking using {} + {}'.format(args.reranker, llm_name))
        if args.truncate_by_space > 0:
            # This option is added to follow RankGPT's setting
            print('Truncating each paragraph to {} words.'.format(args.truncate_by_space))

    if args.reranker == 'icr':
        print('Using ICR with scoring strategy: {}'.format(args.scoring_strategy))
    
    if args.actual_topk > 0:
        print('Using actual topk: ', args.actual_topk)
        assert not args.save_retrieval_results, 'Cannot save retrieval results when using actual topk.'
    
    # Set output file name
    reranker_str = f'{args.reranker}_{args.llm_name.split("/")[-1]}'
    if args.reranker == 'icr':
        reranker_str += f'_scoring_{args.scoring_strategy}'
    if args.rerank_sliding_window_size == -1:
        reranker_str += '_no_sw'
    if args.save_per_doc_results != 'none':
        assert args.reranker == 'icr', 'Only ICR supports saving per-doc results.'

        if beir_exp:
            per_doc_output_file = './output/per_doc_results/rerank_{}_{}_{}_{}_top_{}.json'.format(
                args.retriever,
                args.data,
                reranker_str,
                args.save_per_doc_results,
                
                args.top_k
                )
        else:
            per_doc_output_file = './output/per_doc_results/rerank_{}_{}_{}_{}_top_{}.json'.format(
                args.retriever,
                args.data,
                reranker_str,
                args.save_per_doc_results,
                args.top_k
                )
        if args.truncate_by_space > 0:
            per_doc_output_file=per_doc_output_file.replace('.json', '_trunc_{}.json'.format(args.truncate_by_space))

        if args.debug:
            per_doc_output_file=per_doc_output_file.replace('.json', '_debug.json')
            per_doc_output_file=per_doc_output_file.replace('.json', '_calib_type_{}.json'.format(args.calib_query_type))
            
        if args.reverse_doc_order:
            per_doc_output_file=per_doc_output_file.replace('.json', '_reverse_order.json')
        
        print('Saving per-doc results to {}.'.format(per_doc_output_file))
        all_per_doc_results = []
        
    if args.save_retrieval_results:
        if beir_exp:
            retrieval_output_file = './output/retrieval_results/rerank_{}_{}_{}_top_{}.json'.format(
                args.retriever,
                args.data,
                reranker_str,
                args.top_k
                )
        else:
            retrieval_output_file = './output/retrieval_results/rerank_{}_{}_{}_top_{}.json'.format(
                args.retriever,
                args.data,
                reranker_str,
                args.top_k
                )
        if args.truncate_by_space > 0:
            retrieval_output_file=retrieval_output_file.replace('.json', '_trunc_{}.json'.format(args.truncate_by_space))
        if args.reverse_doc_order:
            retrieval_output_file=retrieval_output_file.replace('.json', '_reverse_order.json')
        print('Saving retrieval results to {}.'.format(retrieval_output_file))
    retrieval_results = {} # stored in BEIR's format

    # Initialize the reranker model
    if not args.no_rerank:
        if args.reranker == 'icr':
            tokenizer = AutoTokenizer.from_pretrained(llm_name)
            reranker = InContextReranker(
                llm_name,
                scoring_strategy=args.scoring_strategy,
                use_fa2=not args.use_eager_attn,
                retrieval_type=args.retrieval_type,
                reverse_doc_order=args.reverse_doc_order,
                sliding_window_size=args.rerank_sliding_window_size,
                sliding_window_stride=args.rerank_sliding_window_stride
                )
        elif args.reranker == 'rankgpt':
            reranker = RankGPTModel(
                llm_name,
                use_vllm=not(args.disable_vllm),
                sliding_window_size=args.rerank_sliding_window_size,
                sliding_window_stride=args.rerank_sliding_window_stride)
    else:
        print('Directly reporting {} results'.format(args.retriever))
        

        
    
    
    if args.data in NO_TITLE_DATASETS:
        print('Not adding title to paragraphs for the [{}] dataset.'.format(args.data))

    if args.reranker == 'rankgpt':
        format_correct_rates = []

    for i, query in enumerate(tqdm(query_set)):
        
        question = query['question']
        if beir_exp:
            paragraphs = [p for p in query['paragraphs'] if p['idx'] != query[id_key]] # remove same doc from pool for some datasets.
        else:
            paragraphs = query['paragraphs']
        if args.actual_topk > 0:
            paragraphs = paragraphs[:args.actual_topk]

        if args.truncate_by_space > 0:
            # Truncate each paragraph by space.
            # We follow the implementation of RankGPT and truncate the documents to 300 words for BEIR experiments.
            for p in paragraphs:
                p['paragraph_text'] = ' '.join(p['paragraph_text'].split(' ')[:args.truncate_by_space])
        
        if args.shuffle_documents:
            random.shuffle(paragraphs)
        
        
        total_gold_doc_num = min(args.top_k, query['num_gold_docs'])
        total_supporting_items = len([x for x in paragraphs if x['is_supporting']])
        
        if args.data in NO_TITLE_DATASETS:
            passages = [(p['paragraph_text']).strip() for p in paragraphs] 
            gold_docs = set([(p['paragraph_text']).strip() for p in paragraphs if p['is_supporting']])
        else:
            passages = [(p['title'] + '\n' + p['paragraph_text']).strip() for p in paragraphs] 

            gold_docs = set([(p['title'] + '\n' + p['paragraph_text']).strip() for p in paragraphs if p['is_supporting']])

        


        gold_ids = [_i for _i, p in enumerate(paragraphs) if p['is_supporting']]

        if args.debug:
            print('question: ', question)
            print('gold docs: ', gold_ids)

        if not args.no_rerank:
            if args.reranker == 'rankgpt':
                sorted_doc_ids, format_correct_rate = reranker.rerank(question, passages)
                format_correct_rates.append(format_correct_rate)
                sorted_doc_scores = np.array(list(range(len(passages), 0, -1))) / len(passages)
            elif args.reranker == 'icr':
                (sorted_doc_ids, sorted_doc_scores), per_doc_results = reranker.rerank(question, passages, return_per_doc_results=args.save_per_doc_results, calib_query_type=args.calib_query_type)
            else:
                print('Unknown reranker type!')
            try:
                if args.reranker == 'rankgpt':
                    sorted_doc_ids, format_correct_rate = reranker.rerank(question, passages)
                    format_correct_rates.append(format_correct_rate)
                    sorted_doc_scores = np.array(list(range(len(passages), 0, -1))) / len(passages)
                elif args.reranker == 'icr':
                    (sorted_doc_ids, sorted_doc_scores), per_doc_results = reranker.rerank(question, passages, return_per_doc_results=args.save_per_doc_results, calib_query_type=args.calib_query_type)
                else:
                    print('Unknown reranker type!')

            except Exception as e:
                print(e)
                print('Error in retrieval for example No. {}, fall back to ColBERTv2 Results...'.format(i))
            
                sorted_doc_ids = list(range(len(passages)))                    
                sorted_doc_scores = np.array(list(range(len(passages), 0, -1)))/len(passages)
                per_doc_results = None
        else:
            # Report retriever performance
            sorted_doc_ids = list(range(len(passages)))
            sorted_doc_scores = np.array(list(range(len(passages), 0, -1)))/len(passages)

            per_doc_results = None
        if args.debug:
            print('sorted doc ids: ', sorted_doc_ids)
        
        if beir_exp:
            _id = query[id_key]
        else:
            _id = i
        retrieval_results[_id] = {}
        for _i, sorted_idx in enumerate(sorted_doc_ids):
            if beir_exp:
                retrieval_results[_id][str(paragraphs[sorted_idx]['idx'])] = sorted_doc_scores[_i]
            else:
                retrieval_results[_id][sorted_idx] = sorted_doc_scores[_i]

        recalls_at = []

        if args.save_per_doc_results != 'none':
            _per_doc_result = {
                'query': question,
                'docs':[]
            }
            for _i, _id in enumerate(sorted_doc_ids):

                _doc_result = {
                    'input_rank': _id,
                    'is_gold': _id in gold_ids,
                    'retrieval_score': np.round(sorted_doc_scores[_i], 5).tolist(),
                    'toks': per_doc_results[_i][0],
                    'scores': per_doc_results[_i][1].tolist()
                }
                _per_doc_result['docs'].append(_doc_result)
            all_per_doc_results.append(_per_doc_result)

        for k in ks:
            retrieved_docs = np.array(passages)[sorted_doc_ids[:k]]
            retrieved_docs = set(retrieved_docs)
            true_positives = gold_docs.intersection(retrieved_docs)
            n_tp = len(true_positives) # regular evaluation
            
            if args.oracle:
                n_tp = min(total_supporting_items, k) # oracle setting for performance upper bound

            if total_gold_doc_num == 0:
                recalls_at.append(0)
            else:
                recalls_at.append(n_tp / total_gold_doc_num)

        recalls.append(recalls_at)

    if not args.beir_eval:
        print(pd.DataFrame(recalls, columns=ks).agg(['mean']).T)
    
    if args.reranker == 'rankgpt':
        print('RankGPT Format Correct Rate: ', np.mean(format_correct_rates))


    if args.save_per_doc_results != 'none':
        json.dump(all_per_doc_results, open(per_doc_output_file, 'w'), indent=2)
        print(f'Saved results to {per_doc_output_file}')
    if args.save_retrieval_results:
        if args.data in NO_TITLE_DATASETS:
            retrieval_output_file = retrieval_output_file.replace('.json', '_no_title.json')
        if args.debug:
            retrieval_output_file = retrieval_output_file.replace('.json', '_debug.json')
        if args.no_rerank:
            retrieval_output_file = retrieval_output_file.replace('.json', '_{}.json'.format(args.retriever))
        if args.reranker == 'rankgpt':
            retrieval_output_file = retrieval_output_file.replace('.json', '_fcr_{}.json'.format(np.mean(format_correct_rates)))
        
        json.dump(retrieval_results, open(retrieval_output_file, 'w'), indent=2)
        print(f'Saved retrieval results to {retrieval_output_file}')
        
    if args.beir_eval:
        print('---- BEIR Evaluation ----')
        qrel_name = 'beir-v1.0.0-{}-test'.format(args.data)
        _qrels = get_qrels(qrel_name)
        evaluator = EvaluateRetrieval()
        qrels = {}
        for qid in retrieval_results:
            assert isinstance(qid, str)
            try:
                __qrels = _qrels[qid]
            except:
                try:
                    __qrels = _qrels[int(qid)]
                except:
                    print('Error in qrels for query id: ', qid)
                    continue
            
            # make sure the qrels are in the right format
            qrels[qid] = {}
            for doc_id in __qrels:
                qrels[qid][str(doc_id)] = __qrels[doc_id]
               
            doc_keys = list(qrels[qid].keys())
            for key in doc_keys:
                if not isinstance(qrels[qid][key], int):
                    qrels[qid][key] = int(qrels[qid][key]) # make sure the relevance is integer
                if qrels[qid][key] == 0:
                    qrels[qid].pop(key)
            
        ndcg, _, recall, precision = evaluator.evaluate(qrels, retrieval_results, ks)
        print('NDCG:\n', json.dumps(ndcg, indent=2))

