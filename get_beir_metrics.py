import json
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.datasets.data_loader import GenericDataLoader
from pyserini.search import get_qrels
# dataset = 'fiqa'



def get_beir_metrics(llm_name, dataset, split, scoring_strategy, retriever='bm25'):
    print(f'Processing [{llm_name}]...')
    print(f'Processing [{scoring_strategy}]...')
    top_k=20
    # for dataset in ['scidocs']:
    print(f'Processing [{dataset}]...')

    

    retrieval_output_file = './output/retrieval_results/{}_{}_{}_scoring_{}_top_{}_reverse_order.json'.format(
                dataset,
                split,
                llm_name.split('/')[-1],
                scoring_strategy,
                top_k
                )


    retrieval_output_file = './output/retrieval_results/rerank_{}_{}_icr_{}_scoring_{}_top_{}_trunc_300_reverse_order.json'.format(
                retriever,
                dataset,
                llm_name.split('/')[-1],
                scoring_strategy,
                top_k
                )
    retrieval_output_file = 'output/retrieval_results/rerank_bm25_dbpedia-entity_rankgpt_Meta-Llama-3.1-8B-Instruct_top_20_trunc_300_fcr_0.99.json'
    if dataset in ['arguana', 'fiqa']:
        retrieval_output_file=retrieval_output_file.replace('.json', '_no_title.json')

# split='full'

# for dataset in ['trec-covid']:
#     print(f'Processing {dataset}...')
#     retrieval_output_file = './output/retrieval_results/rankgpt_{}_{}_{}_top_{}.json'.format(
#                 dataset,
#                 split,
#                 llm_name.split('/')[-1],
#                 top_k
#                 )


    ks = [10]

    with open(retrieval_output_file, 'r') as f:
        retrieval_results = json.load(f)

    # qrel_file = '/research/nfs_su_809/chen.10216/projects/IC-RAG/data/beir/{}/qrels/test.tsv'.format(dataset)
    # _, _, _qrels = GenericDataLoader(data_folder = '/research/nfs_su_809/chen.10216/projects/IC-RAG/data/beir/{}'.format(dataset)).load(split='test')
    qrel_name = 'beir-v1.0.0-{}-test'.format(dataset)
    print(qrel_name)
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
  

    print(f'{len(retrieval_results.keys())} examples evaluated')

    ndcg, _, recall, precision = evaluator.evaluate(qrels, retrieval_results, ks)

    print('NDCG:\n', json.dumps(ndcg, indent=2))
    print()
    # print('Recall:\n', json.dumps(recall, indent=2))
if __name__ == '__main__':
    llm_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    # llm_name = 'mistralai/Mistral-7B-Instruct-v0.2'
    # llm_name = 'CohereForAI/c4ai-command-r-v01'
    # split doesn't matter here, just make sure the corresponding file exists

    split='dev' 
    # split='full' 
    scoring_strategy = 'masked_NA_calibration'
    # for dataset in ['scifact', 'fiqa', 'arguana', 'fever', 'trec-covid','webis-touche2020','nfcorpus','dbpedia-entity','climate-fever','scidocs']:
    # for dataset in ['trec-covid', 'nfcorpus','dbpedia-entity','scifact', 'signal1m','trec-news', 'robust04']:
    for dataset in ['dbpedia-entity']:
        get_beir_metrics(llm_name, dataset, split, scoring_strategy, retriever='bm25')