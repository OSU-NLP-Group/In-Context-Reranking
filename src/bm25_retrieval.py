import json
from pyserini.search import LuceneSearcher, get_topics, get_qrels



def bm25_retrieve_beir(task, K):
    print('Retrieving top-{} using BM25 for task:{}'.format(K, task))
    topics = get_topics('beir-v1.0.0-{}-test'.format(task))
    qrels = get_qrels('beir-v1.0.0-{}-test'.format(task))
    searcher = LuceneSearcher.from_prebuilt_index('beir-v1.0.0-{}.flat'.format(task))

    ICR_data = []
    for topic_idx in list(topics.keys()):
        topic = topics[topic_idx]
        query = topic['title']
        hits = searcher.search(query, k=K)
        hit_items = 0
        _sample = {
            "idx": str(topic_idx),
            "question": query,
            "paragraphs":[],
        }
        for hit in hits:
            doc_id = hit.docid
            _doc_json = json.loads(searcher.doc(doc_id).raw())
            _is_support = False
            qrel_key_type = type(list(qrels[topic_idx].keys())[0])
            if qrel_key_type == int:
                _doc_id = int(doc_id)
            else:
                _doc_id = doc_id

            if _doc_id in qrels[topic_idx].keys():
                if int(qrels[topic_idx][_doc_id]) > 0:
                    _is_support = True

            if _is_support:
                hit_items += 1
            _sample['paragraphs'].append({
            'idx': _doc_json['_id'],
            'title': _doc_json['title'],
            'paragraph_text': _doc_json['text'],
            'is_supporting': _is_support,
        })
            
        ICR_data.append(_sample)
        _sample['num_gold_docs'] = hit_items
    output_file_name = '../retriever_outpout/icr_beir_{}_bm25_top_{}.json'.format(task, K)
    with open(output_file_name, 'w') as f:
        json.dump(ICR_data, f, indent=2)
    print('Saved retrieval results to ', output_file_name)

# for task in ['trec-covid','nfcorpus','dbpedia-entity','scifact','scidocs','fiqa','fever','climate-fever', 'nq']:
for task in ['dbpedia-entity']:
    bm25_retrieve_beir(task,100)