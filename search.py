from bsbi import BSBIIndex
from compression import VBEPostings
from letor import LETOR

# sebelumnya sudah dilakukan indexing
# BSBIIndex hanya sebagai abstraksi untuk index tersebut
BSBI_instance = BSBIIndex(data_dir='TP3/collections',
                          postings_encoding=VBEPostings,
                          output_dir='index')

# BSBI_instance = BSBIIndex(data_dir='testing',
#                           postings_encoding=VBEPostings,
#                           output_dir='testing')

queries = ["definisi medis alexia"]

for query in queries:
    print("Query  : ", query)
    print("Results BM25:")
    # for (score, doc) in BSBI_instance.retrieve_tfidf(query, k=10):
    #     print(f"{doc:30} {score:>.3f}")
    for (score, doc) in BSBI_instance.retrieve_bm25(query, k = 100)[:10]:
       print(f"{doc:30} {score:>.3f}")

    ret_scores = BSBI_instance.retrieve_bm25(query, k = 100)

    letor = LETOR('TP3/qrels-folder')

    letor.load_model('ranker_model.txt', 'lsi.model')

    final_pred = letor.make_pred(ret_scores , query)

    print("SERP/Ranking LETOR :")
    for (dname, score) in final_pred[:10]:
        print(dname, score)

    print()
