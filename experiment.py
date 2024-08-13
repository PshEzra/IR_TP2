import math
import re
import os
from bsbi import BSBIIndex
from compression import VBEPostings
from tqdm import tqdm
from collections import defaultdict
from letor import LETOR

# >>>>> 3 IR metrics: RBP p = 0.8, DCG, dan AP


def rbp(ranking, p=0.8):
    """ menghitung search effectiveness metric score dengan 
        Rank Biased Precision (RBP)

        Parameters
        ----------
        ranking: List[int]
           vektor biner seperti [1, 0, 1, 1, 1, 0]
           gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
           Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                   di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                   di rank-6 tidak relevan

        Returns
        -------
        Float
          score RBP
    """
    score = 0.
    for i in range(1, len(ranking) + 1):
        pos = i - 1
        score += ranking[pos] * (p ** (i - 1))
    return (1 - p) * score


def dcg(ranking):
    """ menghitung search effectiveness metric score dengan 
        Discounted Cumulative Gain

        Parameters
        ----------
        ranking: List[int]
           vektor biner seperti [1, 0, 1, 1, 1, 0]
           gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
           Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                   di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                   di rank-6 tidak relevan

        Returns
        -------
        Float
          score DCG
    """
    # TODO
    score = 0
    for i in range(1, len(ranking)+1):
        score += ranking[i-1] / (math.log2(i+1))
    return score


def prec(ranking, k):
    """ menghitung search effectiveness metric score dengan 
        Precision at K

        Parameters
        ----------
        ranking: List[int]
           vektor biner seperti [1, 0, 1, 1, 1, 0]
           gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
           Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                   di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                   di rank-6 tidak relevan

        k: int
          banyak dokumen yang dipertimbangkan atau diperoleh

        Returns
        -------
        Float
          score Prec@K
    """
    # TODO
    score = 0
    for i in range(k):
        score += ranking[i] / k
    return score


def ap(ranking):
    """ menghitung search effectiveness metric score dengan 
        Average Precision

        Parameters
        ----------
        ranking: List[int]
           vektor biner seperti [1, 0, 1, 1, 1, 0]
           gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
           Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                   di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                   di rank-6 tidak relevan

        Returns
        -------
        Float
          score AP
    """
    # TODO
    relevan = 0
    for i in ranking:
        relevan += i
    if relevan == 0: return 0
    score = 0
    for i in range(len(ranking)):
        score += (prec(ranking, i+1) * ranking[i]) / relevan

    return score

# >>>>> memuat qrels


#TP3/qrels-folder/test_qrels.txt
def load_qrels(qrel_file = "TP3/qrels-folder/test_qrels.txt"):
  qrels = defaultdict(lambda: defaultdict(lambda: 0))
  with open(qrel_file) as file:
    for line in file:
      parts = line.strip().split()
      qid = parts[0]
      did = int(parts[1])
      qrels[qid][did] = 1
  return qrels


# >>>>> EVALUASI !

#TP3/qrels-folder/test_queries.txt
def eval_retrieval(qrels, query_file="TP3/qrels-folder/test_queries.txt", k=100):
    """ 
      loop ke semua query, hitung score di setiap query,
      lalu hitung MEAN SCORE-nya.
      untuk setiap query, kembalikan top-1000 documents
    """
    BSBI_instance = BSBIIndex(data_dir='TP3/collections',
                              postings_encoding=VBEPostings,
                              output_dir='index')

    with open(query_file) as file:
        rbp_scores_tfidf = []
        dcg_scores_tfidf = []
        ap_scores_tfidf = []

        rbp_scores_bm25 = []
        dcg_scores_bm25 = []
        ap_scores_bm25 = []

        rbp_scores_letor = []
        dcg_scores_letor = []
        ap_scores_letor = []

        rbp_scores_letor_tf = []
        dcg_scores_letor_tf = []
        ap_scores_letor_tf = []

        for qline in tqdm(file):
            parts = qline.strip().split()
            qid = parts[0]
            query = " ".join(parts[1:])

            """
            Evaluasi TF-IDF
            """
            ranking_tfidf = []
            tfidf_rank = BSBI_instance.retrieve_tfidf(query, k=k)
            for (score, doc) in tfidf_rank:
                did = int(os.path.splitext(os.path.basename(doc))[0])
                # Alternatif lain:
                # 1. did = int(doc.split("\\")[-1].split(".")[0])
                # 2. did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
                # 3. disesuaikan dengan path Anda
                if (did in qrels[qid]):
                    ranking_tfidf.append(1)
                else:
                    ranking_tfidf.append(0)
            rbp_scores_tfidf.append(rbp(ranking_tfidf))
            dcg_scores_tfidf.append(dcg(ranking_tfidf))
            ap_scores_tfidf.append(ap(ranking_tfidf))

            """
            Evaluasi BM25
            """
            ranking_bm25 = []

            bm25_rank = BSBI_instance.retrieve_bm25(query, k=k)
            # nilai k1 dan b dapat diganti-ganti
            for (score, doc) in bm25_rank: # , k1=1, b=0.5
                did = int(os.path.splitext(os.path.basename(doc))[0])
                # Alternatif lain:
                # 1. did = int(doc.split("\\")[-1].split(".")[0])
                # 2. did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
                # 3. disesuaikan dengan path Anda
                if (did in qrels[qid]):
                    ranking_bm25.append(1)
                else:
                    ranking_bm25.append(0)
            rbp_scores_bm25.append(rbp(ranking_bm25))
            dcg_scores_bm25.append(dcg(ranking_bm25))
            ap_scores_bm25.append(ap(ranking_bm25))

            # ===TP3===

            letor = LETOR('TP3/qrels-folder')

            letor.load_model('ranker_model.txt', 'lsi.model')

            final_pred = letor.make_pred(bm25_rank, query)

            ranking_letor = []
            for (doc, score) in final_pred: # , k1=1, b=0.5
                did = int(os.path.splitext(os.path.basename(doc))[0])
                # Alternatif lain:
                # 1. did = int(doc.split("\\")[-1].split(".")[0])
                # 2. did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
                # 3. disesuaikan dengan path Anda
                if (did in qrels[qid]):
                    ranking_letor.append(1)
                else:
                    ranking_letor.append(0)
            rbp_scores_letor.append(rbp(ranking_letor))
            dcg_scores_letor.append(dcg(ranking_letor))
            ap_scores_letor.append(ap(ranking_letor))

            '''
            final_pred_tf = letor.make_pred(tfidf_rank, query)

            ranking_letor_tf = []
            for (doc, score) in final_pred_tf:  # , k1=1, b=0.5
                did = int(os.path.splitext(os.path.basename(doc))[0])
                # Alternatif lain:
                # 1. did = int(doc.split("\\")[-1].split(".")[0])
                # 2. did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
                # 3. disesuaikan dengan path Anda
                if (did in qrels[qid]):
                    ranking_letor_tf.append(1)
                else:
                    ranking_letor_tf.append(0)
            rbp_scores_letor_tf.append(rbp(ranking_letor_tf))
            dcg_scores_letor_tf.append(dcg(ranking_letor_tf))
            ap_scores_letor_tf.append(ap(ranking_letor_tf))'''

    print("Hasil evaluasi TF-IDF terhadap 150 queries")
    print("RBP score =", sum(rbp_scores_tfidf) / len(rbp_scores_tfidf))
    print("DCG score =", sum(dcg_scores_tfidf) / len(dcg_scores_tfidf))
    print("AP score  =", sum(ap_scores_tfidf) / len(ap_scores_tfidf))

    print("Hasil evaluasi BM25 terhadap 150 queries")
    print("RBP score =", sum(rbp_scores_bm25) / len(rbp_scores_bm25))
    print("DCG score =", sum(dcg_scores_bm25) / len(dcg_scores_bm25))
    print("AP score  =", sum(ap_scores_bm25) / len(ap_scores_bm25))

    '''
    print("Hasil evaluasi LETOR reranking terhadap tf-idf")
    print("RBP score =", sum(rbp_scores_letor_tf) / len(rbp_scores_letor_tf))
    print("DCG score =", sum(dcg_scores_letor_tf) / len(dcg_scores_letor_tf))
    print("AP score  =", sum(ap_scores_letor_tf) / len(ap_scores_letor_tf))
    '''
    print("Hasil evaluasi LETOR reranking terhadap BM25")
    print("RBP score =", sum(rbp_scores_letor) / len(rbp_scores_letor))
    print("DCG score =", sum(dcg_scores_letor) / len(dcg_scores_letor))
    print("AP score  =", sum(ap_scores_letor) / len(ap_scores_letor))


if __name__ == '__main__':
    qrels = load_qrels()

    #assert qrels["Q1002252"][5599474] == 1, "qrels salah"
    #assert not (6998091 in qrels["Q1007972"]), "qrels salah"

    eval_retrieval(qrels)
