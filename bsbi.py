import os
import pickle
import contextlib
import heapq
import math
import re

from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, merge_and_sort_posts_and_tfs
from compression import VBEPostings
from tqdm import tqdm

from mpstemmer import MPStemmer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from operator import itemgetter


class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen (misal,
                    /collection/0/gamma.txt) to docIDs
    data_dir(str): Path ke data
    output_dir(str): Path ke output index files
    postings_encoding: Lihat di compression.py, kandidatnya adalah StandardPostings,
                    VBEPostings, dsb.
    index_name(str): Nama dari file yang berisi inverted index
    """

    def __init__(self, data_dir, output_dir, postings_encoding, index_name="main_index"):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding

        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []

    def save(self):
        """Menyimpan doc_id_map and term_id_map ke output directory via pickle"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)

    def load(self):
        """Memuat doc_id_map and term_id_map dari output directory"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)

    def pre_processing_text(self, content):
        """
        Melakukan preprocessing pada text, yakni stemming dan removing stopwords
        """
        # https://github.com/ariaghora/mpstemmer/tree/master/mpstemmer

        stemmer = MPStemmer()
        stemmed = stemmer.stem(content)
        remover = StopWordRemoverFactory().create_stop_word_remover()
        return remover.remove(stemmed)

    def parsing_block(self, block_path):
        """
        Lakukan parsing terhadap text file sehingga menjadi sequence of
        <termID, docID> pairs.

        Gunakan tools available untuk stemming bahasa Indonesia, seperti
        MpStemmer: https://github.com/ariaghora/mpstemmer 
        Jangan gunakan PySastrawi untuk stemming karena kode yang tidak efisien dan lambat.

        JANGAN LUPA BUANG STOPWORDS! Kalian dapat menggunakan PySastrawi 
        untuk menghapus stopword atau menggunakan sumber lain seperti:
        - Satya (https://github.com/datascienceid/stopwords-bahasa-indonesia)
        - Tala (https://github.com/masdevid/ID-Stopwords)

        Untuk "sentence segmentation" dan "tokenization", bisa menggunakan
        regex atau boleh juga menggunakan tools lain yang berbasis machine
        learning.

        Parameters
        ----------
        block_path : str
            Relative Path ke directory yang mengandung text files untuk sebuah block.

            CATAT bahwa satu folder di collection dianggap merepresentasikan satu block.
            Konsep block di soal tugas ini berbeda dengan konsep block yang terkait
            dengan operating systems.

        Returns
        -------
        List[Tuple[Int, Int]]
            Returns all the td_pairs extracted from the block
            Mengembalikan semua pasangan <termID, docID> dari sebuah block (dalam hal
            ini sebuah sub-direktori di dalam folder collection)

        Harus menggunakan self.term_id_map dan self.doc_id_map untuk mendapatkan
        termIDs dan docIDs. Dua variable ini harus 'persist' untuk semua pemanggilan
        parsing_block(...).
        """
        # TODO
        stemmer = MPStemmer()
        stopwords = StopWordRemoverFactory().get_stop_words()

        ret_list = []

        for path in os.walk(self.data_dir + "/" + block_path):
            for files in path[2]:
                with open(self.data_dir + "/" + block_path + "/" + files, 'r', encoding="utf8") as f:
                    terms = re.findall(r'\w+', f.read())  # f.read().split()

                    stemmed_term = [stemmer.stem(term.lower())
                                    if term else ''
                                    for term in terms
                                    ]
                    stemmed_term_no_empty = [
                        term for term in stemmed_term
                        if not ((term == '') or (term == None))
                    ]

                    doc_id = self.doc_id_map[self.data_dir + "/" + block_path + "/" + files]

                    no_stopwords_term = [term for term in stemmed_term_no_empty if term not in stopwords]

                    for term in no_stopwords_term:
                        ret_list.append((self.term_id_map[term], doc_id))

        return ret_list

    def write_to_index(self, td_pairs, index):
        """
        Melakukan inversion td_pairs (list of <termID, docID> pairs) dan
        menyimpan mereka ke index. Disini diterapkan konsep BSBI dimana 
        hanya di-maintain satu dictionary besar untuk keseluruhan block.
        Namun dalam teknik penyimpanannya digunakan strategi dari SPIMI
        yaitu penggunaan struktur data hashtable (dalam Python bisa
        berupa Dictionary)

        ASUMSI: td_pairs CUKUP di memori

        Di Tugas Pemrograman 1, kita hanya menambahkan term dan
        juga list of sorted Doc IDs. Sekarang di Tugas Pemrograman 2,
        kita juga perlu tambahkan list of TF.

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index pada disk (file) yang terkait dengan suatu "block"
        """
        # TODO
        term_dict = {}
        for term_id, doc_id in td_pairs:
            if term_id not in term_dict:
                term_dict[term_id] = dict()
            if doc_id not in term_dict[term_id].keys():
                term_dict[term_id][doc_id] = 0
            term_dict[term_id][doc_id] += 1


        for term_id in sorted(term_dict.keys()):
            term_list = []
            tf_list = []
            for doc in sorted(term_dict[term_id].keys()):
                term_list.append(doc)
                tf_list.append(term_dict[term_id][doc])

            index.append(term_id, term_list, tf_list)



    def merge_index(self, indices, merged_index):
        """
        Lakukan merging ke semua intermediate inverted indices menjadi
        sebuah single index.

        Ini adalah bagian yang melakukan EXTERNAL MERGE SORT

        Gunakan fungsi merge_and_sort_posts_and_tfs(..) di modul util

        Parameters
        ----------
        indices: List[InvertedIndexReader]
            A list of intermediate InvertedIndexReader objects, masing-masing
            merepresentasikan sebuah intermediate inveted index yang iterable
            di sebuah block.

        merged_index: InvertedIndexWriter
            Instance InvertedIndexWriter object yang merupakan hasil merging dari
            semua intermediate InvertedIndexWriter objects.
        """
        # kode berikut mengasumsikan minimal ada 1 term
        merged_iter = heapq.merge(*indices, key=lambda x: x[0])
        curr, postings, tf_list = next(merged_iter)  # first item
        for t, postings_, tf_list_ in merged_iter:  # from the second item
            if t == curr:
                zip_p_tf = merge_and_sort_posts_and_tfs(list(zip(postings, tf_list)),
                                                        list(zip(postings_, tf_list_)))
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(curr, postings, tf_list)

    def retrieve_tfidf(self, query, k=10):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = (1 + log tf(t, D))       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

        catatan: 
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_li
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        # TODO
        self.load()

        stemmer = MPStemmer()
        stopwords = StopWordRemoverFactory().get_stop_words()

        terms = re.findall(r'\w+', query)

        stem_query = [stemmer.stem(term.lower()) if term else '' for term in terms]
        stem_no_empty = [self.term_id_map[term] for term in stem_query if not ((term == '') or (term == None))]

        no_stopwords_query = stem_no_empty

        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as index:
            scores = {}
            index.get_avg_doc_length()

            for query_index in no_stopwords_query:
                if query_index in index.postings_dict.keys():
                    df_query = index.postings_dict[query_index][1]
                else:
                    continue
                N_len = len(index.doc_length)#index.total_doc_length
                w_query = math.log((N_len / df_query), 10)
                post_list = index.get_postings_list(query_index)

                for doc_id, doc_tf in post_list:
                    w_doc = (1 + math.log(doc_tf, 10))
                    if doc_id not in scores.keys():
                        scores[doc_id] = 0
                    scores[doc_id] += w_doc * w_query

            ret_list = []

            for i in scores.keys():
                # print(self.doc_id_map[i], scores[i], index.doc_length[i])
                ret_list.append((scores[i], self.doc_id_map[i]))

            return sorted(ret_list, reverse=True)[:k]


    def retrieve_bm25(self, query, k=10, k1=1.2, b=0.75):
        """
        Melakukan Ranked Retrieval dengan skema scoring BM25 dan framework TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        """
        # TODO
        self.load()

        stemmer = MPStemmer()
        stopwords = StopWordRemoverFactory().get_stop_words()

        terms = re.findall(r'\w+', query)

        stem_query = [stemmer.stem(term.lower()) if term else '' for term in terms]
        stem_no_empty = [self.term_id_map[term] for term in stem_query if not ((term == '') or (term == None))]

        no_stopwords_query = stem_no_empty

        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as index:
            scores = {}
            index.get_avg_doc_length()

            for query_index in no_stopwords_query:
                if query_index in index.postings_dict.keys():
                    df_query = index.postings_dict[query_index][1]
                else:
                    continue
                N_len = len(index.doc_length)
                w_query = math.log((N_len / df_query), 10)
                post_list = index.get_postings_list(query_index)

                for doc_id, doc_tf in post_list:

                    if doc_id not in scores.keys():
                        scores[doc_id] = 0
                    scores[doc_id] += w_query * (((k1 + 1) * doc_tf) / ((k1 * ((1 - b) + (b * (index.doc_length[doc_id] / index.avg_doc_length)))) + doc_tf))

            ret_list = []

            for i in scores.keys():
                ret_list.append((scores[i], self.doc_id_map[i]))

            return sorted(ret_list, reverse=True)[:k]

    def do_indexing(self):
        """
        Base indexing code
        BAGIAN UTAMA untuk melakukan Indexing dengan skema BSBI (blocked-sort
        based indexing)

        Method ini scan terhadap semua data di collection, memanggil parsing_block
        untuk parsing dokumen dan memanggil write_to_index yang melakukan inversion
        di setiap block dan menyimpannya ke index yang baru.
        """
        # loop untuk setiap sub-directory di dalam folder collection (setiap block)
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            td_pairs = self.parsing_block(block_dir_relative)
            index_id = 'intermediate_index_'+block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, directory=self.output_dir) as index:
                self.write_to_index(td_pairs, index)
                td_pairs = None

        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
                           for index_id in self.intermediate_indices]
                self.merge_index(indices, merged_index)


if __name__ == "__main__":

    BSBI_instance = BSBIIndex(data_dir='TP3/collections',
                              postings_encoding=VBEPostings,
                              output_dir='index')
    # BSBI_instance = BSBIIndex(data_dir='testing',
    #                           postings_encoding=VBEPostings,
    #                           output_dir='testing')
    BSBI_instance.do_indexing()  # memulai indexing!
