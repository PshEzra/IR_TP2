import random
import lightgbm as lgb
import numpy as np

from gensim.models import TfidfModel
from gensim.models import LsiModel
from gensim.corpora import Dictionary
from scipy.spatial.distance import cosine

class LETOR:
    def __init__(self, main_dir):
        self.main_dir = main_dir
        self.doc = {}
        self.queries = {}
        self.q_docs_rel = {}
        self.NUM_LATENT_TOPICS = 200 #originally 200
        self.NUM_NEGATIVES = 1
        self.dictionary = Dictionary()
        self.dataset = []
        self.group_qid_count = []
        self.lsi_model = None
        self.ranker = None

    def open_train(self):
        with open(self.main_dir+"/train_docs.txt", encoding="utf8") as file:
            for line in file:
                content = line.strip().split(" ")
                self.doc[content[0]] = content[1:]

        with open(self.main_dir+"/train_queries.txt", encoding="utf8") as file:
            for line in file:
                content = line.strip().split(" ")
                self.queries[content[0]] = content[1:]

        with open(self.main_dir+"/train_qrels.txt", encoding="utf8") as file:
            for line in file:
                q_id, doc_id, rel, _ = line.split(" ")
                if (q_id in self.queries) and (doc_id in self.doc):
                    if q_id not in self.q_docs_rel:
                        self.q_docs_rel[q_id] = []
                    self.q_docs_rel[q_id].append((doc_id, int(rel)))

        for q_id in self.q_docs_rel:
            docs_rels = self.q_docs_rel[q_id]
            self.group_qid_count.append(len(docs_rels) + self.NUM_NEGATIVES)
            for doc_id, rel in docs_rels:
                self.dataset.append((self.queries[q_id], self.doc[doc_id], rel))
            # tambahkan satu negative (random sampling saja dari documents)
            self.dataset.append((self.queries[q_id], random.choice(list(self.doc.values())), 0))

    def build_lsi(self):
        bow_corpus = [self.dictionary.doc2bow(d, allow_update=True) for d in self.doc.values()]
        self.lsi_model = LsiModel(bow_corpus, num_topics=self.NUM_LATENT_TOPICS)

    def get_features(self):
        self.build_lsi()

        X = []
        Y = []
        for (query, doc, rel) in self.dataset:
            X.append(self.features(query, doc))
            Y.append(rel)

        # ubah X dan Y ke format numpy array
        X = np.array(X)
        Y = np.array(Y)

        return X, Y

    def vector_rep(self, text):
        rep = [topic_value for (_, topic_value) in self.lsi_model[self.dictionary.doc2bow(text)]]
        return rep if len(rep) == self.NUM_LATENT_TOPICS else [0.] * self.NUM_LATENT_TOPICS

    def features(self, query, doc):
        v_q = self.vector_rep(query)
        v_d = self.vector_rep(doc)
        q = set(query)
        d = set(doc)
        cosine_dist = cosine(v_q, v_d)
        jaccard = len(q & d) / len(q | d)
        return v_q + v_d + [cosine_dist] + [jaccard]

    def train_model(self):
        self.open_train()

        print("Building LSI")

        X, Y = self.get_features()

        print("Creating Ranker")

        ranker = lgb.LGBMRanker(
            objective="lambdarank",
            boosting_type="gbdt",
            n_estimators=100, #originally 100
            importance_type="gain",
            metric="ndcg",
            num_leaves=40, #originally 40
            learning_rate=0.02,
            max_depth=-1)

        print("Training...")

        ranker.fit(X, Y,
                   group=self.group_qid_count,
                   )

        self.ranker = ranker

    def load_model(self, ranker_saved, lsi_saved):
        self.ranker = lgb.Booster(model_file=ranker_saved)
        self.lsi_model = LsiModel.load(lsi_saved)

    def save_model(self, ranker_dir, lsi_dir):
        self.ranker.booster_.save_model(ranker_dir)
        self.lsi_model.save(lsi_dir)

    def make_pred(self, docs, query):

        pred_docs = []
        for score, doc in docs:
            with open(doc, 'r', encoding='utf8') as file:
                pred_docs.append(self.features(query.split(), file.read().strip().split()))

        pred_docs = np.array(pred_docs)

        scores = self.ranker.predict(pred_docs)

        final_scores = []

        for i in range(len(scores)):
            final_scores.append(scores[i] + docs[i][0])

        did_scores = [x for x in zip([dname for (_, dname) in docs], final_scores)]
        sorted_did_scores = sorted(did_scores, key=lambda tup: tup[1], reverse=True)

        return sorted_did_scores

if __name__ == "__main__":
    # 1. Persiapkan data yang akan dilakukan re - ranking
    letor = LETOR('TP3/qrels-folder')

    rankr = letor.train_model()

    #rankr.booster_.save_model('ranker_model.txt', 'lsi.model')

    letor.save_model('ranker_model.txt', 'lsi.model')
    '''
    query = "how much cancer risk can be avoided through lifestyle change ?"

    docs = [("D1",
             "dietary restriction reduces insulin-like growth factor levels modulates apoptosis cell proliferation tumor progression num defici pubmed ncbi abstract diet contributes one-third cancer deaths western world factors diet influence cancer elucidated reduction caloric intake dramatically slows cancer progression rodents major contribution dietary effects cancer insulin-like growth factor igf-i lowered dietary restriction dr humans rats igf-i modulates cell proliferation apoptosis tumorigenesis mechanisms protective effects dr depend reduction multifaceted growth factor test hypothesis igf-i restored dr ascertain lowering igf-i central slowing bladder cancer progression dr heterozygous num deficient mice received bladder carcinogen p-cresidine induce preneoplasia confirmation bladder urothelial preneoplasia mice divided groups ad libitum num dr num dr igf-i igf-i/dr serum igf-i lowered num dr completely restored igf-i/dr-treated mice recombinant igf-i administered osmotic minipumps tumor progression decreased dr restoration igf-i serum levels dr-treated mice increased stage cancers igf-i modulated tumor progression independent body weight rates apoptosis preneoplastic lesions num times higher dr-treated mice compared igf/dr ad libitum-treated mice administration igf-i dr-treated mice stimulated cell proliferation num fold hyperplastic foci conclusion dr lowered igf-i levels favoring apoptosis cell proliferation ultimately slowing tumor progression mechanistic study demonstrating igf-i supplementation abrogates protective effect dr neoplastic progression"),
            ("D2", "study hard as your blood boils"),
            ("D3",
             "processed meats risk childhood leukemia california usa pubmed ncbi abstract relation intake food items thought precursors inhibitors n-nitroso compounds noc risk leukemia investigated case-control study children birth age num years los angeles county california united states cases ascertained population-based tumor registry num num controls drawn friends random-digit dialing interviews obtained num cases num controls food items principal interest breakfast meats bacon sausage ham luncheon meats salami pastrami lunch meat corned beef bologna hot dogs oranges orange juice grapefruit grapefruit juice asked intake apples apple juice regular charcoal broiled meats milk coffee coke cola drinks usual consumption frequencies determined parents child risks adjusted risk factors persistent significant associations children's intake hot dogs odds ratio num num percent confidence interval ci num num num hot dogs month trend num fathers intake hot dogs num ci num num highest intake category trend num evidence fruit intake provided protection results compatible experimental animal literature hypothesis human noc intake leukemia risk potential biases data study hypothesis focused comprehensive epidemiologic studies warranted"),
            ("D4",
             "long-term effects calorie protein restriction serum igf num igfbp num concentration humans summary reduced function mutations insulin/igf-i signaling pathway increase maximal lifespan health span species calorie restriction cr decreases serum igf num concentration num protects cancer slows aging rodents long-term effects cr adequate nutrition circulating igf num levels humans unknown report data long-term cr studies num num years showing severe cr malnutrition change igf num igf num igfbp num ratio levels humans contrast total free igf num concentrations significantly lower moderately protein-restricted individuals reducing protein intake average num kg num body weight day num kg num body weight day num weeks volunteers practicing cr resulted reduction serum igf num num ng ml num num ng ml num findings demonstrate unlike rodents long-term severe cr reduce serum igf num concentration igf num igfbp num ratio humans addition data provide evidence protein intake key determinant circulating igf num levels humans suggest reduced protein intake important component anticancer anti-aging dietary interventions"),
            ("D5",
             "cancer preventable disease requires major lifestyle abstract year num million americans num million people worldwide expected diagnosed cancer disease commonly believed preventable num num cancer cases attributed genetic defects remaining num num roots environment lifestyle lifestyle factors include cigarette smoking diet fried foods red meat alcohol sun exposure environmental pollutants infections stress obesity physical inactivity evidence cancer-related deaths num num due tobacco num num linked diet num num due infections remaining percentage due factors radiation stress physical activity environmental pollutants cancer prevention requires smoking cessation increased ingestion fruits vegetables moderate alcohol caloric restriction exercise avoidance direct exposure sunlight minimal meat consumption grains vaccinations regular check-ups review present evidence inflammation link agents/factors cancer agents prevent addition provide evidence cancer preventable disease requires major lifestyle")]

    # sekedar pembanding, ada bocoran: D3 & D5 relevant, D1 & D4 partially relevant, D2 tidak relevan

    # bentuk ke format numpy array
    X_unseen = []
    for doc_id, doc in docs:
        X_unseen.append(letor.features(query.split(), doc.split()))

    X_unseen = np.array(X_unseen)

    scores = rankr.predict(X_unseen)

    did_scores = [x for x in zip([did for (did, _) in docs], scores)]
    sorted_did_scores = sorted(did_scores, key=lambda tup: tup[1], reverse=True)

    print("query        :", query)
    print("SERP/Ranking :")
    for (did, score) in sorted_did_scores:
        print(did, score)
    # 2. Membuat LSI / LSA Model
    # 3. Train LightGBM LambdaMART Model
    # 4. Melakukan Prediksi
    '''
