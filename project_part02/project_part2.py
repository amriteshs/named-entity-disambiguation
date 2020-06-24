import numpy as np
import xgboost as xgb
import pickle
import spacy
import math
from collections import defaultdict
from itertools import combinations, permutations
from datetime import datetime


class InvertedIndex:
    def __init__(self):
        ## You should use these variable to store the term frequencies for tokens and entities...
        self.tf_tokens = defaultdict(lambda: defaultdict(int))
        self.tf_entities = defaultdict(lambda: defaultdict(int))

        ## You should use these variable to store the inverse document frequencies for tokens and entities...
        self.idf_tokens = defaultdict(float)
        self.idf_entities = defaultdict(float)

        ## other variables
        self.tf_norm_tokens = defaultdict(lambda: defaultdict(float))
        self.tf_norm_entities = defaultdict(lambda: defaultdict(float))

        self.tf_idf_tokens = defaultdict(lambda: defaultdict(float))
        self.tf_idf_entities = defaultdict(lambda: defaultdict(float))

        self.nlp = spacy.load('en_core_web_sm')

    ## Your implementation for indexing the documents...
    def index_documents(self, documents):
        for key, val in documents.items():
            doc = self.nlp(val)
            temp = defaultdict(lambda: defaultdict(list))

            for e in doc.ents:
                if key not in self.tf_entities[e.text]:
                    self.tf_entities[e.text][key] = 1
                    temp[e.text][key] = [e.start_char]
                else:
                    self.tf_entities[e.text][key] += 1
                    temp[e.text][key].append(e.start_char)

            ctr = 0

            for t in doc:
                if t.is_punct or t.is_stop:
                    ctr += len(t.text_with_ws)
                    continue

                if t.text in temp:
                    if key in temp[t.text]:
                        if ctr in temp[t.text][key]:
                            ctr += len(t.text_with_ws)
                            continue

                ctr += len(t.text_with_ws)

                if key not in self.tf_tokens[t.text]:
                    self.tf_tokens[t.text][key] = 1
                else:
                    self.tf_tokens[t.text][key] += 1

        for key1, val1 in self.tf_tokens.items():
            self.idf_tokens[key1] = 1.0 + math.log(len(documents) / (1.0 + len(self.tf_tokens[key1])))

            for key2, val2 in val1.items():
                self.tf_norm_tokens[key1][key2] = 1.0 + math.log(1.0 + math.log(val2))
                self.tf_idf_tokens[key1][key2] = self.tf_norm_tokens[key1][key2] * self.idf_tokens[key1]

        for key1, val1 in self.tf_entities.items():
            self.idf_entities[key1] = 1.0 + math.log(len(documents) / (1.0 + len(self.tf_entities[key1])))

            for key2, val2 in val1.items():
                self.tf_norm_entities[key1][key2] = 1.0 + math.log(val2)
                self.tf_idf_entities[key1][key2] = self.tf_norm_entities[key1][key2] * self.idf_entities[key1]

    ## Your implementation to split the query to tokens and entities...
    def split_query(self, Q, DoE):
        entities = []

        for key, val in DoE.items():
            x = Q.split()
            y = key.split()
            i = 0
            j = 0

            while i < len(x) and j < len(y):
                if x[i] == y[j]:
                    j += 1

                i += 1

            if j == len(y):
                entities.append(key)

        qs = defaultdict(lambda: defaultdict(list))
        ctr = 1

        for i in range(len(entities) + 1):
            for j in combinations(entities, i):
                x = [e.split() for e in j]
                y = Q.split()

                if sum([len(e) for e in x]) <= len(y):
                    flag1 = False

                    for k in [list(e) for e in permutations(x)]:
                        flag2 = False
                        temp = list(y)

                        for l in k:
                            idx = 0
                            flag3 = False

                            for m in l:
                                if m in y:
                                    n = [e for e in range(len(y)) if y[e] == m and e >= idx]

                                    if n:
                                        idx = n[0]
                                        del y[idx]
                                    else:
                                        flag3 = True
                                        break
                                else:
                                    flag3 = True
                                    break

                            if flag3:
                                flag2 = True
                                break

                        if not flag2:
                            flag1 = True
                            break

                        y = list(temp)

                    if flag1:
                        qs[ctr]['tokens'] = y
                        qs[ctr]['entities'] = list(j)

                        ctr += 1

        return qs

    ## Your implementation to return the max score among all the query splits...
    def max_score_query(self, query_splits, doc_id):
        score_list = []

        for key, val in query_splits.items():
            tokens_score = 0.0
            entities_score = 0.0

            for i in val['tokens']:
                if i in self.tf_idf_tokens:
                    if doc_id in self.tf_idf_tokens[i]:
                        tokens_score += self.tf_idf_tokens[i][doc_id]

            for i in val['entities']:
                if i in self.tf_idf_entities:
                    if doc_id in self.tf_idf_entities[i]:
                        entities_score += self.tf_idf_entities[i][doc_id]

            combined_score = entities_score + 0.4 * tokens_score

            qs = defaultdict(list)
            qs['tokens'] = val['tokens']
            qs['entities'] = val['entities']

            score_list.append((combined_score, qs))

        return sorted(score_list, key=lambda k: k[0], reverse=True)[0]


def generate_features(mention, entity, document, men_docs, parsed_entity_pages, offset, index, avgdl):
    feature_list = []
    doc_length = len(document.split())

    # TF-IDF of candidate entities in mention documents
    tf_idf_e = index.tf_idf_entities[' '.join([e for e in entity.split('_')])][document]
    feature_list.append(tf_idf_e)

    # IDF of candidate entities in mention documents
    idf_e = index.idf_entities[' '.join([e for e in entity.split('_')])]
    feature_list.append(idf_e)

    # TF of candidate entities in mention documents
    tf_e = index.tf_norm_entities[' '.join([e for e in entity.split('_')])][document]
    feature_list.append(tf_e)

    # TF of mention tokens in parsed entity pages
    tf_m = sum([1 for p in parsed_entity_pages[entity] if p[1].lower() in [i.lower() for i in mention.split()]]) if entity in parsed_entity_pages else 0
    feature_list.append(tf_m)

    # Okapi BM25
    bm25 = sum([(2.5 * index.tf_idf_tokens[e][document]) / (index.tf_norm_tokens[e][document] + 1.5 * (0.25 + (0.75 * doc_length / avgdl))) for e in entity.split('_')])
    feature_list.append(bm25)

    # Language model
    lm = np.prod([(0.5 * index.tf_norm_tokens[e][document] / doc_length) + 0.5 for e in entity.split('_')])
    feature_list.append(lm)

    # Cosine similarity
    ent = [index.tf_idf_tokens[e][document] for e in entity.split('_')]
    men = [index.tf_idf_tokens[m][document] for m in mention.split()]
    if len(ent) > len(men):
        for i in range(len(ent) - len(men)):
            men.append(0.0)
    else:
        for i in range(len(men) - len(ent)):
            ent.append(0.0)
    dnm = (math.sqrt(np.dot(ent, ent)) * math.sqrt(np.dot(men, men)))
    cs = np.dot(ent, men) / dnm if dnm else np.inf
    feature_list.append(cs)

    return feature_list


def disambiguate_mentions(train_mentions, train_labels, dev_mentions, men_docs, parsed_entity_pages):
    index = InvertedIndex()
    index.index_documents(men_docs)

    avgdl = sum([len(v.split()) for k, v in men_docs.items()]) / len(men_docs)

    dtrain_data = []
    dtrain_labels = []
    dtrain_groups = []

    for key, val in train_mentions.items():
        dtrain_groups.append(len(val['candidate_entities']))

        for v in val['candidate_entities']:
            if v == train_labels[key]['label']:
                dtrain_labels.append(1)
            else:
                dtrain_labels.append(0)

            dtrain_data.append(generate_features(val['mention'], v, val['doc_title'], men_docs, parsed_entity_pages, val['offset'], index, avgdl))

    dtrain_data = np.asarray(dtrain_data)
    dtrain_labels = np.asarray(dtrain_labels)
    dtrain_groups = np.asarray(dtrain_groups)

    xgb_train = xgb.DMatrix(data=dtrain_data, label=dtrain_labels)
    xgb_train.set_group(dtrain_groups)

    dtest_data = []
    dtest_groups = []

    for key, val in dev_mentions.items():
        dtest_groups.append(len(val['candidate_entities']))

        for v in val['candidate_entities']:
            dtest_data.append(generate_features(val['mention'], v, val['doc_title'], men_docs, parsed_entity_pages, val['offset'], index, avgdl))

    dtest_data = np.asarray(dtest_data)
    dtest_groups = np.asarray(dtest_groups)

    xgb_test = xgb.DMatrix(data=dtest_data)
    xgb_test.set_group(dtest_groups)

    params = {
        'max_depth': 8,
        'n_estimators': 5000,
        'eta': 0.05,
        'silent': 1,
        'objective': 'rank:pairwise',
        'min_child_weight': 0.02,
        'lambda': 100
    }

    clf = xgb.train(dtrain=xgb_train, num_boost_round=5000, params=params)
    preds = clf.predict(xgb_test)

    idx_gp = 0
    idx_dg = 0
    predicted = {}

    for key, val in dev_mentions.items():
        predicted[key] = val['candidate_entities'][np.argmax(preds[idx_gp: idx_gp + dtest_groups[idx_dg]])]

        idx_gp += dtest_groups[idx_dg]
        idx_dg += 1

    return predicted


## We will be using the following function to compute the accuracy...
def compute_accuracy(result, data_labels):
    assert set(list(result.keys())) - set(list(data_labels.keys())) == set()

    tp = 0.0

    for id_ in result.keys():
        if result[id_] == data_labels[id_]['label']:
            tp += 1

    assert len(result) == len(data_labels)

    return tp / len(result)


if __name__ == '__main__':
    t_init = datetime.now()

    ### Read the Training Data
    train_file = './Data/train.pickle'
    train_mention = pickle.load(open(train_file, 'rb'))

    ### Read the Training Labels...
    train_label_file = './Data/train_labels.pickle'
    train_label = pickle.load(open(train_label_file, 'rb'))

    ### Read the Dev Data... (For Final Evaluation, we will replace it with the Test Data)
    dev_file = './Data/dev2.pickle'
    dev_mention = pickle.load(open(dev_file, 'rb'))

    ### Read the Parsed Entity Candidate Pages...
    fname = './Data/parsed_candidate_entities.pickle'
    parsed_entity_page = pickle.load(open(fname, 'rb'))

    ### Read the Mention docs...
    mens_docs_file = "./Data/men_docs.pickle"
    men_doc = pickle.load(open(mens_docs_file, 'rb'))

    ## Result of the model...
    result = disambiguate_mentions(train_mention, train_label, dev_mention, men_doc, parsed_entity_page)

    # ## Here, we print out sample result of the model for illustration...
    # for key in list(result)[:5]:
    #     print('KEY: {} \t VAL: {}'.format(key, result[key]))

    ### Read the Dev Labels... (For Final Evaluation, we will replace it with the Test Data)
    dev_label_file = './Data/dev2_labels.pickle'
    dev_labels = pickle.load(open(dev_label_file, 'rb'))

    accuracy = compute_accuracy(result, dev_labels)
    print("Accuracy = ", accuracy)

    t_final = datetime.now()
    tdiff = (t_final - t_init).total_seconds()
    print(f'\nTime taken for execution: {int(tdiff // 60)}min{int(tdiff % 60)}sec')
