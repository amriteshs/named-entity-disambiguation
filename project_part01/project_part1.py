## Import Libraries and Modules here...
import spacy
import math
from collections import defaultdict
from itertools import combinations, permutations


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
