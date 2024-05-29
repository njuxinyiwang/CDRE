import json
import random
from transformers import BertTokenizer
import os
from tqdm import tqdm

from spacy.tokens import Doc
import spacy
import numpy as np
nlp = spacy.load('en_core_web_sm')


def get_anaphors(sents, mentions):
    potential_mentions = []

    for sent_id, sent in enumerate(sents):
        doc_spacy = Doc(nlp.vocab, words=sent)
        for name, tool in nlp.pipeline:
            if name != 'ner':
                tool(doc_spacy)

        for token in doc_spacy:
            potential_mention = ''
            if token.dep_ == 'det' and token.text.lower() == 'the':
                potential_name = doc_spacy.text[token.idx:token.head.idx + len(token.head.text)]
                pos_start, pos_end = token.i, token.i + len(potential_name.split(' '))
                potential_mention = {
                    'pos': [pos_start, pos_end],
                    'type': 'MISC',
                    'sent_id': sent_id,
                    'name': potential_name
                }
            if token.pos_ == 'PRON':
                potential_name = token.text
                pos_start = sent.index(token.text)
                potential_mention = {
                    'pos': [pos_start, pos_start + 1],
                    'type': 'MISC',
                    'sent_id': sent_id,
                    'name': potential_name
                }

            if potential_mention:
                if not any(mention in potential_mention['name'] for mention in mentions):
                    potential_mentions.append(potential_mention)

    return potential_mentions


def add_entity_markers(sample, tokenizer, entity_start, entity_end):
    ''' add entity marker (*) at the end and beginning of entities. '''

    sents = []
    sent_map = []
    sent_pos = []

    sent_start = 0

    for i_s, sent in enumerate(sample['sents']):
        # add * marks to the beginning and end of entities
        new_map = {}

        for i_t, token in enumerate(sent):
            tokens_wordpiece = tokenizer.tokenize(token)
            if (i_s, i_t) in entity_start:
                tokens_wordpiece = ["*"] + tokens_wordpiece
            if (i_s, i_t) in entity_end:
                tokens_wordpiece = tokens_wordpiece + ["*"]
            new_map[i_t] = len(sents)
            sents.extend(tokens_wordpiece)

        sent_end = len(sents)
        # [sent_start, sent_end)
        sent_pos.append((sent_start, sent_end,))
        sent_start = sent_end

        # update the start/end position of each token.
        new_map[i_t + 1] = len(sents)
        sent_map.append(new_map)

    return sents, sent_map, sent_pos

def create_graph(entity_pos):
    anaphor_pos, entity_pos = entity_pos[-1], entity_pos[:-1]
    mention_num = len([mention for entity in entity_pos for mention in entity])
    anaphor_num = len(anaphor_pos)

    N_nodes = mention_num + anaphor_num
    nodes_adj = np.zeros((N_nodes, N_nodes), dtype=np.int32)

    edges_cnt = 1
    # add self-loop
    for i in range(N_nodes):
        nodes_adj[i, i] = edges_cnt

    edges_cnt = 2
    # add mention-anaphor edges
    for i in range(mention_num):
        for j in range(mention_num, N_nodes):
            nodes_adj[i, j] = edges_cnt
            nodes_adj[j, i] = edges_cnt

    entities = []
    i = 0
    for e in entity_pos:
        ms = []
        for _ in e:
            ms.append(i)
            i += 1
        entities.append(ms)

    edges_cnt = 3
    # add co-reference edges
    for e in entities:
        if len(e) == 1:
            continue
        for m1 in e:
            for m2 in e:
                if m1 != m2:
                    nodes_adj[m1, m2] = edges_cnt

    edges_cnt = 4
    # add inter-entity edges
    nodes_adj[nodes_adj == 0] = edges_cnt

    return nodes_adj


class data_sampler(object):

    def __init__(self, config=None, seed=None):

        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_path, additional_special_tokens=["[E11]", "[E12]", "[E21]", "[E22]"])
        self.task_length = config.task_length

        self.task_index = list(range(self.task_length))
        random.shuffle(self.task_index)
        self.seed = seed
        if self.seed != None:
            self.set_seed(self.seed)
        self.id2rel, self.rel2id = self._read_relations(config.relation_file)
        self.task2rels = {}
        self.history_test_data = {}
        self.features = self._read_data()
        self.batch = 0
        self.seen_relations = ["No relation"]


    def _read_data(self):
        features = {}
        all_relations = ["No relation"]
        for i_task in self.task_index:
            with open(os.path.join(self.config.task_splits_dir, f"task_{i_task}_rel_info.json"), "r", encoding="utf-8") as file:
                task_rel = json.loads(file.read())
            current_relations = ["No relation"] + task_rel
            all_relations += task_rel
            self.task2rels[i_task] = task_rel
            with open(os.path.join(self.config.task_splits_dir, f"task_{i_task}_train.json"), "r", encoding="utf-8") as file:
                task_data_train = json.loads(file.read())
            task_features_train = self.read_docred_file(task_data_train, current_relations, all_relations,
                                                        desc=f"Task {i_task} training data.")

            with open(os.path.join(self.config.task_splits_dir, f"task_{i_task}_dev.json"), "r",
                      encoding="utf-8") as file:
                task_data_dev = json.loads(file.read())
            task_features_dev = self.read_docred_file(task_data_dev, current_relations, all_relations,
                                                            desc=f"Task {i_task} validation data.")

            with open(os.path.join(self.config.task_splits_dir, f"task_{i_task}_test.json"), "r",
                      encoding="utf-8") as file:
                task_data_test = json.loads(file.read())
            self.history_test_data[i_task] = task_data_test
            task_features_test = self.read_docred_file(task_data_test, current_relations, all_relations,
                                                      desc=f"Task {i_task} testing data.")

            task_features = {
                "train": task_features_train,
                "dev": task_features_dev,
                "test":task_features_test,
                "current_relations": current_relations
            }
            features[i_task] = task_features

        return features

    def set_seed(self, seed):
        self.seed = seed
        if self.seed != None:
            random.seed(self.seed)
        self.task_index = list(range(self.task_length))
        random.shuffle(self.task_index)
        print(self.task_index)

    def _read_relations(self, file):

        with open(file, 'r', encoding='utf-8') as file:
            rel2id = json.loads(file.read())
        id2rel = {}
        for key in rel2id.keys():
            id2rel[rel2id[key]] = key

        return id2rel, rel2id

    def __iter__(self):
        return self

    def __next__(self):

        if self.batch == self.task_length:
            self.batch = 0
            raise StopIteration()

        current_features = self.features[self.task_index[self.batch]]
        self.seen_relations += self.task2rels[self.task_index[self.batch]]
        self.batch += 1

        cur_training_data = current_features["train"]
        cur_valid_data = current_features["dev"]
        cur_test_data = current_features["test"]
        current_relations = current_features["current_relations"]

        history_data = []
        for i in range(self.batch):
            for sample in self.history_test_data[self.task_index[i]]:
                if sample not in history_data:
                    history_data.append(sample)
        history_test_data = self.read_docred_file(history_data, self.seen_relations, self.seen_relations)

        return cur_training_data, cur_valid_data, cur_test_data, current_relations, history_test_data, self.seen_relations

    def read_docred_file(self, data, current_relations, all_relations, desc=None, max_seq_length=1024):
        i_line = 0
        pos_samples = 0
        neg_samples = 0
        features = []

        for doc_id in tqdm(range(len(data)), desc=desc):

            sample = data[doc_id].copy()
            entities = sample['vertexSet'][:]
            entity_start, entity_end = [], []
            # record entities
            for entity in entities:
                for mention in entity:
                    sent_id = mention["sent_id"]
                    pos = mention["pos"]
                    entity_start.append((sent_id, pos[0],))
                    entity_end.append((sent_id, pos[1] - 1,))

            # add entity markers
            sents, sent_map, sent_pos = add_entity_markers(sample, self.tokenizer, entity_start, entity_end)


            train_triple = {}

            if "labels" in sample:
                for label in sample['labels']:
                    if label['r'] not in current_relations:
                        continue
                    evidence = label['evidence']
                    # r = int(self.rel2id[label['r']])
                    r = all_relations.index(label['r'])
                    # update training triples
                    if (label['h'], label['t']) not in train_triple:
                        train_triple[(label['h'], label['t'])] = [
                            {'relation': r, 'evidence': evidence}]
                    else:
                        train_triple[(label['h'], label['t'])].append(
                            {'relation': r, 'evidence': evidence})

            # get anaphors in the doc
            mentions = set([m['name'] for e in entities for m in e])

            potential_mention = get_anaphors(sample['sents'], mentions)

            entities.append(potential_mention)

            # entity start, end position
            entity_pos = []

            for e in entities:
                entity_pos.append([])
                for m in e:
                    start = sent_map[m["sent_id"]][m["pos"][0]]
                    end = sent_map[m["sent_id"]][m["pos"][1]]
                    label = m["type"]
                    entity_pos[-1].append((start, end,))

            relations, hts, sent_labels = [], [], []

            # for h, t in train_triple.keys():  # for every entity pair with gold relation
            #     relation = [0] * len(self.rel2id)
            #     sent_evi = [0] * len(sent_pos)
            #
            #     for mention in train_triple[h, t]:  # for each relation mention with head h and tail t
            #         relation[mention["relation"]] = 1
            #         for i in mention["evidence"]:
            #             sent_evi[i] += 1
            #
            #     relations.append(relation)
            #     hts.append([h, t])
            #     sent_labels.append(sent_evi)
            #     pos_samples += 1
            #
            # for h in range(len(entities) - 1):
            #     for t in range(len(entities) - 1):
            #         # all entity pairs that do not have relation are treated as negative samples
            #         if h != t and [h, t] not in hts:  # and [t, h] not in hts:
            #             relation = [1] + [0] * (len(self.rel2id) - 1)
            #             sent_evi = [0] * len(sent_pos)
            #             relations.append(relation)
            #
            #             hts.append([h, t])
            #             sent_labels.append(sent_evi)
            #             neg_samples += 1
            observe_matrix = []
            for h in range(len(entities)-1):
                for t in range(len(entities)-1):
                    if h != t:
                        observe_matrix.append([0]*len(all_relations))
                        for rel in current_relations:
                            observe_matrix[-1][all_relations.index(rel)] = 1
                        if (h, t) in train_triple.keys():
                            relation = [0] * len(all_relations)
                            sent_evi = [0] * len(sent_pos)

                            for mention in train_triple[h, t]:  # for each relation mention with head h and tail t

                                relation[mention["relation"]] = 1
                                for i in mention["evidence"]:
                                    sent_evi[i] += 1

                            relations.append(relation)
                            hts.append([h, t])
                            sent_labels.append(sent_evi)
                            pos_samples += 1
                        else:
                            relation = [1] + [0] * (len(all_relations) - 1)
                            sent_evi = [0] * len(sent_pos)
                            relations.append(relation)

                            hts.append([h, t])
                            sent_labels.append(sent_evi)
                            neg_samples += 1


            graph = create_graph(entity_pos)

            assert len(relations) == (len(entities)-1) * (len(entities) - 2)
            assert len(sents) < max_seq_length
            sents = sents[:max_seq_length - 2]  # truncate, -2 for [CLS] and [SEP]
            input_ids = self.tokenizer.convert_tokens_to_ids(sents)
            input_ids = self.tokenizer.build_inputs_with_special_tokens(input_ids)

            feature = [{'input_ids': input_ids,
                        'entity_pos': entity_pos if entity_pos[-1] != [] else entity_pos[:-1],
                        'labels': relations,
                        'observe_matrix': observe_matrix,
                        'hts': hts,
                        'sent_pos': sent_pos,
                        'sent_labels': sent_labels,
                        'title': sample['title'],
                        'graph': graph
                        }]



            i_line += len(feature)
            features.extend(feature)

        return features