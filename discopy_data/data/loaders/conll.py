import json
import logging
import os
from collections import defaultdict
from typing import List

import joblib
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, TFAutoModel

from discopy_data.data.conn_head_mapper import ConnHeadMapper
from discopy_data.data.doc import Document
from discopy_data.data.relation import Relation
from discopy_data.data.sentence import DepRel, Sentence
from discopy_data.data.token import Token
from discopy_data.nn.bert import get_sentence_embeddings, simple_map


def convert_sense(s, lvl):
    if lvl > 0:
        return '.'.join(s.split('.')[:lvl])
    else:
        return s


def load_parsed_conll_dataset(conll_path: str, simple_connectives=False, limit=0, sense_level=-1) -> List[Document]:
    parses_path = os.path.join(conll_path, 'parses.json')
    relations_path = os.path.join(conll_path, 'relations.json')
    docs = []
    parses = json.loads(open(parses_path, 'r').read())
    pdtb = defaultdict(list)
    for relation in [json.loads(s) for s in open(relations_path, 'r').readlines()]:
        pdtb[relation['DocID']].append(relation)
    for doc_id, doc in tqdm(parses.items(), total=len(parses)):
        if 0 < limit < len(docs):
            break
        words = []
        token_offset = 0
        sents = []
        for sent_i, sent in enumerate(doc['sentences']):
            sent_words = [
                Token(token_offset + w_i, sent_i, w_i, t['CharacterOffsetBegin'], t['CharacterOffsetEnd'], surface,
                      xpos=t['PartOfSpeech'])
                for w_i, (surface, t) in enumerate(sent['words'])
            ]
            words.extend(sent_words)
            token_offset += len(sent_words)
            dependencies = [
                DepRel(rel=rel,
                       head=sent_words[int(head.split('-')[-1]) - 1] if not head.startswith('ROOT') else None,
                       dep=sent_words[int(dep.split('-')[-1]) - 1]
                       ) for rel, head, dep in sent['dependencies']
            ]
            sents.append(Sentence(sent_words, dependencies=dependencies, parsetree=sent['parsetree']))
        doc_pdtb = pdtb.get(doc_id, [])
        if simple_connectives:
            connective_head(doc_pdtb)
        relations = [
            Relation([words[i[2]] for i in rel['Arg1']['TokenList']],
                     [words[i[2]] for i in rel['Arg2']['TokenList']],
                     [words[i[2]] for i in rel['Connective']['TokenList']],
                     [convert_sense(s, sense_level) for s in rel['Sense']], rel['Type']) for rel in doc_pdtb
        ]
        docs.append(Document(doc_id=doc_id, sentences=sents, relations=relations))
    return docs


def load_bert_conll_dataset(conll_path: str, simple_connectives=False, limit=0, cache_dir='',
                            bert_model='bert-base-cased', sense_level=-1) -> List[Document]:
    docs = load_parsed_conll_dataset(conll_path, simple_connectives, limit, sense_level)
    logging.info(f'Load {bert_model} Embeddings')
    docs = load_bert_embeddings(docs, cache_dir, bert_model)
    return docs


def load_bert_embeddings(docs: List[Document], cache_dir='', bert_model='bert-base-cased') -> List[Document]:
    if cache_dir and os.path.exists(cache_dir):
        doc_embeddings = joblib.load(cache_dir)
        tokenizer = None
        model = None
        preloaded = True
    else:
        doc_embeddings = {}
        tokenizer = AutoTokenizer.from_pretrained(bert_model)
        model = TFAutoModel.from_pretrained(bert_model)
        preloaded = False
    logging.info(f'Use preloaded embeddings: {preloaded}')
    for doc in tqdm(docs):
        doc_embedding = []
        for sent_i, sent in enumerate(doc.sentences):
            if preloaded:
                token_offset = sent.tokens[0].idx
                embeddings = doc_embeddings[doc.doc_id][token_offset:token_offset + len(sent.tokens)]
            else:
                embeddings = get_sentence_embeddings(sent.tokens, tokenizer, model)
                doc_embedding.append(embeddings)
            sent.embeddings = embeddings
        if doc_embedding:
            doc_embeddings[doc.doc_id] = np.concatenate(doc_embedding)
    if cache_dir and not preloaded:
        joblib.dump(doc_embeddings, cache_dir)
    return docs


def load_embeddings_conll_dataset(conll_path: str, embedder: 'TokenSentenceEmbedder', simple_connectives=False, limit=0,
                                  sense_level=-1) -> List[Document]:
    docs = load_parsed_conll_dataset(conll_path, simple_connectives, limit, sense_level)
    for doc in docs:
        for sent_i, sent in enumerate(doc.sentences):
            embeddings = embedder.get_sentence_vector_embeddings(sent.tokens)
            sent.embeddings = embeddings
    return docs


class TokenSentenceEmbedder:

    def __init__(self, vector_path):
        def get_coefs(word, *arr):
            return word, np.asarray(arr, dtype=np.float32)

        embedding_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(vector_path, encoding='utf8'))
        print('Found %s word vectors.' % len(embedding_index))
        all_embs = np.stack([e for e in embedding_index.values() if len(e) >= 100])
        self.embedding_index = embedding_index
        self.mean = all_embs.mean()
        self.std = all_embs.std()
        self.embedding_dim = len(next(iter(self.embedding_index.values())))

    def get_sentence_vector_embeddings(self, tokens: List[Token]):
        embeddings = np.random.normal(self.mean, self.std, (len(tokens), self.embedding_dim))
        for i, tok in enumerate(tokens):
            tok = simple_map.get(tok.surface, tok.surface).lower()
            if tok in self.embedding_index:
                embeddings[i] = self.embedding_index[tok]
        return embeddings


def connective_head(pdtb):
    chm = ConnHeadMapper()
    for rel in filter(lambda r: (r['Type'] == 'Explicit') and (len(r['Connective']['CharacterSpanList']) == 1), pdtb):
        head, head_idxs = chm.map_raw_connective(rel['Connective']['RawText'])
        rel['Connective']['TokenList'] = [rel['Connective']['TokenList'][i] for i in head_idxs]
        rel['Connective']['RawText'] = head
        rel['Connective']['CharacterSpanList'] = [
            [rel['Connective']['TokenList'][0][0], rel['Connective']['TokenList'][-1][1]]]
