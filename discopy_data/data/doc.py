import json
import sys
from typing import List

import numpy as np

from .relation import Relation
from .sentence import Sentence, DepRel
from .token import Token


class Document:
    def __init__(self, doc_id, sentences: List[Sentence], relations: List[Relation], meta: dict = None):
        self.doc_id = doc_id
        self.meta = meta or {}
        self.sentences: List[Sentence] = sentences
        self.relations: List[Relation] = relations
        self.text = '\n'.join([s.get_text() for s in self.sentences])

    def to_json(self):
        return {
            'docID': self.doc_id,
            'meta': self.meta,
            'text': self.text,
            'sentences': [s.to_json() for s in self.sentences],
            'relations': [r.to_json(self.doc_id, rel_id=r_i) for r_i, r in enumerate(self.relations)]
        }

    @staticmethod
    def from_json(doc: dict, load_dependencies=True, load_relations=True):
        words = []
        token_offset = 0
        sents = []
        for sent_i, sent in enumerate(doc['sentences']):
            sent_words = [
                Token(token_offset + w_i, sent_i, w_i, t['characterOffsetBegin'], t['characterOffsetEnd'], t['surface'],
                      upos=t.get('upos', ""), xpos=t.get('xpos', ""), lemma=t.get('lemma', ""))
                for w_i, t in enumerate(sent['tokens'])
            ]
            words.extend(sent_words)
            token_offset += len(sent_words)
            if load_dependencies:
                dependencies = [
                    DepRel(rel=t['deprel'],
                           head=sent_words[int(t['head'])] if t['head'] > 0 else None,
                           dep=sent_words[t_i]
                           ) for t_i, t in enumerate(sent.get('dependencies', []))
                ]
                if not dependencies:
                    sys.stderr.write(f"No dependencies in {doc['docID']}-{sent_i}")
            else:
                dependencies = []
            sents.append(Sentence(sent_words, dependencies=dependencies, parsetree=sent.get('parsetree')))
        if load_relations:
            relations = [
                Relation([words[i] for i in rel['Arg1']['TokenList']],
                         [words[i] for i in rel['Arg2']['TokenList']],
                         [words[i] for i in rel['Connective']['TokenList']],
                         rel['Sense'], rel['Type'])
                for rel in doc.get('relations', [])
            ]
        else:
            relations = []
        return Document(doc_id=doc['docID'], sentences=sents, relations=relations, meta=doc.get('meta'))

    def get_tokens(self):
        return [token for sent in self.sentences for token in sent.tokens]

    def get_embeddings(self) -> np.array:
        return np.concatenate([s.get_embeddings() for s in self.sentences], axis=0)

    def get_embedding_dim(self) -> int:
        return int(self.sentences[0].embeddings.shape[-1])

    def with_relations(self, relations):
        return Document(self.doc_id, self.sentences, relations, meta=self.meta)

    def __str__(self):
        return json.dumps(self.to_json(), indent=2)

    def get_explicit_relations(self):
        return [r for r in self.relations if r.is_explicit()]
