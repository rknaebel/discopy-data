import json
import os
from collections import defaultdict
from typing import List

from discopy_data.data.doc import Document
from discopy_data.data.relation import Relation


def extract(src: str):
    docs = json.load(open(os.path.join(src, "parses.json")))
    for doc_id, doc in docs.items():
        sentences = [''.join([sentence['words'][0][0]] +
                             [('' if sentence['words'][t_i][1]['CharacterOffsetEnd'] == t[1][
                                 'CharacterOffsetBegin'] else ' ') + t[0]
                              for t_i, t in enumerate(sentence['words'][1:])]) for sentence in doc['sentences']]
        yield {
            'docID': doc_id,
            'meta': {
                'corpus': 'pdtb',
                'part': os.path.basename(src),
            },
            'text': "\n".join(sentences),
            'sentences': doc['sentences'],
        }


def update_annotations(src: str, docs: List[Document]):
    relations_grouped = defaultdict(list)
    relations = [json.loads(line) for line in open(os.path.join(src, "relations.json"))]
    for rel in relations:
        relations_grouped[rel['DocID']].append(rel)
    for doc in docs:
        words = doc.get_tokens()
        doc_relations = [
            Relation([words[t[2]] for t in rel['Arg1']['TokenList']],
                     [words[t[2]] for t in rel['Arg2']['TokenList']],
                     [words[t[2]] for t in rel['Connective']['TokenList']],
                     rel['Sense'],
                     rel['Type']) for rel in relations_grouped[doc.doc_id]
        ]
        yield doc.with_relations(doc_relations)
