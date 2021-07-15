import json
import os
from collections import defaultdict

from discopy_data.data.doc import Document
from discopy_data.data.relation import Relation
from discopy_data.data.sentence import Sentence
from discopy_data.data.token import Token


# from glob import glob
#
# import discourse.dataset.bbc
# import discourse.dataset.global_voices
# import discourse.dataset.label
# import discourse.dataset.ted
# import numpy as np
# import pandas as pd
# import spacy
# from tqdm import tqdm
#
# nlp = spacy.load('en')
#
#
# def parse_text(text):
#     sentences = []
#     offset = 0
#     for doc, line in nlp.pipe([(t, t) for t in text.splitlines()], as_tuples=True):
#         sents = list(doc.sents)
#         for sent in sents:
#             sentences.append({
#                 'Sentence': sent.string.strip(),
#                 'Length': len(sent.string),
#                 'Tokens': [t.text for t in sent],
#                 'POS': [t.pos_ for t in sent],
#                 'Offset': [t.idx - sent[0].idx for t in sent],
#                 'Dep': [(t.dep_, (t.head.text, t.head.i), (t.text, t.i)) for t in sent],
#                 'SentenceOffset': offset + sent[0].idx,
#                 'Parse': sent._.parse_string,
#                 'NER_iob': [t.ent_iob_ for t in sent],
#                 'NER_type': [t.ent_type_ for t in sent],
#             })
#         sentences.append(dict())
#         offset += len(line) + 1
#     return sentences
#
#
# def parse_wsj_file(path):
#     wsj = []
#     with open(path, 'r', encoding='latin-1') as fh:
#         wsj_filename = path.split('/')[-1]
#         section, file_no = wsj_filename[-4:-2], wsj_filename[-2:]
#
#         wsj_file = fh.read()
#
#         for sentence_no, sent in enumerate(parse_text(wsj_file)[1:]):
#             if sent:
#                 sent.update({
#                     'Section': int(section),
#                     'FileNumber': int(file_no),
#                     'SentenceNumber': int(sentence_no)
#                 })
#                 wsj.append(sent)
#     return wsj
#
#
# def load_wsj(path):
#     wsj_file_paths = glob(path + '/raw/*/wsj_*')
#     wsj = []
#     for p in tqdm(wsj_file_paths, total=len(wsj_file_paths)):
#         wsj.extend(discourse.dataset.parse_wsj_file(p))
#     return wsj
#
#
# def load_wsj_df(path):
#     wsj = load_wsj(path)
#     wsj = pd.DataFrame(wsj)
#     wsj = wsj.set_index(['Section', 'FileNumber', 'SentenceNumber']).sort_index().reset_index()
#
#     v_words = discourse.utils.Vocab().fit_on_texts(wsj.Tokens)
#     v_pos = discourse.utils.Vocab().fit_on_texts(wsj.POS)
#     v_dep = discourse.utils.Vocab().fit_on_texts(wsj.Dep.apply(lambda deps: [x[0] for x in deps]))
#
#     wsj['Tokens_'] = wsj.Tokens.apply(v_words.sequence_to_ids)
#     wsj['POS_'] = wsj.POS.apply(v_pos.sequence_to_ids)
#     wsj['Dep_'] = wsj.Dep.apply(v_dep.sequence_to_ids)
#
#     return wsj, v_words, v_pos, v_dep
#
#
# def load_pdtb2(path):
#     pdtb = pd.read_csv(path + '/pdtb2.csv', encoding='latin-1')
#
#     pdtb = pdtb[np.logical_not(pdtb.Relation.isna())]
#     pdtb.Relation = pdtb.Relation.astype('category')
#     pdtb.Section = pdtb.Section.astype('int32')
#     pdtb.FileNumber = pdtb.FileNumber.astype('int32')
#
#     # extend data by semantic classes
#     pdtb_extend = pdtb[pdtb.ConnHeadSemClass2.notna()].copy()
#
#     pdtb['SemClassAll'] = pdtb.ConnHeadSemClass1
#     pdtb_extend['SemClassAll'] = pdtb_extend.ConnHeadSemClass2
#
#     pdtb = pd.concat([pdtb, pdtb_extend])
#     pdtb = pdtb.drop(columns=['ConnHeadSemClass1', 'ConnHeadSemClass2'])
#     pdtb.SemClassAll = pdtb.SemClassAll.astype(str).astype('category')
#
#     pdtb['Sense1'] = pdtb.SemClassAll.apply(lambda x: '.'.join(x.split('.')[:1])).astype('category')
#     pdtb['Sense2'] = pdtb.SemClassAll.apply(lambda x: '.'.join(x.split('.')[:2])).astype('category')
#     pdtb['Sense3'] = pdtb.SemClassAll.apply(lambda x: '.'.join(x.split('.')[:3])).astype('category')
#
#     return pdtb
#
#
# def load_pdtb3(pdtb_path):
#     def parse_item(line, path):
#         item = line.strip().split('|')
#         fields = ['Relation', 'ConnSpanList', 'ConnSrc', 'ConnType', 'ConnPol', 'ConnDet', 'ConnFeatSpannList', 'Conn1',
#                   'SClass1A', 'SClass1B', 'Conn2', 'SClass2A', 'SClass2B', 'Sup1SpanList',
#                   'Arg1SpanList', 'Arg1Src', 'Arg1Type', 'Arg1Pol', 'Arg1Det', 'Arg1FeatSpanList',
#                   'Arg2SpanList', 'Arg2Src', 'Arg2Type', 'Arg2Pol', 'Arg2Det', 'Arg2FeatSpanList',
#                   'Sup2SpanList', 'AdjuReason', 'AdjuDisagr', 'PBRole', 'PBVerb', 'Offset', 'Provenance', 'Link'
#                   ]
#         assert len(item) == len(fields)
#         item = {k: v for (k, v) in zip(fields, item)}
#         section = int(path[-4:-2])
#         filenumber = int(path[-2:])
#         item['Section'] = section
#         item['FileNumber'] = filenumber
#         item['wsj_id'] = "wsj_{:02}{:02}".format(section, filenumber)
#         return item
#
#     tmp = []
#     for path in glob(pdtb_path + '/gold/*/wsj_*'):
#         with open(path, 'r', encoding='latin-1') as fh:
#             tmp.extend([parse_item(item, path) for item in fh.readlines()])
#
#     pdtb = pd.DataFrame(tmp)
#
#     pdtb['SemClassAll'] = pdtb.SClass1A
#     pdtb.SemClassAll = pdtb.SemClassAll.astype(str).astype('category')
#
#     pdtb['Sense1'] = pdtb.SemClassAll.apply(lambda x: '.'.join(x.split('.')[:1]) if x else None).astype('category')
#     pdtb['Sense2'] = pdtb.SemClassAll.apply(lambda x: '.'.join(x.split('.')[:2]) if x else None).astype('category')
#     pdtb['Sense3'] = pdtb.SemClassAll.apply(lambda x: '.'.join(x.split('.')[:3]) if x else None).astype('category')
#
#     pdtb.rename(columns={
#         'Arg1SpanList': 'Arg1_SpanList',
#         'Arg2SpanList': 'Arg2_SpanList',
#         'ConnSpanList': 'Connective_SpanList',
#     },
#         inplace=True)
#
#     return pdtb


def extract(src: str):
    for part in ['en.train', 'en.dev', 'en.test', 'en.blind-test']:
        try:
            docs = json.load(open(os.path.join(src, part, "parses.json")))
        except FileNotFoundError:
            continue
        for doc_id, doc in docs.items():
            words = []
            token_offset = 0
            sentences = []
            for sent_i, sent in enumerate(doc['sentences']):
                sent_words = [
                    Token(token_offset + w_i, sent_i, w_i, t['CharacterOffsetBegin'], t['CharacterOffsetEnd'], surface,
                          t['PartOfSpeech'])
                    for w_i, (surface, t) in enumerate(sent['words'])
                ]
                words.extend(sent_words)
                token_offset += len(sent_words)
                sentences.append(Sentence(sent_words))
            meta = {
                'corpus': 'pdtb',
                'part': part,
            }
            yield Document(doc_id=doc_id, sentences=sentences, relations=[], meta=meta).to_json()


def update_annotations(src: str):
    relations_grouped = defaultdict(list)
    for part in ['en.train', 'en.dev', 'en.test', 'en.blind-test']:
        try:
            relations = [json.loads(line) for line in open(os.path.join(src, part, "relations.json"))]
            for rel in relations:
                relations_grouped[rel['DocID']].append(rel)
        except FileNotFoundError:
            continue

    def helper(doc: Document):
        words = doc.get_tokens()
        doc_relations = [
            Relation([words[t[2]] for t in rel['Arg1']['TokenList']],
                     [words[t[2]] for t in rel['Arg2']['TokenList']],
                     [words[t[2]] for t in rel['Connective']['TokenList']],
                     rel['Sense'],
                     rel['Type']) for rel in relations_grouped[doc.doc_id]
        ]
        return doc.with_relations(doc_relations)

    return helper
