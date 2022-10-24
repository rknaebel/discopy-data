import glob
import os

from tqdm import tqdm

from discopy_data.data.doc import Document
from discopy_data.data.relation import Relation


def extract(source_path: str):
    for doc_i, path in tqdm(enumerate(glob.glob(f"{source_path}/raw/*/wsj_*"))):
        content = open(path, 'r', encoding='latin-1').read()
        yield {
            'meta': {
                'corpus': 'pdtb3',
                'path': os.path.abspath(path),
                'wsj': os.path.basename(path),
            },
            'text': content,
        }


def get_spans(span_str):
    if not span_str:
        return []
    return [tuple(int(a) for a in span.split('..')) for span in span_str.strip(';').split(';')]


def update_annotations(src: str, options: dict):
    relation_files = {os.path.basename(path): path for path in glob.glob(f"{src}/gold/*/wsj_*")}
    fields = ['Relation', 'ConnSpanList', 'ConnSrc', 'ConnType', 'ConnPol', 'ConnDet', 'ConnFeatSpannList', 'Conn1',
              'SClass1A', 'SClass1B', 'Conn2', 'SClass2A', 'SClass2B', 'Sup1SpanList',
              'Arg1SpanList', 'Arg1Src', 'Arg1Type', 'Arg1Pol', 'Arg1Det', 'Arg1FeatSpanList',
              'Arg2SpanList', 'Arg2Src', 'Arg2Type', 'Arg2Pol', 'Arg2Det', 'Arg2FeatSpanList',
              'Sup2SpanList', 'AdjuReason', 'AdjuDisagr', 'PBRole', 'PBVerb', 'Offset', 'Provenance', 'Link'
              ]

    def helper(doc: Document):
        relations = [{column: value for column, value in zip(fields, line.strip().split('|'))}
                     for line in open(relation_files[doc.meta['wsj']], 'r', encoding='latin-1')]
        words = doc.get_tokens()
        doc_relations = []
        for rel in relations:
            arg1_spans = get_spans(rel['Arg1SpanList'])
            arg2_spans = get_spans(rel['Arg2SpanList'])
            conn_spans = get_spans(rel['ConnSpanList'])
            senses = [rel['SClass1A']]
            if rel['SClass1B']:
                senses.append(rel['SClass1B'])
            doc_relations.append(
                Relation([t for t in words if any(start <= t.offset_begin < end or start < t.offset_end <= end
                                                  for start, end in arg1_spans)],
                         [t for t in words if any(start <= t.offset_begin < end or start < t.offset_end <= end
                                                  for start, end in arg2_spans)],
                         [t for t in words if any(start <= t.offset_begin < end or start < t.offset_end <= end
                                                  for start, end in conn_spans)],
                         senses, rel['Relation']))
        return doc.with_relations(doc_relations)

    return helper
