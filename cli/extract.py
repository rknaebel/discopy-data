import datetime
import os
import sys
from typing import List

import click
import trankit
import ujson as json
from tqdm import tqdm

import discopy_data.dataset.anthology
import discopy_data.dataset.argessay
import discopy_data.dataset.bbc
import discopy_data.dataset.global_voices
import discopy_data.dataset.pdtb
import discopy_data.dataset.press_gov
import discopy_data.dataset.short_stories
import discopy_data.dataset.ted
import discopy_data.dataset.un_debates
from discopy_data.data.doc import Document
from discopy_data.data.sentence import Sentence, DepRel
from discopy_data.data.token import Token

document_extractor = {
    'anthology': discopy_data.dataset.anthology.extract,
    'args-essay': discopy_data.dataset.argessay.extract,
    'bbc': discopy_data.dataset.bbc.extract,
    'gv': discopy_data.dataset.global_voices.extract,
    'pdtb': discopy_data.dataset.pdtb.extract,
    'press-gov': discopy_data.dataset.press_gov.extract,
    'short-stories': discopy_data.dataset.short_stories.extract,
    'ted': discopy_data.dataset.ted.extract,
    'un-debates': discopy_data.dataset.un_debates.extract,
}


def load_parser(use_gpu=False):
    tmp_stdout = sys.stdout
    sys.stdout = sys.stderr
    parser = trankit.Pipeline('english', cache_dir=os.path.expanduser('~/.trankit/'), gpu=use_gpu)
    # parser._tagbatchsize = tagbatch
    parser("Init")
    sys.stdout = tmp_stdout
    return parser


def get_parsed_sentences_raw(parser, doc):
    parsed = parser(doc['text'])
    token_offset = 0
    sents = []
    for sent_i, sent in enumerate(parsed['sentences']):
        words = [
            Token(token_offset + w_i, sent_i, w_i, t['dspan'][0], t['dspan'][1], t['text'],
                  t['upos'])
            for w_i, t in enumerate(sent['tokens'])
        ]
        dependencies = [
            DepRel(rel=t['deprel'].lower(),
                   head=words[int(t['head']) - 1] if t['deprel'].lower() != 'root' else None,
                   dep=words[dep]
                   ) for dep, t in enumerate(sent['tokens'])
        ]
        token_offset += len(words)
        sents.append(Sentence(words, dependencies=dependencies).to_json())
    return sents


simple_map = {
    "''": '"',
    "``": '"',
    "-LRB-": "(",
    "-RRB-": ")",
    "-LCB-": "{",
    "-RCB-": "}",
    "n't": "not"
}


def get_parsed_sentences_tokenized(parser, doc):
    doc = Document.from_json(doc)
    parser_in = [[simple_map.get(t.surface, t.surface) for t in sent.tokens] for sent in doc.sentences]
    parsed = parser(parser_in)
    token_offset = 0
    sents = []
    for sent_i, sent in enumerate(parsed['sentences']):
        words = doc.sentences[sent_i].tokens
        assert len(words) == len(sent['tokens']), 'different number of tokens after parsing'
        for w, t in zip(words, sent['tokens']):
            w.pos = t['upos']
        dependencies = [
            DepRel(rel=t['deprel'].lower(),
                   head=words[int(t['head']) - 1] if t['deprel'].lower() != 'root' else None,
                   dep=words[dep]
                   ) for dep, t in enumerate(sent['tokens'])
        ]
        token_offset += len(words)
        sents.append(Sentence(words, dependencies=dependencies).to_json())
    return sents


def get_parsed_sentences(parser, doc) -> List[dict]:
    if 'sentences' in doc:
        return get_parsed_sentences_tokenized(parser, doc)
    else:
        return get_parsed_sentences_raw(parser, doc)


@click.command()
@click.argument('corpus', type=str)
@click.argument('src', type=click.Path('r'))
@click.option('-o', '--tgt', default='-', type=click.File('w'))
@click.option('-l', '--limit', default=0, type=int)
def main(corpus, src, tgt, limit):
    parser = load_parser()
    t = tqdm()
    doc_i = 0
    for doc in document_extractor[corpus](src):
        if limit and doc_i >= limit:
            break
        if 'docID' not in doc:
            doc['docID'] = f"{doc['meta']['corpus']}_{doc_i:05}"
        doc['meta']['created'] = datetime.datetime.now().isoformat()
        sentences = get_parsed_sentences(parser, doc)
        doc['sentences'] = sentences
        tgt.write(json.dumps(doc) + '\n')
        doc_i += 1
        t.update(1)
    t.close()


if __name__ == '__main__':
    main()
