import datetime
import os
import sys
from typing import List

from tqdm import tqdm

from discopy_data.data.doc import Document
from discopy_data.data.sentence import Sentence, DepRel
from discopy_data.data.token import Token


def load_parser(use_gpu=False):
    import trankit
    tmp_stdout = sys.stdout
    sys.stdout = sys.stderr
    parser = trankit.Pipeline('english', cache_dir=os.path.expanduser('~/.trankit/'), gpu=use_gpu)
    parser.tokenize("Init")
    sys.stdout = tmp_stdout
    return parser


def get_tokenized_sentences(parser, text):
    parsed = parser.tokenize(text)
    token_offset = 0
    sents = []
    for sent_i, sent in enumerate(parsed['sentences']):
        words = [
            Token(token_offset + w_i, sent_i, w_i, t['dspan'][0], t['dspan'][1], t['text'])
            for w_i, t in enumerate(sent['tokens'])
        ]
        token_offset += len(words)
        sents.append(Sentence(words).to_json())
    return sents


def get_parsed_sentences(parser, text):
    parsed = parser(text)
    token_offset = 0
    sents = []
    for sent_i, sent in enumerate(parsed['sentences']):
        words = [
            Token(token_offset + w_i, sent_i, w_i, t['dspan'][0], t['dspan'][1], t['text'],
                  upos=t['upos'], xpos=t['xpos'], lemma=t['lemma'])
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


def load_texts(texts: List[str], tokenize_only=False) -> List[Document]:
    parser = load_parser()
    for text_i, text in tqdm(enumerate(texts)):
        sentences = get_tokenized_sentences(parser, text) if tokenize_only else get_parsed_sentences(parser, text)
        yield Document.from_json({
            'docID': hash(text),
            'meta': {
                'fileID': f'raw_{text_i:05}',
                'corpus': 'raw',
                'created': datetime.datetime.now().isoformat(),
            },
            'text': text,
            'sentences': sentences
        }, load_relations=False)


def load_texts_fast(texts: List[str], tokenize_only=True) -> List[Document]:
    from nltk.tokenize import sent_tokenize
    from nltk.tokenize import TreebankWordTokenizer

    for text_i, text in tqdm(enumerate(texts)):
        sentence_splits = [len(s) for s in sent_tokenize(text)]
        sents = []
        sent_i = 0
        w_i = 0
        words = []
        sent_offset = 0
        for tok_i, (tok_start, tok_end) in enumerate(TreebankWordTokenizer().span_tokenize(text)):
            form = text[tok_start:tok_end]
            words.append(Token(tok_i, sent_i, w_i, tok_start, tok_end, form))
            w_i += 1
            if tok_end >= (sent_offset + sentence_splits[sent_i]):
                sents.append(Sentence(words).to_json())
                sent_i += 1
                if sent_i >= len(sentence_splits):
                    break
                w_i = 0
                sent_offset += sentence_splits[sent_i]

        yield Document.from_json({
            'docID': hash(text),
            'meta': {
                'fileID': f'raw_{text_i:05}',
                'corpus': 'raw',
                'created': datetime.datetime.now().isoformat(),
            },
            'text': text,
            'sentences': sents
        }, load_relations=False)
