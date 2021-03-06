import re
import string
from typing import List

RE_PUNCT = re.compile(r'^[\s{}]+$'.format(re.escape(string.punctuation)))


class Token:

    def __init__(self, idx, sent_idx, local_idx, offset_begin, offset_end, surface, upos="", xpos="", lemma=""):
        self.surface: str = surface
        self.upos: str = upos
        self.xpos: str = xpos
        self.lemma: str = lemma
        # global word index
        self.idx: int = idx
        # sentence index
        self.sent_idx: int = sent_idx
        # local word index regarding sentence
        self.local_idx: int = local_idx
        self.offset_begin: int = offset_begin
        self.offset_end: int = offset_end

    def __str__(self):
        return f"{self.idx}-{self.surface}:{self.upos}"

    def __hash__(self):
        return hash(str(self))

    __repr__ = __str__

    def to_json(self):
        return {
            'surface': self.surface,
            'characterOffsetBegin': self.offset_begin,
            'characterOffsetEnd': self.offset_end,
            'upos': self.upos,
            'xpos': self.xpos,
            'lemma': self.lemma
        }

    def to_json_indices(self):
        return self.offset_begin, self.offset_end, self.idx, self.sent_idx, self.local_idx

    def __eq__(self, other: 'Token'):
        return all([
            self.idx == other.idx, self.surface == other.surface,
            self.sent_idx == other.sent_idx, self.local_idx == other.local_idx, self.offset_begin == other.offset_begin,
            self.offset_end == other.offset_end
        ])


class TokenSpan:

    def __init__(self, tokens):
        self.tokens: List[Token] = list(tokens)

    def get_sentence_idxs(self):
        return sorted(set(t.sent_idx for t in self.tokens))

    def get_character_spans(self):
        spans = []
        if not self.tokens:
            return []
        span_begin = self.tokens[0].offset_begin
        span_end = self.tokens[0].offset_end
        cur_tok_idx = self.tokens[0].idx
        for t in self.tokens[1:]:
            if t.idx != cur_tok_idx + 1:
                spans.append((span_begin, span_end))
                span_begin = t.offset_begin
            span_end = t.offset_end
            cur_tok_idx = t.idx
        spans.append((span_begin, span_end))
        return spans

    def overlap(self, other: 'TokenSpan') -> int:
        return sum(int(i == j) for i in self.tokens for j in other.tokens)

    def without_punct(self):
        return TokenSpan(t for t in self.tokens if not RE_PUNCT.match(t.surface))

    def add(self, token: Token):
        self.tokens.append(token)

    def __or__(self, other):
        tokens = sorted(set(self.tokens) | set(other.tokens), key=lambda t: t.idx)
        # TODO consistency check!
        return TokenSpan(tokens)

    def __and__(self, other):
        tokens = sorted(set(self.tokens) & set(other.tokens), key=lambda t: t.idx)
        return TokenSpan(tokens)

    def __len__(self):
        return len(self.tokens)
