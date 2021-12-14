from typing import List

import numpy as np

from discopy_data.data.sentence import Sentence
from discopy_data.data.token import Token

simple_map = {
    "''": '"',
    "``": '"',
    "-LRB-": "(",
    "-RRB-": ")",
    "-LCB-": "{",
    "-RCB-": "}",
    "n't": "not"
}


def get_sentence_embedder(bert_model):
    from transformers import AutoTokenizer, TFAutoModel
    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    model = TFAutoModel.from_pretrained(bert_model)

    def helper(tokens: List[Token]):
        return get_sentence_embeddings(tokens, tokenizer, model)

    return helper


def get_sentence_embeddings(tokens: List[Token], tokenizer, model, last_hidden_only=False):
    subtokens = [tokenizer.tokenize(simple_map.get(t.surface, t.surface)) for t in tokens]
    lengths = [len(s) for s in subtokens]
    tokens_ids = tokenizer.convert_tokens_to_ids([ts for t in subtokens for ts in t])
    tokens_ids = tokenizer.build_inputs_with_special_tokens(tokens_ids)
    outputs = model(np.array([tokens_ids]), output_hidden_states=True)
    if last_hidden_only:
        hidden_state = outputs.last_hidden_state.numpy()
    else:
        hidden_state = np.concatenate(outputs.hidden_states[-4:], axis=-1)[0]
    embeddings = np.zeros((len(lengths), hidden_state.shape[-1]), np.float32)
    len_left = 1
    for i, length in enumerate(lengths):
        embeddings[i] = hidden_state[len_left]
        len_left += length
    return embeddings


def get_sentence_vector_embeddings(tokens: List[Token], embedding_index, mean, std):
    embedding_dim = len(next(iter(embedding_index.values())))
    embeddings = np.random.normal(mean, std, (len(tokens), embedding_dim))
    for i, tok in enumerate(tokens):
        tok = simple_map.get(tok.surface, tok.surface)
        if tok in embedding_index:
            embeddings[i] = embedding_index[tok]
    return embeddings


def get_doc_sentence_embeddings(sentences: List[Sentence], tokenizer, model, last_hidden_only=False):
    tokens = [[simple_map.get(t.surface, t.surface) for t in sent.tokens] for sent in sentences]
    subtokens = [[tokenizer.tokenize(t) for t in sent] for sent in tokens]
    lengths = [[len(t) for t in s] for s in subtokens]
    inputs = tokenizer(tokens, padding=True, return_tensors='tf', is_split_into_words=True)
    outputs = model(inputs, output_hidden_states=True)
    if last_hidden_only:
        hidden_state = outputs.hidden_states[-2].numpy()
    else:
        hidden_state = np.concatenate(outputs.hidden_states[-4:], axis=-1)
    embeddings = np.zeros((sum(len(s) for s in tokens), hidden_state.shape[-1]), np.float32)
    e_i = 0
    for sent_i, _ in enumerate(inputs['input_ids']):
        len_left = 1
        for length in lengths[sent_i]:
            embeddings[e_i] = hidden_state[sent_i][len_left]
            len_left += length
            e_i += 1
    return embeddings
