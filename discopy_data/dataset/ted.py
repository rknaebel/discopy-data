import multiprocessing
import os
import re

import pandas as pd

regex_1 = re.compile(r'(\s*\([^)]*\)\s*)+')
regex_2 = re.compile(r'([.?!])([A-Z])')
regex_3 = re.compile(r'\s+')


def extract(t, meta):
    text = t.transcript.strip()
    text = regex_1.sub(' ', text)
    text = regex_2.sub(r'\1 \2', text)
    text = regex_3.sub(r' ', text)

    return {
        'Meta': {
            'url': t.url.strip(),
            'transcript': t.transcript.strip(),
            'description': meta.description,
            'title': meta.title,
            'tags': meta.tags,
            'main_speaker': meta.main_speaker,
        },
        'Text': text,
        'Corpus': 'ted',
        'DocID': 'ted-{}'.format(t.url[len('https://www.ted.com/talks/'):].strip())
    }


def get_data(path):
    data_path = os.path.join(path, 'ted/transcripts.csv')
    meta_path = os.path.join(path, 'ted/ted_main.csv')

    ted_corpus = pd.read_csv(data_path)
    ted_meta = pd.read_csv(meta_path).set_index('url')

    ted_json = []
    for i, t in ted_corpus.iterrows():
        ted_json.append((t, ted_meta.loc[t.url]))

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        ted_json = pool.starmap(extract, ted_json, chunksize=8)

    return ted_json
