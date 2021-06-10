import re
import zipfile

import pandas as pd

regex_1 = re.compile(r'(\s*\([^)]*\)\s*)+')
regex_2 = re.compile(r'([.?!])([A-Z])')
regex_3 = re.compile(r'\s+')


def extract_item(t, meta):
    text = t.transcript.strip()
    text = regex_1.sub(' ', text)
    text = regex_2.sub(r'\1 \2', text)
    text = regex_3.sub(r' ', text)

    return {
        'meta': {
            'url': t.url.strip(),
            # 'transcript': t.transcript.strip(),
            'description': meta.description,
            'title': meta.title,
            'tags': meta.tags,
            'main_speaker': meta.main_speaker,
            'corpus': 'ted',
        },
        'text': text,
    }


def extract(source_path: str):
    with zipfile.ZipFile(source_path) as zh:
        data_path = zh.open('ted/transcripts.csv')
        meta_path = zh.open('ted/ted_main.csv')

        ted_corpus = pd.read_csv(data_path)
        ted_meta = pd.read_csv(meta_path).set_index('url')

        for i, t in ted_corpus.iterrows():
            yield extract_item(t, ted_meta.loc[t.url])
