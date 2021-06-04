import multiprocessing
import os
import re
from glob import glob

regex_ws = re.compile(r'\s+')


def extract(path):
    with open(path, 'r', encoding='latin-1') as fh:
        text = fh.read()
    title = text.split('\n\n')[0]
    corpus, topic, id = path.split('/')[-3:]
    text = regex_ws.sub(r' ', text.strip()[len(title) + 2:])
    return {
        'Meta': {
            'title': title,
            'topic': topic,
        },
        'Text': text,
        'Corpus': corpus,
        'DocID': corpus + '-{}-{}'.format(topic, id[:-len('.txt')])
    }


def get_data(path):
    news_files = glob(os.path.join(path, 'bbc/*/*.txt'))

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        news_json = pool.map(extract, news_files, chunksize=8)

    return news_json


def get_sports_data(path):
    news_files = glob(os.path.join(path, 'bbcsport/*/*.txt'))

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        news_json = pool.map(extract, news_files, chunksize=8)

    return news_json
