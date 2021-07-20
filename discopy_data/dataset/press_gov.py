import json
import re
import zipfile

from tqdm import tqdm

WS = re.compile(r'[\s\xa0]+')


def clean_text(s):
    return re.sub(WS, ' ', s)


def extract(source_path: str):
    doc_i = 0
    with zipfile.ZipFile(source_path, 'r') as zh:
        with zh.open(zh.filelist[0].filename, 'r') as fh:
            for row_i, row in tqdm(enumerate(fh.read().decode().splitlines())):
                if row_i == 0:
                    continue
                doc = json.loads(row)
                contents = re.sub(WS, ' ', doc['contents'])
                yield {
                    'meta': {
                        'title': re.sub(WS, ' ', doc['title']),
                        'corpus': 'press-gov',
                        'published': doc['date'],
                        'components': doc['components']
                    },
                    'text': contents,
                }
                doc_i += 1
