import datetime
import os
import random
import re
import sys
import zipfile

import click
import spacy
import ujson as json
from tqdm import tqdm

from discopy_data.data.loaders.raw import load_texts

LEADING_LINE_NB = re.compile(r'^(\d+[.:]\s+)')


def process_document(txt):
    lines = [line.strip() for line in txt.splitlines() if re.search(r'[a-zA-Z]', line)]
    return " ".join(map(lambda l: re.sub(LEADING_LINE_NB, '', l), lines))


@click.command()
@click.option('-i', '--src', default='-', type=click.Path('r'))
@click.option('-o', '--tgt', default='-', type=click.File('w'))
@click.option('-l', '--limit', default=0, type=int)
def main(src, tgt, limit):
    nlp = spacy.load('en')
    doc_i = 0
    with zipfile.ZipFile(src, 'r') as zh:
        for zf in tqdm(random.sample(zh.filelist, len(zh.filelist))):
            if not zf.filename.endswith('.txt'):
                continue
            if limit and doc_i >= limit:
                break
            try:
                file_id = os.path.basename(zf.filename)[:-len('.txt')]
                content = process_document(zh.open(zf).read().decode())
                parses = load_texts(texts=[content], nlp=nlp)[0]
                if len(parses.sentences) <= 2:
                    continue
                parses = parses.to_json()
                doc = {
                    'docID': f"un-debates_{doc_i:05}",
                    'meta': {
                        'fileID': file_id,
                        'year': int(file_id[-4:]),
                        'corpus': 'un-debates',
                        'date': datetime.datetime.now().isoformat(),
                    },
                    'text': parses['text'],
                    'sentences': parses['sentences'],
                }
                tgt.write(json.dumps(doc) + '\n')
                doc_i += 1
            except UnicodeDecodeError:
                print("Unicode error in file:", zf.filename, file=sys.stderr)
                continue


if __name__ == '__main__':
    main()
