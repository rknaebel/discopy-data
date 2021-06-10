import csv
import zipfile
from io import TextIOWrapper

from tqdm import tqdm


def extract(source_path: str):
    with zipfile.ZipFile(source_path) as zh:
        with zh.open(zh.filelist[0].filename, 'r') as fh:
            reader = csv.reader(TextIOWrapper(fh, 'utf-8'))
            for row_i, row in tqdm(enumerate(reader)):
                if row_i == 0:
                    continue
                yield {
                    'meta': {
                        'title': row[0],
                        'corpus': 'short-stories',
                    },
                    'text': row[1],
                }
