import os
import re
import sys
import zipfile

LEADING_LINE_NB = re.compile(r'^(\d+[.:]\s+)')


def process_document(txt):
    lines = [line.strip() for line in txt.splitlines() if re.search(r'[a-zA-Z]', line)]
    return " ".join(map(lambda l: re.sub(LEADING_LINE_NB, '', l), lines))


def extract(source_path: str):
    with zipfile.ZipFile(source_path) as zh:
        for zf in zh.filelist:
            if not zf.filename.endswith('.txt') or zf.filename.startswith('_'):
                continue
            try:
                file_id = os.path.basename(zf.filename)[:-len('.txt')]
                content = process_document(zh.open(zf).read().decode())
                yield {
                    'meta': {
                        'fileID': file_id,
                        'year': int(file_id[-4:]),
                        'corpus': 'un-debates',
                    },
                    'text': content,
                }
            except UnicodeDecodeError:
                print("Unicode error in file:", zf.filename, file=sys.stderr)
                continue
