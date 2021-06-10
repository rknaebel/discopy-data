import re
import zipfile

regex_ws = re.compile(r'\s+')


def extract(source_path: str):
    with zipfile.ZipFile(source_path) as zh:
        for fn in zh.filelist:
            if not fn.filename.endswith('.txt'):
                continue
            text = zh.open(fn).read().decode('latin-1')
            title = text.split('\n\n')[0]
            corpus, topic, id = fn.filename.split('/')[-3:]
            yield {
                'meta': {
                    'title': title,
                    'topic': topic,
                    'corpus': corpus,
                },
                'text': regex_ws.sub(r' ', text.strip()[len(title) + 2:]),
            }
