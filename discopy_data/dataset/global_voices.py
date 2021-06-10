import os
import re
import zipfile

regex_all = re.compile(r'<URL>(.+)<\/URL>\s*<TITLE>(.+)<\/TITLE>(.*)<AUTHOR\s*name=\"(.*)\">', flags=re.DOTALL)
regex_p = re.compile(r'\s*\<P\>\s*', flags=re.DOTALL)
regex_qs = re.compile(r'\<QUOTE\>\s*', flags=re.DOTALL)
regex_qe = re.compile(r'\s*\<\/QUOTE\>', flags=re.DOTALL)
regex_headline = re.compile(r'\<HEADLINE.*\>.*\<\/HEADLINE\>', flags=re.DOTALL)
regex_photo = re.compile(r'Photo.*\n')
regex_ws = re.compile(r'\s+')


def extract(source_path: str):
    with zipfile.ZipFile(source_path) as zh:
        for fn in zh.filelist:
            if not os.path.split(fn.filename)[1]:
                continue
            raw_text = zh.open(fn).read().decode()
            try:
                result = regex_all.search(raw_text)
                url, title, text, author = result.groups()
                text = regex_p.sub('\n', text)
                text = regex_qs.sub('*', text)
                text = regex_qe.sub('*', text)
                text = regex_headline.sub('', text)
                text = regex_photo.sub('', text)
                text = regex_ws.sub(r' ', text)
                yield {
                    'meta': {
                        'url': url,
                        'title': title.strip(),
                        'author:': author.strip(),
                        'corpus': 'global_voices',
                    },
                    'text': text.strip(),
                }
            except AttributeError:
                continue
