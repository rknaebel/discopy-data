import multiprocessing
import os
import re
from glob import glob

regex_all = re.compile(r'<URL>(.+)<\/URL>\s*<TITLE>(.+)<\/TITLE>(.*)<AUTHOR\s*name=\"(.*)\">', flags=re.DOTALL)
regex_p = re.compile(r'\s*\<P\>\s*', flags=re.DOTALL)
regex_qs = re.compile(r'\<QUOTE\>\s*', flags=re.DOTALL)
regex_qe = re.compile(r'\s*\<\/QUOTE\>', flags=re.DOTALL)
regex_headline = re.compile(r'\<HEADLINE.*\>.*\<\/HEADLINE\>', flags=re.DOTALL)
regex_photo = re.compile(r'Photo.*\n')
regex_ws = re.compile(r'\s+')


def extract(news_file):
    with open(news_file, 'r') as fh:
        raw_text = fh.read()
    try:
        result = regex_all.search(raw_text)

        url, title, text, author = result.groups()
        text = regex_p.sub('\n', text)
        text = regex_qs.sub('*', text)
        text = regex_qe.sub('*', text)
        text = regex_headline.sub('', text)
        text = regex_photo.sub('', text)
        text = regex_ws.sub(r' ', text)
        return {
            'Meta': {
                'url': url,
                'title': title.strip(),
                'author:': author.strip(),
                'raw': raw_text.strip()
            },
            'Text': text.strip(),
            'Corpus': 'global_voices',
            'DocID': 'gv-{}'.format(url[len('https://globalvoices.org/'):])
        }
    except AttributeError:
        return None


def get_data(path):
    news_files = glob(os.path.join(path, 'global_voices/split/en/*'))

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        news_json = pool.map(extract, news_files, chunksize=8)

    return list(filter(lambda x: x is not None, news_json))
