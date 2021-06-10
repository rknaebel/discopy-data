import datetime

import click
import ujson as json
from tqdm import tqdm

import discopy_data.dataset.anthology
import discopy_data.dataset.argessay
import discopy_data.dataset.bbc
import discopy_data.dataset.global_voices
import discopy_data.dataset.press_gov
import discopy_data.dataset.short_stories
import discopy_data.dataset.ted
import discopy_data.dataset.un_debates

document_extractor = {
    'anthology': discopy_data.dataset.anthology.extract,
    'args-essay': discopy_data.dataset.argessay.extract,
    'bbc': discopy_data.dataset.bbc.extract,
    'gv': discopy_data.dataset.global_voices.extract,
    'press-gov': discopy_data.dataset.press_gov.extract,
    'short-stories': discopy_data.dataset.short_stories.extract,
    'ted': discopy_data.dataset.ted.extract,
    'un-debates': discopy_data.dataset.un_debates.extract,
}


@click.command()
@click.argument('corpus', type=str)
@click.argument('src', type=click.Path('r'))
@click.option('-o', '--tgt', default='-', type=click.File('w'))
@click.option('-l', '--limit', default=0, type=int)
def main(corpus, src, tgt, limit):
    t = tqdm()
    doc_i = 0
    for doc in document_extractor[corpus](src):
        if limit and doc_i >= limit:
            break
        doc['docID'] = f"{doc['meta']['corpus']}_{doc_i:05}"
        doc['meta']['date'] = datetime.datetime.now().isoformat()
        tgt.write(json.dumps(doc) + '\n')
        doc_i += 1
        t.update(1)
    t.close()


if __name__ == '__main__':
    main()
