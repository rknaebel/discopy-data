import click
import ujson as json
from tqdm import tqdm

import discopy_data.dataset.argessay
import discopy_data.dataset.pdtb
from discopy_data.data.doc import Document

document_annotations = {
    'args-essay': discopy_data.dataset.argessay.update_annotations,
    'pdtb': discopy_data.dataset.pdtb.update_annotations,
}


@click.command()
@click.argument('corpus', type=str)
@click.argument('annotations', type=str)
@click.option('-s', '--src', default='-', type=click.File('r'))
@click.option('-o', '--tgt', default='-', type=click.File('w'))
def main(corpus, annotations, src, tgt):
    t = tqdm()
    update_annotations = document_annotations[corpus]
    docs = [Document.from_json(json.loads(line)) for line in src if line.strip()]
    for doc in update_annotations(annotations, docs):
        tgt.write(json.dumps(doc.to_json()) + '\n')
        t.update(1)
    t.close()


if __name__ == '__main__':
    main()
