import sys

import click
import ujson as json

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
    update_annotations = document_annotations[corpus](annotations)
    for line in src:
        if not line.strip():
            continue
        doc = Document.from_json(json.loads(line))
        doc = update_annotations(doc)
        tgt.write(json.dumps(doc.to_json()) + '\n')
        tgt.flush()
    sys.stderr.write('Annotation done!\n')


if __name__ == '__main__':
    main()
