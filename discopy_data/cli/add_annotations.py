import json
import sys

import click

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
@click.option('--simple-connectives', is_flag=True)
@click.option('--sense-level', default=-1, type=int)
def main(corpus, annotations, src, tgt, simple_connectives, sense_level):
    options = {
        'simple_connectives': simple_connectives,
        'sense_level': sense_level,
    }
    update_annotations = document_annotations[corpus](annotations, options)
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
