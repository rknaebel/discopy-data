import json
import os
import sys

import click

from discopy_data.data.doc import Document
from discopy_data.data.update import get_constituent_parse, get_dependency_parse


@click.command()
@click.option('-s', '--src', default='-', type=click.File('r'))
@click.option('-o', '--tgt', default='-', type=click.File('w'))
@click.option('--constituent-parser', default='crf-con-en', type=str)
@click.option('--dependency-parser', default='biaffine-dep-en', type=str)
@click.option('-c', '--constituents', is_flag=True)
@click.option('-d', '--dependencies', is_flag=True)
@click.option('--cuda', default='', type=str)
def main(src, tgt, constituent_parser, dependency_parser, constituents, dependencies, cuda):
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda
    import supar
    sys.stderr.write('SUPAR load constiuent parser!\n')
    cparser = supar.Parser.load(constituent_parser) if constituents else None
    sys.stderr.write('SUPAR load dependency parser!\n')
    dparser = supar.Parser.load(dependency_parser) if dependencies else None
    for line in src:
        doc = Document.from_json(json.loads(line))
        for sent_i, sent in enumerate(doc.sentences):
            inputs = [(t.surface, t.upos) for t in sent.tokens]
            if cparser:
                parsetree = get_constituent_parse(cparser, inputs)
                doc.sentences[sent_i].parsetree = parsetree
            if dparser:
                dependencies = get_dependency_parse(dparser, inputs, sent.tokens)
                doc.sentences[sent_i].dependencies = dependencies
        tgt.write(json.dumps(doc.to_json()) + '\n')
        tgt.flush()
    sys.stderr.write('Supar parsing done!\n')


if __name__ == '__main__':
    main()
