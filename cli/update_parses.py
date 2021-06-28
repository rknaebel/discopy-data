import os

import click
import ujson as json
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = ''

from discopy_data.data.doc import Document
from discopy_data.data.update import update_dataset_parses


@click.command()
@click.option('-s', '--src', default='-', type=click.File('r'))
@click.option('-o', '--tgt', default='-', type=click.File('w'))
@click.option('-c', '--constituency', default='crf-con-en', type=str)
@click.option('-d', '--dependency', default='biaffine-dep-en', type=str)
def main(src, tgt, constituency, dependency):
    docs = [Document.from_json(json.loads(line)) for line in src if line.strip()]
    update_dataset_parses(docs, constituent_parser=constituency, dependency_parser=dependency)
    for doc in tqdm(docs):
        tgt.write(json.dumps(doc.to_json()) + '\n')


if __name__ == '__main__':
    main()
