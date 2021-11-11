import json
import re

import click

from discopy_data.data.loaders.raw import load_texts, load_texts_fast


@click.command()
@click.option('-i', '--src', default='-', type=click.File('r'))
@click.option('-o', '--tgt', default='-', type=click.File('w'))
@click.option('-t', '--tokenize-only', is_flag=True)
@click.option('-f', '--fast', is_flag=True)
def main(src, tgt, tokenize_only, fast):
    document_loader = load_texts_fast if fast else load_texts
    for doc in document_loader(re.split(r'\n\n\n+', src.read()), tokenize_only=tokenize_only):
        tgt.write(json.dumps(doc.to_json()) + '\n')


if __name__ == '__main__':
    main()
