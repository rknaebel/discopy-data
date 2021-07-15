import sys


def extract_raw(source_path: str):
    if source_path == '-':
        fh = sys.stdin
    else:
        fh = open(source_path, 'r')
    for line_i, line in enumerate(fh):
        content = line
        yield {
            'meta': {
                'fileID': f'raw_{line_i}',
                'corpus': 'raw',
            },
            'text': content,
        }
