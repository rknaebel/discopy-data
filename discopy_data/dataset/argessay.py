import zipfile

from tqdm import tqdm

from discopy_data.data.doc import Document
from discopy_data.data.relation import Relation


def extract_arguments(annos, text, doc_id):
    args = {}
    for a in annos:
        if a[0][0] == 'T':
            args[a[0]] = {
                'type': a[1].split(" ")[0],
                'offset': text.find(a[2]),
                'length': len(a[2]),
            }
    for a in annos:
        if a[0][0] == 'A':
            arg_type, arg_id, arg_stance = a[1].split(' ')
            args[arg_id]['stance'] = arg_stance
    arguments = []
    for r in annos:
        if r[0][0] == 'R':
            rtype, arg1, arg2 = r[1].split(' ')
            arg1 = args[arg1.split(':')[1]]
            arg2 = args[arg2.split(':')[1]]
            arguments.append({
                'Sense': [rtype],
                'ID': len(arguments),
                'Arg1': arg1,
                'Type': 'Argumentation',
                'DocID': doc_id,
                'Arg2': arg2
            })
    return arguments


def extract(source_path: str):
    with zipfile.ZipFile(source_path) as zh_all:
        brat_file = [f for f in zh_all.filelist if f.filename.endswith('brat-project-final.zip')][0]
        with zipfile.ZipFile(zh_all.open(brat_file)) as zh_brat:
            annotation_files = sorted(filter(lambda f: not f.startswith('_') and f.endswith('.txt'),
                                             (f.filename for f in zh_brat.filelist)))
            for doc_i, path in tqdm(enumerate(annotation_files)):
                content = zh_brat.open(path).read().decode().splitlines(keepends=True)
                yield {
                    'meta': {
                        'title': content[0].strip(),
                        'corpus': 'argumentative_essays',
                        'path': path,
                    },
                    'text': '\n'.join(p.strip() for p in content[2:]),
                }


def update_annotations(source_path, doc: Document):
    with zipfile.ZipFile(source_path) as zh_all:
        brat_file = [f for f in zh_all.filelist if f.filename.endswith('brat-project-final.zip')][0]
        with zipfile.ZipFile(zh_all.open(brat_file)) as zh_brat:
            content = zh_brat.open(doc.meta['path'][:-3] + 'ann').read().decode().splitlines(keepends=True)
            annos = [tuple(a.strip().split("\t")) for a in content]
            arguments = extract_arguments(annos, doc.text, doc.doc_id)
            words = doc.get_tokens()
            relations = [
                Relation([t for t in words if
                          arg['Arg1']['offset'] <= t.offset_begin <= (arg['Arg1']['offset'] + arg['Arg1']['length'])],
                         [t for t in words if
                          arg['Arg2']['offset'] <= t.offset_begin <= (arg['Arg2']['offset'] + arg['Arg2']['length'])],
                         [],
                         arg['Sense'], 'Argumentation') for arg in arguments
            ]
            doc.relations = relations
