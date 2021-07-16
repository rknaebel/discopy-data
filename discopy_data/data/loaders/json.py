import json
from typing import List, TextIO

from discopy_data.data.doc import Document


def load_documents(fh: TextIO, tags: str = '') -> List[Document]:
    if tags:
        tags = tags.split(',')
    docs = []
    try:
        for line in fh:
            doc = Document.from_json(json.loads(line))
            if not tags or doc.meta.get('part', '') in tags:
                docs.append(doc)
    except EOFError:
        pass
    return docs
