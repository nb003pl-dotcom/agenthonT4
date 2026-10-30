import hashlib
import tomllib
from pathlib import Path
from .common import read, digest


def relations(manifest, docs, roster):
    result = {}
    for entry in manifest['files']:
        if entry.get('role') != 'corpus':
            continue
        did = Path(entry['path']).stem
        if did not in docs:
            raise ValueError('manifest_doc_missing: ' + did)
        ids = entry.get('entity_ids', [])
        shared = entry.get('shared', False)
        if type(shared) is not bool or not isinstance(ids, list) or any(x not in roster for x in ids):
            raise ValueError('invalid_official_relation: ' + did)
        result[did] = {'entity_ids': sorted(set(ids)), 'shared': shared,
                       'scope': 'shared' if shared else ('entity' if ids else 'unknown'),
                       'source': 'official_root_manifest'}
    return result


class Inputs:
    def __init__(self, path):
        from preprocess_unit import load_unit
        self.path = Path(path).resolve()
        self.task, self.docs, self.identity = load_unit(self.path)
        self.card = tomllib.loads((self.path / 'card.toml').read_text())
        self.entities = {e['entity_id']: e for e in self.task['entities']}
        if len(self.entities) != len(self.task['entities']) or not self.entities:
            raise ValueError('duplicate_or_empty_entities')
        if self.task['target']['type'] not in ('regression', 'classification', 'ranking'):
            raise ValueError('unknown_target')
        self.relations = relations(read(self.path / 'manifest.json'), self.docs, self.entities)
        self.binding = digest([self.identity, self.relations])

    def allowed(self, entity, doc):
        r = self.relations.get(doc, {})
        return entity in self.entities and (r.get('shared') is True or entity in r.get('entity_ids', []))

    def pools(self, entity):
        return {
            'entity': [d for d, r in self.relations.items() if not r['shared'] and entity in r['entity_ids']],
            'shared': [d for d, r in self.relations.items() if r['shared']],
        }

    def catalog(self):
        return [{'doc_id': d, 'title': v.get('title', ''), 'doc_date': v['doc_date'],
                 **self.relations[d]} for d, v in self.docs.items()]
