"""Cached dense BGE-M3, batched cloud or local. Never truncates source."""
import hashlib
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import httpx
import numpy as np
from .common import ROOT, digest, event, check_time, device


class Embedder:
    def __init__(self, cache, config, deadline):
        from tokenizers import Tokenizer
        self.config, self.deadline = config, deadline
        self.root = Path(cache); self.root.mkdir(parents=True, exist_ok=True)
        self.role = 'document'
        self.local = config['bge_provider'] == 'local'
        self.model_dir = Path(config.get('bge_model_dir') or ROOT/'models/bge-m3')
        self.tokenizer = Tokenizer.from_file(str(ROOT/'vendor/pipeline/models/bge-m3/tokenizer.json'))
        self.tokenizer.no_truncation(); self.tokenizer.no_padding()
        self.manifest = {'revision': 'remote-unknown', 'device': 'local' if self.local else 'siliconflow'}
        self.binding = digest(['bge-m3-dense-cls-v1', config['bge_provider'],
                               self._weight_identity() if self.local else 'https://api.siliconflow.cn/v1/embeddings'])
        self.db = sqlite3.connect(self.root/'vectors.sqlite3')
        self.db.execute('CREATE TABLE IF NOT EXISTS vectors (binding TEXT, key TEXT, value BLOB, sha TEXT, PRIMARY KEY(binding,key))')
        self.model = None
        self.client = None if self.local else httpx.Client(follow_redirects=False,
            limits=httpx.Limits(max_connections=2,max_keepalive_connections=2,keepalive_expiry=60))

    def _weight_identity(self):
        from pipeline import verify_model
        return verify_model(self.model_dir)

    def close(self):
        self.db.close()
        if self.client is not None:self.client.close()

    def encode(self, texts):
        texts = list(texts)
        if any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError('empty_embedding_input')
        lengths = [len(self.tokenizer.encode(t).ids) for t in texts]
        if any(n > 8192 for n in lengths):
            raise ValueError('BGE input over 8192: no truncation')
        found, missing = {}, {}
        for t in texts:
            key = digest(t)
            if key in found or key in missing: continue
            row = self.db.execute('SELECT value,sha FROM vectors WHERE binding=? AND key=?', (self.binding, key)).fetchone()
            if row:
                if hashlib.sha256(row[0]).hexdigest() != row[1]: raise ValueError('vector_checksum')
                v = np.frombuffer(row[0], dtype='<f4').copy()
                self._validate(v); found[key] = v
            else: missing[key] = t
        check_time(self.deadline)
        groups = [list(missing.items())[i:i+8] for i in range(0, len(missing), 8)]
        begin = time.monotonic()
        if self.local:
            batches = [self._local(g) for g in groups]
        else:
            with ThreadPoolExecutor(max_workers=2) as pool:
                batches = list(pool.map(self._remote, groups))
        with self.db:
            for group, vectors in zip(groups, batches):
                for (key, _), v in zip(group, vectors):
                    self._validate(v)
                    blob = v.astype('<f4').tobytes()
                    self.db.execute('INSERT OR REPLACE INTO vectors VALUES (?,?,?,?)',
                                    (self.binding, key, blob, hashlib.sha256(blob).hexdigest()))
                    found[key] = v
        event(self.root/'embedding-events.jsonl', {'role': self.role, 'texts': len(texts),
              'unique_texts': len(found), 'cache_hits': len(found)-len(missing), 'encoded': len(missing),
              'seconds': time.monotonic()-begin})
        return np.asarray([found[digest(t)] for t in texts], dtype=np.float32), lengths

    @staticmethod
    def _validate(v):
        if v.shape != (1024,) or not np.isfinite(v).all() or abs(float(np.linalg.norm(v))-1) > 1e-4:
            raise ValueError('invalid_bge_vector')

    def _remote(self, group):
        texts=[t for _,t in group]
        details={'role':self.role,'texts':len(group),'input_hashes':[k for k,_ in group],
                 'characters':[len(t) for t in texts],
                 'bge_tokens':[len(self.tokenizer.encode(t).ids) for t in texts]}
        for attempt in range(3):
            check_time(self.deadline)
            start=time.monotonic()
            try:
                r = self.client.post('https://api.siliconflow.cn/v1/embeddings',
                    headers={'Authorization': 'Bearer '+self.config['bge_key']},
                    json={'model': 'BAAI/bge-m3', 'input': texts, 'encoding_format':'float'},
                    timeout=min(60,max(.001,self.deadline-time.time())))
            except (httpx.ConnectError,httpx.ReadError,httpx.RemoteProtocolError) as exc:
                event(self.root/'embedding-http.jsonl',dict(details,attempt=attempt+1,
                      error=type(exc).__name__,seconds=time.monotonic()-start))
                if attempt==2:raise
                delay=2**attempt
                if self.deadline-time.time()<=delay:raise TimeoutError('task_deadline_before_retry') from exc
                time.sleep(delay);continue
            record=dict(details,attempt=attempt+1,status=r.status_code,seconds=time.monotonic()-start)
            if r.status_code!=200:
                # Error responses only: omit headers and redact the configured secret.
                record['error_body']=r.text.replace(self.config['bge_key'],'[REDACTED]')[:2000]
            event(self.root/'embedding-http.jsonl',record)
            if r.status_code in (429,503,504) and attempt<2:
                time.sleep(min(2**attempt, max(0,self.deadline-time.time()))); continue
            if r.status_code != 200: raise RuntimeError('embedding_http_'+str(r.status_code))
            rows=r.json().get('data',[])
            if sorted(x.get('index',-1) for x in rows)!=list(range(len(group))): raise ValueError('embedding_batch_indices')
            result=[]
            for row in sorted(rows,key=lambda x:x['index']):
                v=np.asarray(row['embedding'],dtype=np.float32); norm=np.linalg.norm(v)
                if not np.isfinite(norm) or norm<=0: raise ValueError('embedding_norm')
                result.append(v/norm)
            return result
        raise RuntimeError('embedding_retries_exhausted')

    def _local(self, group):
        import torch
        if self.model is None:
            from transformers import AutoTokenizer, AutoModel
            self.local_tokenizer=AutoTokenizer.from_pretrained(self.model_dir,local_files_only=True)
            self.model=AutoModel.from_pretrained(self.model_dir,local_files_only=True).to(device(self.config.get('device','auto'))).eval()
        result=[]
        for _,t in group:
            encoded=self.local_tokenizer(t,return_tensors='pt',truncation=False).to(self.model.device)
            if encoded['input_ids'].shape[1]>8192:raise ValueError('local_bge_token_limit')
            with torch.inference_mode():
                v=self.model(**encoded).last_hidden_state[:,0]
                v=torch.nn.functional.normalize(v,dim=1).float().cpu().numpy()[0]
            result.append(v)
        return result
