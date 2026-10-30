"""Download the fixed official BGE-M3 dense checkpoint and verify its files."""
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import time
import shutil

REPO = 'BAAI/bge-m3'
REVISION = '5617a9f61b028005a4858fdac845db406aefb181'
MODEL_DIR = Path(__file__).resolve().parent / 'models' / 'bge-m3'
FILES = ['config.json', 'tokenizer_config.json', 'special_tokens_map.json',
         'tokenizer.json', 'sentencepiece.bpe.model', 'README.md', 'pytorch_model.bin']


def hashes(path):
    sha = hashlib.sha256()
    git = hashlib.sha1(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            sha.update(block)
            git.update(block)
    return sha.hexdigest(), git.hexdigest()


def download_large(name, size, target):
    """Use verified byte ranges; a server ignoring Range fails before body read."""
    pieces = MODEL_DIR / '.download-parts'
    pieces.mkdir(exist_ok=True)
    count = 16
    step = (size + count - 1) // count

    def fetch(index):
        lo, hi = index * step, min(size, (index + 1) * step) - 1
        path = pieces / f'{name}.{index:02d}'
        if path.exists() and path.stat().st_size == hi - lo + 1:
            return path
        for attempt in range(4):
            try:
                url = f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}?download=true&segment={index}'
                request = urllib.request.Request(url, headers={'Range': f'bytes={lo}-{hi}'})
                with urllib.request.urlopen(request, timeout=120) as response:
                    if response.status != 206 or response.headers.get('Content-Range') != f'bytes {lo}-{hi}/{size}':
                        raise ValueError('Server did not honor the requested byte range')
                    with path.open('wb') as stream:
                        while block := response.read(1024 * 1024):
                            stream.write(block)
                if path.stat().st_size != hi - lo + 1:
                    raise ValueError('Incomplete download range')
                print(f'Weight segment {index + 1}/{count} ready', flush=True)
                return path
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(3 * (attempt + 1))

    with ThreadPoolExecutor(max_workers=count) as pool:
        paths = list(pool.map(fetch, range(count)))
    partial = target.with_name(target.name + '.assembled')
    with partial.open('wb') as stream:
        for path in paths:
            with path.open('rb') as source:
                while block := source.read(8 * 1024 * 1024):
                    stream.write(block)
    partial.rename(target)
    # Keep pieces until the caller verifies the official whole-file hash.


def main():
    url = f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true'
    with urllib.request.urlopen(url, timeout=60) as response:
        metadata = json.load(response)
    assert metadata['sha'] == REVISION
    remote = {f['rfilename']: f for f in metadata['siblings']}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {'repo': REPO, 'revision': REVISION, 'representation': 'dense CLS + L2',
                'license': 'MIT (official model card)', 'files': {}}
    for name in FILES:
        info = remote[name]
        target = MODEL_DIR / name
        cache = Path.home() / '.cache/huggingface/hub/models--BAAI--bge-m3/blobs' / info.get('lfs', {}).get('sha256', info['blobId'])
        if not target.exists() and cache.is_file() and cache.stat().st_size == info['size']:
            print(f'Copying matching local cache for {name}; verifying below', flush=True)
            temporary = target.with_name(target.name + '.copying')
            shutil.copyfile(cache, temporary)
            temporary.rename(target)
        if not target.exists():
            partial = target.with_name(target.name + '.part')
            print(f'Downloading {name}: {info["size"]:,} bytes', flush=True)
            if info['size'] > 100 * 1024 * 1024:
                download_large(name, info['size'], target)
            else:
                subprocess.run(['curl', '--fail', '--location', '--retry', '4', '--retry-delay', '3',
                                '--connect-timeout', '30', '--continue-at', '-', '--output', str(partial),
                                f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}?download=true'], check=True)
                partial.rename(target)
        sha, git = hashes(target)
        expected = info.get('lfs', {}).get('sha256')
        if target.stat().st_size != info['size'] or (sha != expected if expected else git != info['blobId']):
            raise ValueError(f'Integrity check failed: {target}; remove invalid file before retrying')
        manifest['files'][name] = {'bytes': target.stat().st_size, 'sha256': sha}
        print(f'Verified {name}: {sha}', flush=True)
        for leftover in (MODEL_DIR / '.download-parts').glob(f'{name}.*'):
            leftover.unlink()
        target.with_name(target.name + '.part').unlink(missing_ok=True)
    (MODEL_DIR / 'MODEL_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Model ready: {MODEL_DIR}', flush=True)


if __name__ == '__main__':
    main()
