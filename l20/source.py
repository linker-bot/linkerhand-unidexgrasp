"""Locations and integrity checks for the read-only upstream submodule."""
from pathlib import Path
import hashlib
import json
import subprocess

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).resolve().parent / 'assets'
SOURCE_ROOT = REPOSITORY_ROOT / 'third_party/linkerhand-urdf'
SOURCE_MODEL = SOURCE_ROOT / 'L20/right'
OFFICIAL_URDF = SOURCE_MODEL / 'linkerhand_l20_right.urdf'
RUNTIME_ASSETS = ASSETS / 'l20_right'
SOURCE_MANIFEST = RUNTIME_ASSETS / 'manifest.json'
URDF_COMMIT = '49f26021b0a91ff5c33150df22970c07bb548672'
INITIALIZE_COMMAND = 'git submodule update --init --recursive -- third_party/linkerhand-urdf'

def verify_official_source(root=SOURCE_ROOT, manifest_path=SOURCE_MANIFEST,
                           check_revision=False):
    root = Path(root).resolve()
    model_directory = root / 'L20/right'
    if not (model_directory/'linkerhand_l20_right.urdf').is_file():
        raise RuntimeError('Official L20 source submodule is missing. From the repository root run: '+INITIALIZE_COMMAND)
    if check_revision:
        try:
            top = subprocess.check_output(['git','-C',str(root),'rev-parse','--show-toplevel'],text=True,stderr=subprocess.DEVNULL).strip()
            revision = subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
        except (OSError,subprocess.CalledProcessError) as exc:
            raise RuntimeError('Cannot verify the official source submodule. Run: '+INITIALIZE_COMMAND) from exc
        if Path(top).resolve()!=root or revision!=URDF_COMMIT:
            raise RuntimeError('Official source submodule must be at '+URDF_COMMIT+'. Run: '+INITIALIZE_COMMAND)
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get('commit')!=URDF_COMMIT:
        raise ValueError('Source asset manifest commit mismatch')
    for record in manifest['files']:
        path = model_directory/record['path']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:
            raise ValueError('Official source asset checksum mismatch: '+str(path))
    return manifest
