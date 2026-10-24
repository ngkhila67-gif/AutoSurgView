"""Resolve native manifests relative to their location and verify binary bytes."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys

_DLL_DIRECTORIES = []


def manifest_path(folder, value):
    path = Path(value)
    return path if path.is_absolute() else Path(folder) / path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def library_record(metadata, stem):
    if os.name == 'nt':
        platform, suffix = 'windows', '.dll'
    elif sys.platform.startswith('linux'):
        platform, suffix = 'linux', '.so'
    else:
        raise RuntimeError('The supplied native backends support Windows x64 and Linux x86-64.')
    if 'libraries' in metadata:
        if platform not in metadata['libraries']:
            raise RuntimeError(f'No {platform} native binary in this manifest; build it with native/build_native.py.')
        return metadata['libraries'][platform]
    return {'file': stem + suffix, 'sha256': metadata['dll_sha256']}


def load_library(folder, manifest_name, stem):
    folder = Path(folder).resolve()
    description = folder / manifest_name
    if not description.is_file():
        raise RuntimeError(f'Native manifest missing: {description}. Run simulation/native/build_native.py.')
    metadata = json.loads(description.read_text(encoding='utf-8'))
    record = library_record(metadata, stem)
    binary = manifest_path(folder, record['file'])
    if not binary.is_file() or digest(binary) != record['sha256']:
        raise RuntimeError(f'Native binary missing or checksum mismatch: {binary}')
    if os.name == 'nt':
        runtime = manifest_path(folder, metadata['runtime_dll_directory']).resolve()
        if not runtime.is_dir():
            raise RuntimeError(f'Native runtime directory missing: {runtime}')
        for name, expected in metadata.get('runtime_files', {}).items():
            path = runtime / name
            if not path.is_file() or digest(path) != expected:
                raise RuntimeError(f'Native runtime missing or checksum mismatch: {path}')
        _DLL_DIRECTORIES.append(os.add_dll_directory(str(runtime)))
    try:
        library = ctypes.CDLL(str(binary))
    except OSError as error:
        raise RuntimeError(f'Cannot load {binary.name}. Install its platform runtime dependencies or rebuild locally: {error}') from error
    return library, metadata, description
