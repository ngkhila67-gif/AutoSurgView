"""Refresh or verify hashes for tracked and non-ignored repository files."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'PACKAGE MANIFEST.json'

def file_names(metadata,check):
    if not (ROOT/'.git').exists():
        if not check:
            raise RuntimeError('Refreshing the file inventory requires a Git checkout. Download ZIP supports --check.')
        return sorted(name for name in metadata['files'] if (ROOT/name).is_file())
    command=['git','-C',str(ROOT),'ls-files','--cached','--others','--exclude-standard','-z']
    names=set(subprocess.check_output(command).decode('utf-8').split('\0'))-{''}
    names.discard(MANIFEST.name)
    return sorted(name for name in names if (ROOT/name).is_file())

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true',help='Validate without modifying the manifest')
    args=parser.parse_args()
    metadata=json.loads(MANIFEST.read_text(encoding='utf-8'))
    files={}
    for name in file_names(metadata,args.check):
        path=ROOT/name
        if not path.resolve().is_relative_to(ROOT.resolve()):raise ValueError('File outside repository: '+name)
        if path.is_symlink():raise ValueError('Do not include symbolic links: '+name)
        files[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    if args.check:
        expected=metadata['files']
        differences=sorted(n for n in set(expected)|set(files) if expected.get(n)!=files.get(n))
        if differences:
            print(json.dumps({'valid':False,'changed_missing_or_extra':differences},indent=2));return 1
    else:
        metadata['files']=files
        temporary=MANIFEST.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
        temporary.replace(MANIFEST)
    print(json.dumps({'valid':True,'files_hashed':len(files),'manifest_excluded':MANIFEST.name,'mode':'check' if args.check else 'refresh'}))
    return 0

if __name__=='__main__':sys.exit(main())
