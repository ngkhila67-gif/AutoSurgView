"""Compile locally archived source files without modifying their implementations."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
LIBS=Path(os.environ.get('NC_SSBOT_LIB', ROOT/'third_party'))  # folder holding orocos_kdl and Eigen
KDL=LIBS/'orocos_kdl'
EIGEN=LIBS/'PCL 1.10.1/3rdParty/Eigen/eigen3'
COLLISION=Path(os.environ.get('NC_ROBOT_SOURCE', ROOT.parents[1]/'robot-controller'))
TOOLCHAIN=Path(os.environ.get('MSYS2_UCRT64', 'C:/msys64/ucrt64'))

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    vendor=ROOT/'vendor'
    vendor.mkdir(exist_ok=True)
    for src,dst in [(KDL/'src',vendor/'kdl'),(EIGEN/'Eigen',vendor/'eigen/Eigen')]:
        shutil.copytree(src,dst,dirs_exist_ok=True)
    shutil.copy2(KDL/'COPYING',vendor/'KDL_COPYING')
    shutil.copy2(KDL/'build/src/config.h',vendor/'kdl/config.h')
    (vendor/'collision').mkdir(exist_ok=True)
    compile_collision=ROOT/'compile_utf8'
    compile_collision.mkdir(exist_ok=True)
    for name in ['collision.cpp','collision.h','q_tree.h']:
        shutil.copy2(COLLISION/name,vendor/'collision'/name)
        # Only source encoding changes. Keep byte-identical originals in vendor.
        original=(COLLISION/name).read_bytes().decode('gb18030')
        (compile_collision/name).write_bytes(original.encode('utf-8'))
    sources=['chain.cpp','segment.cpp','joint.cpp','frames.cpp','frames_io.cpp',
             'rigidbodyinertia.cpp','rotationalinertia.cpp','jntarray.cpp','jacobian.cpp',
             'chainfksolverpos_recursive.cpp','chainjnttojacsolver.cpp',
             'chainiksolvervel_pinv.cpp','chainiksolverpos_nr.cpp',
             'utilities/svd_HH.cpp','utilities/utility.cxx','utilities/utility_io.cxx','utilities/error_stack.cxx']
    args=[str(TOOLCHAIN/'bin/g++.exe'),'-std=c++14','-O2','-w','-shared',
          '-include','kdl_compile_compat.h',
          '-static-libgcc','-static-libstdc++','-Ivendor/kdl','-Ivendor/eigen',
          '-Icompile_utf8',f'-I{TOOLCHAIN.as_posix()}/include/opencv4',
          'backend.cpp','compile_utf8/collision.cpp',
          *['vendor/kdl/'+s for s in sources],
          f'-L{TOOLCHAIN.as_posix()}/lib','-lopencv_core','-o','original_backend.dll']
    env=os.environ.copy();env['PATH']=str(TOOLCHAIN/'bin')+os.pathsep+env['PATH']
    result=subprocess.run(args,cwd=ROOT,env=env,capture_output=True,text=True,errors='replace')
    (ROOT/'build.log').write_text(result.stdout+result.stderr,encoding='utf-8')
    if result.returncode:
        print((result.stdout+result.stderr)[-10000:]);return result.returncode
    provenance={}
    for p in (vendor/'kdl').rglob('*'):
        if p.is_file():
            src=(KDL/'build/src/config.h') if p.name=='config.h' else (KDL/'src'/p.relative_to(vendor/'kdl'))
            assert p.read_bytes()==src.read_bytes()
            provenance[str(p.relative_to(ROOT))]=dict(source=str(src),sha256=digest(p))
    for name in ['collision.cpp','collision.h','q_tree.h']:
        p=vendor/'collision'/name
        assert p.read_bytes()==(COLLISION/name).read_bytes()
        provenance[str(p.relative_to(ROOT))]=dict(source=str(COLLISION/name),sha256=digest(p))
    record=dict(engine='archived_cpp_native_v2',kdl_version='1.5.1',
                kdl_source=str(KDL),collision_source=str(COLLISION),
                runtime_dll_directory=str(TOOLCHAIN/'bin'),
                compiler=subprocess.check_output([args[0],'--version'],env=env,text=True).splitlines()[0],
                command=args,source_files=provenance,
                binding_sha256=digest(ROOT/'backend.cpp'),dll_sha256=digest(ROOT/'original_backend.dll'),
                build_compatibility='Collision source transcoded GB18030 to UTF-8 with identical decoded text; forced declaration of KDL::PI already defined in archived utility.cxx. No solver or collision algorithm edits.',
                transcoded_source_sha256={str(p.relative_to(ROOT)):digest(p) for p in compile_collision.glob('*')},
                compatibility_header_sha256=digest(ROOT/'kdl_compile_compat.h'),
                scope='Original archived KDL solver source and NC deployment collision.cpp compiled without algorithm edits. Historical Python collision module identity remains unverified.')
    (ROOT/'build_manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Compiled archived KDL 1.5.1 and unmodified NC collision.cpp successfully.')
    return 0

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.exit(main())
