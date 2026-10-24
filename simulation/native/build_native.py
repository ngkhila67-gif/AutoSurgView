"""Build both archived backends and install them at the default package paths."""
import argparse,hashlib,json,os,shlex,shutil,subprocess,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
SIM=HERE.parent
SOURCE=HERE/'source'
HISTORY=SIM/'outputs/historical_reproduction_20261002'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))

def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')

def build(compiler=None):
    if os.name!='nt' and not sys.platform.startswith('linux'):
        raise RuntimeError('Build on Windows x64 or Linux x86-64.')
    platform,extension=('windows','.dll') if os.name=='nt' else ('linux','.so')
    source_manifest=read(HERE/'source_manifest.json')
    for relative,expected in source_manifest['files'].items():
        if sha(SOURCE/relative)!=expected:raise ValueError('Native source checksum mismatch: '+relative)
    env=os.environ.copy()
    if os.name=='nt':
        toolchain=Path(os.environ.get('MSYS2_UCRT64','C:/msys64/ucrt64'))
        compiler=compiler or os.environ.get('CXX') or str(toolchain/'bin/g++.exe')
        env['PATH']=str(toolchain/'bin')+os.pathsep+env.get('PATH','')
        flags=['-static-libgcc','-static-libstdc++','-I'+(toolchain/'include/opencv4').as_posix()]
        libraries=['-L'+(toolchain/'lib').as_posix(),'-lopencv_core']
    else:
        compiler=compiler or os.environ.get('CXX','g++')
        flags=['-fPIC']+shlex.split(subprocess.check_output(['pkg-config','--cflags','opencv4'],text=True))
        libraries=shlex.split(subprocess.check_output(['pkg-config','--libs-only-L','opencv4'],text=True))+['-lopencv_core','-Wl,--no-undefined']
    out=HERE/'build'/platform;out.mkdir(parents=True,exist_ok=True)
    common=[compiler,'-std=c++14','-O2','-shared','-include','elf_exports.h','-include','kdl_compile_compat.h',
            '-Ivendor/kdl','-Ivendor/eigen','-Icollision',*flags]
    units=['vendor/kdl/'+name for name in source_manifest['kdl_compilation_units']]
    jobs=[('original_backend','baseline_backend.cpp',[HERE,SIM/'benchmark_three_seed/native'],'build_manifest.json'),
          ('historical_backend_05405','historical_backend.cpp',[HISTORY],'historical_backend_manifest.json')]
    installed=[]
    for stem,binding,folders,manifest_name in jobs:
        output=out/(stem+extension)
        command=common+[binding,'collision/collision.cpp',*units,*libraries,'-o',os.path.relpath(output,SOURCE)]
        result=subprocess.run(command,cwd=SOURCE,env=env,capture_output=True)
        (out/(stem+'.log')).write_bytes(result.stdout+result.stderr)
        if result.returncode:raise RuntimeError('Native compilation failed; see '+str(out/(stem+'.log')))
        for folder in folders:
            folder.mkdir(parents=True,exist_ok=True);target=folder/output.name;shutil.copy2(output,target)
            metadata=read(folder/manifest_name)
            metadata['libraries'][platform]={'file':target.name,'sha256':sha(target),
                'compiler':subprocess.check_output([compiler,'--version'],env=env,text=True).splitlines()[0]}
            write(folder/manifest_name,metadata);installed.append(str(target.relative_to(SIM)))
    if os.name=='nt':
        runtime=HERE/'runtime/libwinpthread-1.dll';shutil.copy2(toolchain/'bin/libwinpthread-1.dll',runtime)
        notice=toolchain/'share/licenses/libwinpthread/COPYING'
        if notice.is_file():shutil.copy2(notice,HERE/'runtime/WINPTHREADS COPYING.txt')
        for folder,name in [(HERE,'build_manifest.json'),(SIM/'benchmark_three_seed/native','build_manifest.json'),
                            (HISTORY,'historical_backend_manifest.json')]:
            metadata=read(folder/name);metadata['runtime_files']={'libwinpthread-1.dll':sha(runtime)};write(folder/name,metadata)
    print(json.dumps({'platform':platform,'installed':installed,'source_files_verified':len(source_manifest['files'])},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler',help='C++ executable; otherwise CXX or the platform default')
    build(parser.parse_args().compiler)
