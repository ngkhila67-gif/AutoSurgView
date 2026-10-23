"""Lightweight, serial study launcher: no Torch imports in the parent process."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent


def load(path, attempts=12):
    """Retry short Windows sharing conflicts; malformed scientific JSON is an error."""
    for attempt in range(attempts):
        try:
            return json.loads(Path(path).read_text(encoding='utf-8-sig'))
        except PermissionError:
            if attempt == attempts-1: raise
            time.sleep(.025*(attempt+1))


def save(path, data):
    path=Path(path);temp=path.with_name(path.name+'.runner.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    for attempt in range(10):
        try: temp.replace(path);return
        except PermissionError: time.sleep(.05*(attempt+1))
    temp.replace(path)


def utc(): return datetime.now(timezone.utc).isoformat()
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(out, protocol):
    for name,expected in protocol['source_sha256'].items():
        if digest(ROOT/'ncsim'/name)!=expected:raise RuntimeError('Frozen source changed: '+name)
    for name,expected in protocol['input_sha256'].items():
        if digest(out/name)!=expected:raise RuntimeError('Frozen input changed: '+name)
    for name,expected in protocol['backend_sha256'].items():
        if digest(ROOT/name)!=expected:raise RuntimeError('Frozen backend changed: '+name)


class StatusReporter:
    """Best-effort UI telemetry. A file lock must never terminate a training job."""
    def __init__(self, out, protocol, jobs):
        self.out=Path(out);self.protocol=protocol;self.jobs=jobs;self.cache={}

    def update(self, status, **extra):
        states=[];warnings=[]
        for method,seed in self.jobs:
            key=(method,seed);path=self.out/f'{method}_seed_{seed}'/'progress.json'
            try:
                if path.exists():
                    item=load(path)
                    if not isinstance(item,dict):raise ValueError('Progress must be a JSON object')
                    self.cache[key]=dict(item)
                elif key in self.cache:
                    raise FileNotFoundError(str(path))
                else:
                    item=dict(status='queued',method=method,seed=seed)
            except (OSError,json.JSONDecodeError,ValueError) as exc:
                item=dict(self.cache.get(key,dict(status='progress_unavailable',method=method,seed=seed)))
                item.update(progress_stale=True,progress_read_error=str(exc))
                warnings.append(dict(method=method,seed=seed,error=str(exc)))
            item=dict(item);item['log']=f'{method}_seed_{seed}.log';states.append(item)
        document=dict(status=status,stage=self.protocol['stage'],updated_utc=utc(),jobs=states,
                      telemetry_warnings=warnings,**extra)
        try:
            save(self.out/'status.json',document)
            return True
        except OSError as exc:
            print(f'[{utc()}] Progress display write deferred; worker remains running: {exc}',file=sys.stderr,flush=True)
            return False


def memory_available():
    if os.name!='nt':return dict(available_commit_gb=None,available_physical_gb=None)
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_=[('dwLength',ctypes.c_ulong),('dwMemoryLoad',ctypes.c_ulong),
            ('ullTotalPhys',ctypes.c_ulonglong),('ullAvailPhys',ctypes.c_ulonglong),
            ('ullTotalPageFile',ctypes.c_ulonglong),('ullAvailPageFile',ctypes.c_ulonglong),
            ('ullTotalVirtual',ctypes.c_ulonglong),('ullAvailVirtual',ctypes.c_ulonglong),
            ('ullAvailExtendedVirtual',ctypes.c_ulonglong)]
    state=MEMORYSTATUSEX();state.dwLength=ctypes.sizeof(state)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
        raise ctypes.WinError()
    return dict(available_commit_gb=state.ullAvailPageFile/2**30,available_physical_gb=state.ullAvailPhys/2**30)


def run(out, min_commit_gb=1.5, resume_evaluation_job=None):
    out=Path(out).resolve();p=load(out/'protocol.json');verify(out,p)
    jobs=[(m,s) for s in p['seeds'] for m in p['methods']]
    planned={f'{m}_seed_{s}':(m,s) for m,s in jobs}
    if resume_evaluation_job is not None:
        if resume_evaluation_job not in planned:
            raise ValueError('Resume job is not in the frozen method/seed plan')
        from run_learning_worker import validate_resume
        validate_resume(out,*planned[resume_evaluation_job],protocol=p)
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    launch_record=dict(pid=os.getpid(),created_utc=utc(),parallel=1,launcher=str(Path(__file__).resolve()),launcher_sha256=digest(__file__),worker_helper_sha256=digest(ROOT/'run_learning_worker.py'),
        resume_evaluation_job=resume_evaluation_job,min_commit_gb=min_commit_gb)
    history=out/'launch_history';history.mkdir(exist_ok=True)
    save(history/f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{os.getpid()}.json",launch_record)
    save(out/'runner.json',launch_record)
    update=StatusReporter(out,p,jobs).update

    current=None
    try:
        for method,seed in jobs:
            if (out/'STOP_REQUESTED').exists():update('stopped_between_jobs');return 2
            job=out/f'{method}_seed_{seed}'
            resuming=False
            if job.exists():
                if load(job/'progress.json')['status']=='completed':continue
                if job.name == resume_evaluation_job:
                    from run_learning_worker import validate_resume
                    validate_resume(out,method,seed,protocol=p)
                    resuming=True
                else:
                    raise RuntimeError('Existing incomplete attempt must be inspected; no silent restart: '+str(job))
            mem=memory_available()
            while mem['available_commit_gb'] is not None and mem['available_commit_gb']<min_commit_gb:
                update('waiting_for_memory',memory=mem)
                if (out/'STOP_REQUESTED').exists():update('stopped_waiting_for_memory');return 2
                time.sleep(15);mem=memory_available()
            verify(out,p)
            with (out/f'{method}_seed_{seed}.log').open('a' if resuming else 'w',encoding='utf-8') as log:
                current=subprocess.Popen([sys.executable,'-X','utf8','-X','faulthandler','-u',str(ROOT/'run_learning_worker.py'),
                    'resume-evaluation' if resuming else 'worker','--out',str(out),'--method',method,'--seed',str(seed)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),env=env)
                while current.poll() is None:
                    update('running',memory=memory_available(),worker_pid=current.pid)
                    time.sleep(5)
                code=current.returncode;current=None
            if code:
                progress=job/'progress.json'
                item=load(progress) if progress.exists() else dict(method=method,seed=seed)
                item.update(status='failed',exit_code=code,updated_utc=utc())
                if not job.exists():job.mkdir()
                save(progress,item)
                update('failed_remaining_jobs_queued',failed_method=method,failed_seed=seed,exit_code=code)
                return code
            update('aggregating')
            with (out/'aggregate.log').open('a',encoding='utf-8') as log:
                result=subprocess.run([sys.executable,'-X','utf8','-m','ncsim.learning_benchmark','aggregate','--out',str(out)],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),env=env)
            if result.returncode:
                update('aggregation_failed',exit_code=result.returncode);return result.returncode
            update('rendering_report')
            with (out/'report.log').open('a',encoding='utf-8') as log:
                rendered=subprocess.run([sys.executable,'-X','utf8','-m','ncsim.learning_report','--out',str(out)],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),env=env)
            # Rendering is secondary; valid scientific results remain valid if a plot fails.
            if rendered.returncode:
                save(out/'report_error.json',dict(exit_code=rendered.returncode,updated_utc=utc()))
        update('completed');return 0
    except Exception as exc:
        update('runner_failed',error=str(exc));raise
    finally:
        if current is not None and current.poll() is None:current.terminate()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True);parser.add_argument('--min-commit-gb',type=float,default=1.5)
    parser.add_argument('--resume-evaluation-job',help='Explicit completed-training job whose frozen evaluation journal may resume')
    args=parser.parse_args();sys.exit(run(args.out,args.min_commit_gb,args.resume_evaluation_job))
