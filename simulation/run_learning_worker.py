"""I/O-only Windows recovery around frozen learning-study scientific modules.

No source file, model, RNG, case set, reward, or evaluator math is changed.
Retries re-invoke original I/O functions. Only progress.json telemetry may be
skipped after persistent sharing errors; scientific artifacts remain mandatory.
"""
from __future__ import annotations
import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
RETRY_ATTEMPTS = 16
PATCHED_FUNCTIONS = ["ncsim.learning_evaluation._save", "ncsim.learning_evaluation._write_csv",
                     "ncsim.learning_benchmark.dump", "ncsim.learning_benchmark.read"]
REASON = "Windows sharing violation during atomic progress replacement; preserve frozen training and resume verified evaluation journal"


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sharing_conflict(exc):
    return isinstance(exc, PermissionError) or (isinstance(exc, OSError) and getattr(exc, "winerror", None) in (32, 33))


def retry_io(function, *args, telemetry=False, attempts=RETRY_ATTEMPTS, event=None, **kwargs):
    """Retry only sharing/access errors, not corrupt data or scientific failures."""
    for attempt in range(attempts):
        try:
            result = function(*args, **kwargs)
            if attempt and event:
                event(dict(event="io_retry_recovered", function=getattr(function, "__name__", str(function)),
                           attempts=attempt + 1, path=str(args[0]) if args else None))
            return result
        except OSError as exc:
            if not sharing_conflict(exc):
                raise
            if attempt + 1 == attempts:
                if not telemetry:
                    raise
                warning = dict(event="progress_write_deferred", function=getattr(function, "__name__", str(function)),
                               attempts=attempts, path=str(args[0]) if args else None, error=str(exc))
                print(f"[{utc()}] Progress telemetry deferred; scientific work continues: {exc}", file=sys.stderr, flush=True)
                if event:
                    event(warning)
                return None
            time.sleep(min(.05 * (attempt + 1), .4))


def read_json(path):
    return retry_io(lambda p: json.loads(Path(p).read_text(encoding="utf-8-sig")), path)


def write_json(path, value):
    path = Path(path)
    def once():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + f".{os.getpid()}.io.tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(path)
    retry_io(once)


def identify_job(out, method, seed, protocol):
    out = Path(out).resolve()
    if method not in protocol["methods"] or seed not in protocol["seeds"]:
        raise ValueError("Job is not in the frozen method/seed plan")
    job = (out / f"{method}_seed_{seed}").resolve()
    if job.parent != out:
        raise ValueError("Job path must be a direct child of the study directory")
    return job


def validate_resume(out, method, seed, protocol=None):
    """Pure-I/O validation; never permit an arbitrary failed training restart."""
    from run_learning_study import verify
    out = Path(out).resolve()
    protocol = read_json(out / "protocol.json") if protocol is None else protocol
    verify(out, protocol)
    job = identify_job(out, method, seed, protocol)
    manifest = read_json(job / "training_manifest.json")
    budget = protocol["steps_per_method_seed"]
    if (manifest.get("actual_env_steps") != budget or manifest.get("requested_env_steps") != budget):
        raise ValueError("Evaluation resume requires the complete frozen training budget")
    if manifest.get("method") != method or manifest.get("seed") != seed:
        raise ValueError("Training manifest does not match the authorized method/seed")
    if manifest.get("protocol_sha256") != sha256(out / "protocol.json"):
        raise ValueError("Training manifest protocol checksum mismatch")
    if manifest.get("checkpoint_kind") != "validation_selected":
        raise ValueError("Only the original validation-selected checkpoint may be evaluated")
    selected = sha256(job / "best_model.zip")
    if manifest.get("selected_model_sha256") != selected:
        raise ValueError("Selected model checksum mismatch")
    if manifest.get("final_model_sha256") != sha256(job / "final_model.zip"):
        raise ValueError("Final training model checksum mismatch")
    selection = read_json(job / "best_validation.json")
    if manifest.get("selection") != selection:
        raise ValueError("Validation selection manifest mismatch")
    step = selection.get("step")
    if not isinstance(step, int) or not 0 < step <= budget:
        raise ValueError("Selected checkpoint step is outside the frozen budget")
    checkpoint = str(selection.get("checkpoint", ""))
    if not checkpoint or Path(checkpoint).name != checkpoint:
        raise ValueError("Invalid selected checkpoint path")
    if sha256(job / checkpoint) != selected:
        raise ValueError("Selected checkpoint file does not match best_model")
    return protocol, manifest


def archive_recovery(out, job, attempt_id):
    """Copy failure evidence; do not move or edit journals, model files or inputs."""
    out, job = Path(out).resolve(), Path(job).resolve()
    if job.parent != out:
        raise ValueError("Recovery job is outside the study")
    destination = out / "recovery_history" / job.name / attempt_id
    destination.mkdir(parents=True, exist_ok=False)
    relative_files = ["progress.json", "runtime_io_manifest.json", "runtime_io_events.jsonl",
                      "test_raw/progress.json", "test_raw/summary.json",
                      "test_shared_filter/progress.json", "test_shared_filter/summary.json"]
    sources = [(job / name, "job/" + name) for name in relative_files]
    sources += [(out / name, "study/" + name) for name in
                (job.name + ".log", "runner_stdout.log", "runner_stderr.log", "runner.json", "status.json")]
    records = []
    for source, relative in sources:
        if source.is_file():
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            retry_io(shutil.copy2, source, target)
            original_hash, copy_hash = sha256(source), sha256(target)
            if original_hash != copy_hash:
                raise RuntimeError("Failure evidence changed while being copied: " + str(source))
            records.append(dict(source=str(source), copy=relative, sha256=copy_hash))
    journals = []
    for mode in ("raw", "shared_filter"):
        journal = job / f"test_{mode}" / "episodes.jsonl"
        if journal.exists():
            with journal.open(encoding="utf-8") as handle:
                count = sum(1 for _ in handle)
            journals.append(dict(path=str(journal), sha256=sha256(journal), saved_cases=count))
    write_json(destination / "recovery_snapshot.json", dict(created_utc=utc(), reason=REASON,
               copied_files=records, journal_identity_before_resume=journals))
    return str(destination)


class RuntimeIO:
    def __init__(self, out, method, seed, protocol, mode):
        self.out = Path(out).resolve()
        self.job = identify_job(self.out, method, seed, protocol)
        self.attempt_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + f"_{os.getpid()}"
        self.record = dict(attempt_id=self.attempt_id, started_utc=utc(), pid=os.getpid(), mode=mode,
            method=method, seed=seed, reason=REASON, helper=str(Path(__file__).resolve()),
            helper_sha256=sha256(__file__), patched_functions=PATCHED_FUNCTIONS,
            original_source_sha256={name: sha256(ROOT / "ncsim" / name)
                                    for name in ("learning_benchmark.py", "learning_evaluation.py")},
            protocol_sha256=sha256(self.out / "protocol.json"),
            scientific_behavior="unchanged; original functions are invoked again only for Windows I/O conflicts",
            progress_after_retries="nonfatal telemetry only; all other artifacts remain mandatory")
        self.initialized = False

    def ensure(self):
        if self.initialized or not self.job.is_dir():
            return
        path = self.job / "runtime_io_manifest.json"
        old = read_json(path) if path.exists() else dict(version=1, attempts=[])
        if not isinstance(old.get("attempts"), list):
            raise ValueError("Malformed runtime I/O manifest")
        old["attempts"].append(self.record)
        write_json(path, old)
        self.initialized = True
        self.event(dict(event="runtime_io_attempt_started", **self.record), strict=True)

    def event(self, value, strict=False):
        value = dict(value)
        value.update(utc=utc(), attempt_id=self.attempt_id)
        def append():
            with (self.job / "runtime_io_events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
        try:
            retry_io(append)
        except OSError as exc:
            if strict:
                raise
            print(f"[{utc()}] Runtime event log unavailable: {exc}", file=sys.stderr, flush=True)


def install_io_retries(benchmark, evaluation, runtime):
    """Wrap exactly four I/O functions; never patch numerical execution."""
    for module, name, progress_allowed in ((evaluation, "_save", True), (evaluation, "_write_csv", False),
                                            (benchmark, "dump", True), (benchmark, "read", False)):
        original = getattr(module, name)
        @functools.wraps(original)
        def wrapped(*args, _original=original, _progress_allowed=progress_allowed, **kwargs):
            runtime.ensure()
            telemetry = bool(_progress_allowed and args and Path(args[0]).name == "progress.json")
            return retry_io(_original, *args, telemetry=telemetry, event=runtime.event, **kwargs)
        setattr(module, name, wrapped)


def resume_evaluation(out, method, seed, benchmark, evaluation, runtime):
    out = Path(out).resolve()
    protocol, manifest = validate_resume(out, method, seed)
    job = identify_job(out, method, seed, protocol)
    snapshot = archive_recovery(out, job, runtime.attempt_id)
    runtime.record["recovery_snapshot"] = snapshot
    runtime.ensure()
    benchmark.check_protocol(out)
    runtime.event(dict(event="verified_evaluation_resume", model_sha256=manifest["selected_model_sha256"],
                       actual_env_steps=manifest["actual_env_steps"], train_again=False), strict=True)
    benchmark.torch.set_num_threads(1)
    benchmark.torch.set_num_interop_threads(1)
    started = time.perf_counter()
    try:
        cases = benchmark.read(out / "test_cases.json")
        algorithm = "ppo" if method == "ppo_legacy" else method
        for shield in (False, True):
            mode = "shared_filter" if shield else "raw"
            def progress(value=None, _mode=mode):
                data = dict(status=f"evaluating_test_{_mode}", method=method, seed=seed,
                    steps=protocol["steps_per_method_seed"], budget=protocol["steps_per_method_seed"],
                    updated_utc=utc(), evaluation_resume=True, recovery_attempt_id=runtime.attempt_id,
                    elapsed_seconds=manifest["learning_wall_seconds"] + time.perf_counter() - started,
                    resumed_evaluation_seconds=time.perf_counter() - started)
                if isinstance(value, dict):
                    data.update(completed_cases=value.get("completed_cases"), total_cases=value.get("total_cases"))
                benchmark.dump(job / "progress.json", data)
            progress()
            result = evaluation.evaluate_checkpoint(job / "best_model.zip", algorithm, cases, job / f"test_{mode}",
                safety_filter=shield, progress_callback=progress, trace_limit=2, resume=True)
            if not result.get("complete") or result.get("candidates") != len(cases):
                raise RuntimeError("Resumed evaluation did not finish the full frozen case set")
        benchmark.dump(job / "progress.json", dict(status="completed", method=method, seed=seed,
            steps=protocol["steps_per_method_seed"], budget=protocol["steps_per_method_seed"], updated_utc=utc(),
            elapsed_seconds=manifest["learning_wall_seconds"] + time.perf_counter() - started,
            evaluation_resume=True, recovery_attempt_id=runtime.attempt_id,
            resumed_evaluation_seconds=time.perf_counter() - started,
            prior_failure_timing="preserved in recovery_snapshot; elapsed_seconds excludes prior failed evaluation"))
        runtime.event(dict(event="evaluation_resume_completed", cases_per_mode=len(cases)), strict=True)
    except Exception:
        error = traceback.format_exc()
        benchmark.dump(job / "progress.json", dict(status="failed", method=method, seed=seed, updated_utc=utc(),
                       evaluation_resume=True, recovery_attempt_id=runtime.attempt_id, error=error))
        runtime.event(dict(event="evaluation_resume_failed", error=error))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("worker", "resume-evaluation"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)
    from ncsim import learning_benchmark as benchmark, learning_evaluation as evaluation
    out = Path(args.out).resolve()
    protocol = benchmark.check_protocol(out)
    runtime = RuntimeIO(out, args.method, args.seed, protocol, args.command)
    install_io_retries(benchmark, evaluation, runtime)
    if args.command == "resume-evaluation":
        resume_evaluation(out, args.method, args.seed, benchmark, evaluation, runtime)
    else:
        # The frozen worker itself creates its job directory; first dump writes provenance.
        try:
            benchmark.worker(out, args.method, args.seed)
            runtime.event(dict(event="worker_completed"), strict=True)
        except Exception:
            if runtime.job.is_dir():
                runtime.ensure()
                runtime.event(dict(event="worker_failed", error=traceback.format_exc()))
            raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
