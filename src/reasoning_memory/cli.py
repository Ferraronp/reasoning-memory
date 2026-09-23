import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

from .backend import make_backend
from .config import Config
from .engine import Engine


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def load_tasks(path):
    tasks = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        task = json.loads(line)
        if not isinstance(task.get("id"), str) or not isinstance(task.get("prompt"), str):
            raise ValueError("Each task needs string id and prompt")
        tasks.append(task)
    if not tasks or len({t["id"] for t in tasks}) != len(tasks):
        raise ValueError("Dataset must be nonempty with unique IDs")
    return tasks


def environment():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    packages = sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())
    return {"python": sys.version, "platform": platform.platform(), "git_commit": commit, "packages": packages}


def doctor():
    from . import __version__
    print("reasoning-memory", __version__, flush=True)
    print(json.dumps(environment(), ensure_ascii=False, indent=2))
    try:
        import torch
        print("CUDA:", torch.cuda.is_available())
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            print("GPU:", p.name, "VRAM GiB:", round(p.total_memory / 2**30, 2))
    except ImportError:
        print("PyTorch not installed; mock mode is available.")


def execute(args):
    cfg = Config.load(args.config)
    if args.seed is not None:
        cfg = replace(cfg, seed=args.seed)
    tasks = load_tasks(args.tasks)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        tasks = tasks[:args.limit]
    # Validate before loading model weights or creating an output directory.
    for task in tasks:
        followup = task.get("followup")
        if cfg.protocol in {"guided_two_stage", "guided_chat_two_stage"}:
            if not isinstance(followup, str) or not followup.strip():
                raise ValueError("Two-stage guided protocol requires a nonempty task.followup")
        elif followup is not None:
            raise ValueError("task.followup requires a two-stage guided protocol")
    out = Path(args.output or ("runs/" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")))
    out.mkdir(parents=True, exist_ok=False)  # Never overwrite an earlier experiment.
    write_json(out / "config.json", cfg.to_dict())
    write_json(out / "environment.json", environment())
    write_json(out / "tasks.json", tasks)
    manifest = {"command": args.command, "data_sha256": hashlib.sha256(Path(args.tasks).read_bytes()).hexdigest(),
                "status": "starting", "real_model_run": cfg.backend != "mock",
                "protocol": cfg.protocol, "completed_pairs": 0, "failed_shared_prefixes": 0}
    write_json(out / "manifest.json", manifest)
    results = []
    try:
        backend = make_backend(cfg)
        engine = Engine(backend, cfg)
        for index, task in enumerate(tasks):
            taskdir = out / f"task_{index:05d}"
            taskdir.mkdir()
            with (taskdir / "events.jsonl").open("w", encoding="utf-8") as journal:
                def emit(branch):
                    def save(event):
                        journal.write(json.dumps({"branch": branch, **event}, ensure_ascii=False) + "\n")
                        journal.flush()
                        if event.get("phase") and not event.get("error"):
                            gen = event.get("generation", {})
                            print(f'[{branch}] {event["phase"]} (stage {event.get("stage", 1)}): {gen.get("generated_tokens", 0)} generated tokens; '
                                  f'active context {event["active_after_tokens"]} tokens', flush=True)
                            generated_text = gen.get("text", "")
                            print(generated_text[:1200] + ('\n[Full text in events.jsonl]' if len(generated_text) > 1200 else ''), flush=True)
                        if event.get("error"):
                            print(f'[{branch}] {event["status"]}: {event["error"]}', flush=True)
                            print(event.get("generation", {}).get("text", "")[-2000:], flush=True)
                    return save
                state = engine.start(task["prompt"], task.get("followup"))
                if args.command == "pair":
                    engine.run(state, "full", emit("shared"), stop_after_first=True)
                    write_json(taskdir / "shared_prefix.json", engine.result(state))
                    if not state.archive:
                        # No intervention occurred. Do not invent two failed treatment arms.
                        result = {"task_id": task["id"], "mode": "shared", "pairable": False,
                                  **engine.result(state)}
                        write_json(taskdir / "shared_failure.json", result)
                        results.append({k: v for k, v in result.items() if k not in {"archive", "active_text"}})
                        write_json(out / "results.json", results)
                        manifest["failed_shared_prefixes"] += 1
                        print(task["id"], "PAIR NOT CREATED:", state.status, flush=True)
                        continue
                    modes = ["full", "compact"]
                else:
                    modes = [args.mode]
                for mode in modes:
                    branch = engine.fork(state, mode)
                    write_json(taskdir / f"{mode}_start.json", {"active_text": branch.text})
                    engine.run(branch, mode, emit(mode))
                    result = {"task_id": task["id"], "mode": mode,
                              "pairable": bool(state.archive) if args.command == "pair" else None,
                              **engine.result(branch, task.get("expected"))}
                    write_json(taskdir / f"{mode}.json", result)
                    results.append({k: v for k, v in result.items() if k not in {"archive", "active_text"}})
                    write_json(out / "results.json", results)
                    print(task["id"], mode, result["status"], repr(result["answer"]), flush=True)
                if args.command == "pair" and all(r["status"] == "completed" for r in results[-2:]):
                    manifest["completed_pairs"] += 1
        write_json(out / "backend.json", backend.metadata())
        manifest["status"] = "finished" if all(r["status"] == "completed" for r in results) else "finished_with_errors"
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(out / "manifest.json", manifest)
    summarize(out)
    return 0 if all(r["status"] == "completed" for r in results) else 2


def summarize(directory):
    directory = Path(directory)
    rows = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    summary = {}
    for mode in sorted({r["mode"] for r in rows}):
        group = [r for r in rows if r["mode"] == mode]
        # Also supports old logs, where an unpairable prefix was duplicated across arms.
        # Historical runs used correct=false for protocol failures. Keep those
        # rows in completion counts, but not in answer accuracy.
        scored = [r for r in group if r["status"] == "completed" and
                  r["correct"] is not None and r.get("pairable") is not False]
        summary[mode] = {"tasks": len(group), "completed": sum(r["status"] == "completed" for r in group),
                         "failed": sum(r["status"] != "completed" for r in group),
                         "scored": len(scored),
                         "accuracy": sum(r["correct"] for r in scored) / len(scored) if scored else None,
                         "generated_tokens": sum(r["generated_tokens"] for r in group),
                         "prefill_tokens": sum(r["prefill_tokens"] for r in group)}
    summary["diagnostics"] = {"unpairable_tasks": len({r["task_id"] for r in rows if r.get("pairable") is False}),
                              "note": "Unpairable prefixes are excluded from arm accuracy; report their count separately."}
    write_json(directory / "summary.json", summary)
    print(json.dumps(summary, indent=2), "\nSaved:", directory.resolve())


def main():
    parser = argparse.ArgumentParser(description="Mandatory experiment compaction research starter")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    summary = sub.add_parser("summarize")
    summary.add_argument("directory")
    for command in ("run", "pair"):
        p = sub.add_parser(command)
        p.add_argument("--config", default="configs/mock.json")
        p.add_argument("--tasks", default="data/smoke.jsonl")
        p.add_argument("--output")
        p.add_argument("--limit", type=int)
        p.add_argument("--seed", type=int)
        if command == "run":
            p.add_argument("--mode", choices=["full", "compact"], default="compact")
    args = parser.parse_args()
    if args.command == "doctor":
        doctor()
        return 0
    if args.command == "summarize":
        summarize(args.directory)
        return 0
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
