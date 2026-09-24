"""Readable reports for finished and interrupted runs; full traces stay on disk."""
import json
from pathlib import Path


def report(directory):
    root = Path(directory)
    def read(name, default):
        path = root / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default

    manifest = read("manifest.json", {})
    print(f"Run: {root.name} | {manifest.get('protocol', '?')} | {manifest.get('status', '?')}")
    if manifest.get("error"):
        print("Run error:", manifest["error"])
    config = read("config.json", {})
    print("Model:", config.get("model_id"), "| quantization:", config.get("quantization"))
    print("Completed pairs:", manifest.get("completed_pairs", 0))
    rows = read("results.json", [])
    if not rows:
        print("No completed result records; inspect events.jsonl and manifest.json.")
    for row in rows:
        print(f"{row['task_id']} | {row['mode']} | {row['status']} | correct={row.get('correct')}"
              f" | {row.get('answer_kind', '')} | tokens={row.get('generated_tokens')}")
        if row.get("answer") is not None:
            print("  Answer:", row["answer"])
        if row.get("correct") is False:
            print("  Expected:", json.dumps(row.get("expected")))
    conditions = read("summary.json", {}).get("control_conditions")
    if conditions:
        print("\nRule execution controls (no compaction):")
        for condition, stats in conditions.items():
            print(f"  {condition}: {stats['correct']}/{stats['tasks']} correct; "
                  f"{stats['completed']} completed; {stats['answer_kinds']}")
    for taskdir in sorted(root.glob("task_*")):
        prefix_path = taskdir / "shared_prefix.json"
        if prefix_path.exists():
            prefix = json.loads(prefix_path.read_text(encoding="utf-8"))
            summary = prefix.get("archive", {}).get("e1", {}).get("summary")
            if summary:
                print(f"{taskdir.name} saved conclusion: {summary}")
        journal = taskdir / "events.jsonl"
        if journal.exists():
            for line in journal.read_text(encoding="utf-8").splitlines():
                try:
                    event = json.loads(line)
                except ValueError:
                    continue  # The last journal write may have been interrupted.
                if event.get("error"):
                    print(f"{taskdir.name} [{event.get('branch')}] {event['error']}")
    print("Offloaded KV cache retries:", read("backend.json", {}).get("offloaded_cache_retries", "unavailable"))
