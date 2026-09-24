"""Small rule-execution controls, without inference of rules or compaction."""
from collections import Counter
import json


def clockwise(grid):
    return [list(row) for row in zip(*grid[::-1])]


def counterclockwise(grid):
    return [list(row) for row in zip(*grid)][::-1]


def build_tasks():
    cases = {
        "previous_test": [[0, 7, 0, 0], [0, 0, 0, 8], [9, 0, 0, 0]],
        "dense_rectangle": [[1, 2, 3], [4, 5, 6]],
        "tall_sparse": [[1, 0], [0, 2], [3, 0], [0, 4]],
    }
    rules = {
        "named_rule": "90-degree clockwise rotation, changing grid dimensions from m x n to n x m.",
        "coordinate_rule": (
            "90-degree clockwise rotation, changing grid dimensions from m x n to n x m. "
            "Use zero-based indices: input[r][c] moves to output[c][m-1-r]. "
            "Equivalently, output[r][c] = input[m-1-c][r], "
            "for output row r in 0..n-1 and output column c in 0..m-1."
        ),
    }
    tasks = []
    for case_id, grid in cases.items():
        for condition, rule in rules.items():
            tasks.append({
                "id": f"{case_id}_{condition}", "case_id": case_id,
                "condition": condition, "input_grid": grid, "provided_rule": rule,
                "prompt": (
                    f"Apply this known rule: {rule}\n"
                    f"Input grid: {json.dumps(grid, separators=(',', ':'))}.\n"
                    "Output ONLY the transformed grid as a JSON array of rows, with no prose or Markdown."
                ),
                "expected": clockwise(grid),
            })
    return tasks


def classify_answer(answer, task, status):
    if status != "completed":
        return "unfinished"
    try:
        grid = json.loads(answer)
    except (TypeError, ValueError):
        return "invalid_json"
    if not isinstance(grid, list) or not grid or not all(
        isinstance(row, list) and row and all(type(cell) is int for cell in row) for row in grid
    ) or len({len(row) for row in grid}) != 1:
        return "invalid_grid"
    if grid == task.get("expected"):
        return "correct"
    source = task.get("input_grid")
    if source and grid == counterclockwise(source):
        return "counterclockwise"
    if source and grid == source:
        return "unchanged"
    return "other_wrong_grid"


def summarize_conditions(rows):
    result = {}
    for condition in sorted({r.get("condition") or "unspecified" for r in rows}):
        group = [r for r in rows if (r.get("condition") or "unspecified") == condition]
        scored = [r for r in group if r["status"] == "completed" and r["correct"] is not None]
        correct = sum(r["correct"] for r in scored)
        result[condition] = {
            "tasks": len(group), "completed": sum(r["status"] == "completed" for r in group),
            "scored": len(scored), "correct": correct,
            "accuracy_completed": correct / len(scored) if scored else None,
            "success_rate_all": correct / len(group),
            "answer_kinds": dict(Counter(r.get("answer_kind", "unknown") for r in group)),
        }
    return result


if __name__ == "__main__":
    # Dataset regeneration: python -m reasoning_memory.controls > data/rotation_controls.jsonl
    for task in build_tasks():
        print(json.dumps(task, ensure_ascii=False))
