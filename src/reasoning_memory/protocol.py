"""Strict v0 grammar: flat experiments, mandatory summary, optional restore."""
from dataclasses import dataclass
import re

STOP_STRINGS = ["</summary>", "/>", "</answer>"]
IDENT = r"[A-Za-z][A-Za-z0-9_-]{0,31}"
EXPERIMENT = re.compile(
    rf'(?P<lead>.*?)<experiment id="(?P<id>{IDENT})">(?P<body>.*?)</experiment>'
    rf'\s*<summary id="(?P=id)">(?P<summary>.*?)</summary>\s*', re.S
)
RESTORE = re.compile(rf'(?P<lead>.*?)<restore id="(?P<id>{IDENT})"/>\s*', re.S)
ANSWER = re.compile(r'(?P<lead>.*?)</think>\s*<answer>(?P<answer>.*?)</answer>\s*', re.S)
RESERVED = re.compile(r'</?(?:experiment|summary|restore|restored|answer|think)\b')


class ProtocolError(ValueError):
    pass


@dataclass
class Event:
    kind: str
    lead: str
    id: str = ""
    body: str = ""
    summary: str = ""
    answer: str = ""


def parse(text: str) -> Event:
    for kind, pattern in (("experiment", EXPERIMENT), ("restore", RESTORE), ("answer", ANSWER)):
        match = pattern.fullmatch(text)
        if not match:
            continue
        data = match.groupdict()
        for key in ("lead", "body", "summary", "answer"):
            if RESERVED.search(data.get(key, "")):
                raise ProtocolError(f"Nested or misplaced protocol tag in {key}")
        for key in ("body", "summary", "answer"):
            if key in data and not data[key].strip():
                raise ProtocolError(f"Empty {key}")
        return Event(kind=kind, **data)
    raise ProtocolError("Expected a completed experiment+summary, restore, or final answer")


def system_prompt(allow_restore: bool) -> str:
    restore = (
        'When details of an archived experiment are needed, emit <restore id="e1"/>. '
        'The controller inserts the original experiment body. Each ID can be restored once. '
        'Restored details remain until the next completed experiment, then are archived again.'
        if allow_restore else 'Restoration is disabled; do not emit restore requests.'
    )
    return '''Solve the task using experiments inside ONE thinking block.
Use this exact flat protocol (no nesting, unique IDs e1, e2, ...):
<experiment id="e1">Detailed investigation of one hypothesis or calculation.</experiment>
<summary id="e1">A concise result with necessary conditions and uncertainty.</summary>
The controller may replace a completed experiment with its summary automatically.
Always produce the summary immediately after the experiment. Never decide whether to compact.
Complete at least one experiment before answering. You may conduct more experiments.
Do not emit controller-owned <restored> tags. Do not quote protocol tags inside prose.
Finish by closing the thinking block and giving a concise answer:
</think><answer>final answer only</answer>
''' + restore


def guided_prompt() -> str:
    return '''Solve the task. The controller supplies the phase tags for one experiment.
Inside <experiment>, investigate the task and end with </experiment>.
Inside <summary>, write only a short conclusion with the result and necessary conditions,
then end with </summary>. Do not repeat the investigation.
Inside <answer>, write only the requested final answer and end with </answer>.
Do not start new phases or quote these tags. Continue the already opened phase.'''
