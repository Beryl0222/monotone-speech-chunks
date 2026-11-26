"""Replay the ordered data/events.json operations through one shared core."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .result import Result
from .sentence import SentenceInputs
from .reader import StreamingReader
from .training import train_forward_backward, train_sentence


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items() if k not in {"context", "attention", "gradients"}}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def replay_events(
    materials: Mapping[str, Any] | SentenceInputs,
    events: list[Mapping[str, Any]],
    *,
    include_arrays: bool = False,
) -> Result:
    sentence = (
        materials
        if isinstance(materials, SentenceInputs)
        else SentenceInputs.from_materials(materials, load_constraints=False)
    )
    reader = StreamingReader(sentence, constraints=[])
    records_by_id = {r["id"]: r for r in getattr(sentence, "_raw_records", [])}
    if not records_by_id and not isinstance(materials, SentenceInputs):
        records_by_id = {r["id"]: r for r in materials.get("word_constraints", [])}
    material_revision = None if isinstance(materials, SentenceInputs) else materials.get("constraint_revision")

    steps: list[Result] = []
    train_queries: list[Result] = []

    for number, event in enumerate(events):
        op = event["op"]
        out: Any = None
        if op == "audio_arrive":
            out = reader.audio_arrive(event.get("start"), event.get("stop"), event.get("confirmed_stop"))
        elif op in {"confirm", "audio_confirm"}:
            out = reader.confirm(int(event["confirmed_stop"]))
        elif op == "seal_utterance":
            out = reader.seal_utterance(event.get("last_valid_frame"))
        elif op == "annotations":
            specs = event.get("records", [])
            loaded = []
            for spec in specs:
                if isinstance(spec, str):
                    rec = records_by_id[spec]
                else:
                    rec = spec
                loaded.append(reader.constraints.load(rec))
            sentence.constraints = reader.constraints
            out = Result(op=op, loaded=[r.id for r in loaded])
        elif op == "revise_annotation":
            payload = dict(event)
            rid = payload.get("record", payload.get("id"))
            if (
                material_revision is not None
                and material_revision.get("id") == rid
                and int(material_revision.get("revision", -1)) == int(payload.get("revision", -2))
            ):
                full = dict(material_revision)
                full.update(payload)
                payload = full
            out = reader.revise_annotation(payload)
            sentence.constraints = reader.constraints
        elif op == "online_propose":
            out = reader.online_propose(event.get("word"))
        elif op in {"commit", "confirm_commit"}:
            out = reader.commit(event.get("word"), boundary=event.get("boundary"))
        elif op == "withdraw_uncommitted":
            out = reader.withdraw_uncommitted(event.get("word"))
        elif op == "soft_forward":
            words = event.get("words", sentence.word_ids)
            indices = [reader._index(w) for w in words]
            stop = reader.arrived_stop or sentence.num_frames
            e = sentence.selection_energies(stop)[indices]
            u = sentence.chunk_energies(stop)[indices]
            c = reader.constraints.factors_matrix(stop, sentence.num_words)[indices]
            core = train_forward_backward(
                e=e,
                u=u,
                V=sentence.V,
                chunk_width=sentence.chunk_width,
                constraints=c if reader.constraints.records else None,
                force_end=False,
                valid=sentence.valid[:stop],
            )
            out = Result(
                op=op,
                words=words,
                alpha=core.alpha,
                wait=core.wait,
                context=core.context if include_arrays else None,
                path_mass=core.path_mass,
            )
        elif op == "train_query":
            out = train_sentence(sentence, arrived_stop=reader.arrived_stop, sealed=reader.sealed)
            train_queries.append(out)
        else:
            out = Result(op=op, skipped=True, reason="unknown_operation")
        step = Result(index=number, event=event, output=out)
        steps.append(step)

    final = Result(
        arrived_stop=reader.arrived_stop,
        confirmed_stop=reader.confirmed_stop,
        sealed=reader.sealed,
        last_committed_boundary=reader.last_committed_boundary,
        committed=[dict(r) for r in reader.subtitles],
        provisional_words=list(reader.provisional),
        next_word_index=reader.next_word_index,
    )
    return Result(
        reader=reader,
        steps=steps,
        outputs=[step.output for step in steps],
        results=[step.output for step in steps],
        final=final,
        train_queries=train_queries,
    )


def replay_file(materials_path: str | Path, events_path: str | Path) -> Result:
    materials = json.loads(Path(materials_path).read_text(encoding="utf-8"))
    events = json.loads(Path(events_path).read_text(encoding="utf-8"))
    return replay_events(materials, events)


def process_events(*args: Any, **kwargs: Any) -> Result:
    return replay_events(*args, **kwargs)


def solve(
    materials: Mapping[str, Any] | str | Path | None = None,
    events: list[Mapping[str, Any]] | str | Path | None = None,
) -> Result:
    """Convenience end-to-end entry: load JSON inputs if needed and replay."""
    if materials is None:
        materials = Path(__file__).resolve().parents[1] / "data" / "materials.json"
    if events is None:
        events = Path(__file__).resolve().parents[1] / "data" / "events.json"
    if not isinstance(materials, Mapping):
        materials = json.loads(Path(materials).read_text(encoding="utf-8"))
    if isinstance(events, (str, Path)):
        events = json.loads(Path(events).read_text(encoding="utf-8"))
    return replay_events(materials, events)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Replay monotone chunk subtitle events")
    parser.add_argument("--materials", default="data/materials.json")
    parser.add_argument("--events", default="data/events.json")
    parser.add_argument("--json", action="store_true", help="print JSON-safe step summary")
    args = parser.parse_args(argv)
    report = replay_file(args.materials, args.events)
    if args.json:
        summary = {
            "final": _json_safe(report.final),
            "steps": [
                {
                    "index": s.index,
                    "op": s.event["op"],
                    "output": _json_safe(s.output),
                }
                for s in report.steps
            ],
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        for s in report.steps:
            out = s.output
            if isinstance(out, Mapping):
                status = out.get("status", out.get("accepted", out.get("eligible", "")))
                boundary = out.get("boundary", "")
                print(f"{s.index:2d} {s.event['op']:24s} status={status} boundary={boundary}")
            else:
                print(f"{s.index:2d} {s.event['op']}")
        print(json.dumps(_json_safe(report.final), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
