#!/usr/bin/env python3
"""Merge an external HolosTTS voice pack with the voices of a model bundle.

The published model ships with its own preset voices. This script adds the
voices of another pack (for example a CC0 pack) to them, or replaces them,
and writes the result in the format the target runtime expects:

  - ``voices.npz`` for ``holos-tts-onnx`` and ``holos-tts-mlx``;
  - ``voices.pt`` for PyTorch (``holos-tts``);
  - ``voices.json`` for tools that keep the legacy JSON format;
  - ``voices.bin`` plus ``holos_litert.json`` for LiteRT.

A pack may be a directory or a single voice file. A target may be a model
directory or a single voice file. The output is never written in place
unless ``--force`` is given, and the target is always read before writing.

Examples::

    # Append a CC0 pack to an ONNX/MLX bundle, prefixing its names.
    python scripts/merge_voice_pack.py \\
        holos-voices-uk-cc0 ~/models/holos-onnx \\
        --output ~/models/holos-onnx/voices-cc0.npz \\
        --prefix "cc0/"

    # Replace all voices in a PyTorch voices.pt.
    python scripts/merge_voice_pack.py \\
        holos-voices-uk-cc0 ~/models/voices.pt \\
        --output ~/models/voices-cc0.pt --replace

    # Build a LiteRT bundle's voice files (voices.bin + holos_litert.json).
    python scripts/merge_voice_pack.py \\
        holos-voices-uk-cc0 ~/models/holos-litert \\
        --output ~/models/holos-litert-cc0 --format litert

The pack must contain 256-dimensional float32 embeddings. The ``names`` and
``values`` arrays of ``voices.npz`` are exactly the format used by
``holos-tts-onnx`` and ``holos-tts-mlx``; no extra keys are written.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

VOICE_DIM = 256
NPZ_NAME = "voices.npz"
PT_NAME = "voices.pt"
JSON_NAME = "voices.json"
BIN_NAME = "voices.bin"
LITERT_MANIFEST = "holos_litert.json"
FORMATS = ("auto", "npz", "pt", "json", "litert")


class VoicePackError(Exception):
    """A pack, target, or output does not have the expected format."""


@dataclass
class VoiceSet:
    """Voices loaded from one file or bundle."""

    names: List[str]
    values: np.ndarray
    kind: str
    source: Path
    manifest: Optional[dict] = None
    manifest_path: Optional[Path] = None
    bin_name: str = BIN_NAME


def _name(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _validate(names: Sequence[str], values: np.ndarray, where: Path) -> Tuple[List[str], np.ndarray]:
    if isinstance(names, str):
        raise VoicePackError(f"{where}: 'names' must be a list, not a string")
    names = [_name(name) for name in names]
    try:
        values = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise VoicePackError(f"{where}: embeddings are not numeric: {error}") from error

    if not names:
        raise VoicePackError(f"{where}: no voices")
    if values.ndim != 2:
        raise VoicePackError(f"{where}: expected a 2-D array of voices, got shape {values.shape}")
    if values.shape[1] != VOICE_DIM:
        raise VoicePackError(
            f"{where}: expected {VOICE_DIM}-dimensional voices, got shape {values.shape}"
        )
    if len(names) != values.shape[0]:
        raise VoicePackError(
            f"{where}: {len(names)} names but {values.shape[0]} embeddings"
        )
    if any(not name for name in names):
        raise VoicePackError(f"{where}: empty voice name")
    if any(name != name.strip() for name in names):
        raise VoicePackError(f"{where}: voice names must not start or end with whitespace")
    if len(set(names)) != len(names):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise VoicePackError(f"{where}: duplicate voice names: {', '.join(duplicates)}")
    if not np.isfinite(values).all():
        raise VoicePackError(f"{where}: embeddings contain NaN or infinity")

    return names, np.ascontiguousarray(values, dtype=np.float32)


def _load_npz(path: Path) -> VoiceSet:
    try:
        archive = np.load(path, allow_pickle=False)
    except Exception as error:
        raise VoicePackError(f"{path}: cannot read voices.npz: {error}") from error
    with archive:
        if "names" not in archive or "values" not in archive:
            raise VoicePackError(f"{path}: expected the 'names' and 'values' arrays")
        name_array = archive["names"]
        if name_array.ndim != 1:
            raise VoicePackError(f"{path}: 'names' must be a 1-D array")
        names, values = _validate(
            name_array.tolist(), archive["values"], path
        )
    return VoiceSet(names, values, "npz", path)


def _load_pt(path: Path) -> VoiceSet:
    try:
        import torch
    except ImportError as error:
        raise VoicePackError(
            f"{path}: reading voices.pt needs PyTorch; use voices.npz or voices.json instead"
        ) from error

    try:
        data = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        # weights_only was added in PyTorch 1.13.
        data = torch.load(str(path), map_location="cpu")
    except Exception as error:
        raise VoicePackError(f"{path}: cannot read voices.pt: {error}") from error

    if not isinstance(data, dict):
        raise VoicePackError(f"{path}: voices.pt must contain a dict name -> tensor")

    names: List[str] = []
    rows: List[np.ndarray] = []
    for name, value in data.items():
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        try:
            row = np.asarray(value, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError) as error:
            raise VoicePackError(
                f"{path}: voice {name!r} is not a numeric vector: {error}"
            ) from error
        names.append(_name(name))
        rows.append(row)
    if not rows:
        raise VoicePackError(f"{path}: voices.pt is empty")
    try:
        values = np.stack(rows, axis=0)
    except ValueError as error:
        raise VoicePackError(f"{path}: voices have different sizes: {error}") from error
    names, values = _validate(names, values, path)
    return VoiceSet(names, values, "pt", path)


def _load_json(path: Path) -> VoiceSet:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VoicePackError(f"{path}: cannot read JSON: {error}") from error

    if not isinstance(data, dict):
        raise VoicePackError(f"{path}: expected a JSON object")

    if "names" in data and "values" in data:
        names = data["names"]
        try:
            values = np.asarray(data["values"], dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise VoicePackError(f"{path}: 'values' is not a numeric array: {error}") from error
    else:
        voices_field = data.get("voices")
        if isinstance(voices_field, dict) or (
            isinstance(voices_field, list)
            and voices_field
            and isinstance(voices_field[0], dict)
        ):
            raise VoicePackError(
                f"{path}: this looks like a manifest, not a voices.json"
            )
        names = []
        rows = []
        for name, value in data.items():
            try:
                row = np.asarray(value, dtype=np.float32).reshape(-1)
            except (TypeError, ValueError) as error:
                raise VoicePackError(
                    f"{path}: voice {name!r} is not a numeric vector: {error}"
                ) from error
            names.append(_name(name))
            rows.append(row)
        if not rows:
            raise VoicePackError(f"{path}: voices.json is empty")
        try:
            values = np.stack(rows, axis=0)
        except ValueError as error:
            raise VoicePackError(f"{path}: voices have different sizes: {error}") from error

    names, values = _validate(names, values, path)
    return VoiceSet(names, values, "json", path)


def _load_litert(directory: Path) -> VoiceSet:
    manifest_path = directory / LITERT_MANIFEST
    if not manifest_path.is_file():
        raise VoicePackError(f"{directory}: expected {LITERT_MANIFEST}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VoicePackError(f"{manifest_path}: cannot read JSON: {error}") from error

    voices = manifest.get("voices")
    if not isinstance(voices, dict) or not isinstance(voices.get("names"), list):
        raise VoicePackError(f"{manifest_path}: expected voices.names")

    names = [_name(name) for name in voices["names"]]
    try:
        dim = int(voices.get("dim", VOICE_DIM))
    except (TypeError, ValueError) as error:
        raise VoicePackError(f"{manifest_path}: voices.dim is not an integer") from error
    bin_name = str(voices.get("file", BIN_NAME))
    bin_path = directory / bin_name
    if not bin_path.is_file():
        raise VoicePackError(f"{manifest_path}: voice file {bin_name!r} not found")

    raw = np.fromfile(bin_path, dtype="<f4")
    if dim != VOICE_DIM:
        raise VoicePackError(f"{manifest_path}: expected dim {VOICE_DIM}, got {dim}")
    if raw.size != len(names) * dim:
        raise VoicePackError(
            f"{bin_path}: {raw.size} floats for {len(names)} voices of {dim} values"
        )
    values = raw.reshape(len(names), dim)
    names, values = _validate(names, values, bin_path)
    return VoiceSet(
        names,
        values,
        "litert",
        directory,
        manifest=manifest,
        manifest_path=manifest_path,
        bin_name=bin_name,
    )


def load_voices(path: Path) -> VoiceSet:
    """Load a pack or target from a directory or a voice file."""
    path = path.expanduser()
    if path.is_dir():
        if (path / LITERT_MANIFEST).is_file():
            return _load_litert(path)
        if (path / NPZ_NAME).is_file():
            return _load_npz(path / NPZ_NAME)
        if (path / PT_NAME).is_file():
            return _load_pt(path / PT_NAME)
        if (path / JSON_NAME).is_file():
            return _load_json(path / JSON_NAME)
        raise VoicePackError(
            f"{path}: no {NPZ_NAME}, {PT_NAME}, {JSON_NAME}, or {LITERT_MANIFEST}"
        )

    if not path.is_file():
        raise VoicePackError(f"{path}: file not found")

    if path.name == LITERT_MANIFEST:
        return _load_litert(path.parent)

    suffix = path.suffix.lower()
    if suffix == ".npz":
        return _load_npz(path)
    if suffix == ".pt":
        return _load_pt(path)
    if suffix == ".json":
        return _load_json(path)
    raise VoicePackError(
        f"{path}: unsupported voice file; expected .npz, .pt, .json, or {LITERT_MANIFEST}"
    )


def merge_voices(
    target: VoiceSet,
    pack: VoiceSet,
    replace: bool,
    prefix: str,
    skip_duplicates: bool,
) -> Tuple[List[str], np.ndarray, int]:
    """Return merged names, values, and the number of skipped duplicates."""
    pack_names = [prefix + name for name in pack.names]
    if len(set(pack_names)) != len(pack_names):
        raise VoicePackError("the prefix makes pack voice names non-unique")

    if replace:
        names, values = _validate(pack_names, pack.values, pack.source)
        return names, values, 0

    names = list(target.names)
    values = target.values
    existing = set(names)
    added_rows: List[np.ndarray] = []
    skipped = 0
    duplicates: List[str] = []

    for name, row in zip(pack_names, pack.values):
        if name in existing:
            if skip_duplicates:
                skipped += 1
                continue
            duplicates.append(name)
            continue
        names.append(name)
        added_rows.append(row)
        existing.add(name)

    if duplicates:
        shown = ", ".join(duplicates[:5])
        if len(duplicates) > 5:
            shown += f", ... ({len(duplicates)} total)"
        raise VoicePackError(
            "voice names already exist in the target: "
            f"{shown}. Use --prefix, --skip-duplicates, or --replace."
        )

    if added_rows:
        values = np.concatenate([values, np.stack(added_rows, axis=0)], axis=0)
    names, values = _validate(names, values, target.source)
    return names, values, skipped


def _resolve_kind(requested: str, output: Path, target_kind: str) -> str:
    """Resolve --format auto from the output extension, then the target."""
    if requested != "auto":
        return requested
    if output.is_dir():
        return target_kind
    suffix = output.suffix.lower()
    if suffix == ".npz":
        return "npz"
    if suffix == ".pt":
        return "pt"
    if suffix == ".json":
        return "json"
    if suffix == ".bin":
        if target_kind == "litert":
            return "litert"
        raise VoicePackError(
            "a .bin output needs a LiteRT target with holos_litert.json; "
            "use --format litert"
        )
    return target_kind


def _output_path(output: Path, kind: str) -> Path:
    output = output.expanduser()
    if kind == "litert":
        if output.suffix:
            raise VoicePackError("LiteRT output must be a directory, not a file")
        return output
    default_name = {"npz": NPZ_NAME, "pt": PT_NAME, "json": JSON_NAME}[kind]
    if output.suffix:
        return output
    return output / default_name


def _prepare_output(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise VoicePackError(f"{path}: already exists; use --force to overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)


def _default_file_mode() -> int:
    mask = os.umask(0)
    os.umask(mask)
    return 0o666 & ~mask


def _replace(temporary: str, path: Path) -> None:
    """Move a temporary file into place, keeping the output readable."""
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError:
        mode = _default_file_mode()
    os.chmod(temporary, mode)
    os.replace(temporary, path)


def _atomic_bytes(path: Path, data: bytes, force: bool) -> None:
    _prepare_output(path, force)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
        _replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_npz(path: Path, names: List[str], values: np.ndarray, force: bool) -> None:
    _prepare_output(path, force)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.stem + ".", suffix=".npz", dir=str(path.parent)
    )
    os.close(descriptor)
    try:
        np.savez_compressed(
            temporary,
            names=np.asarray(names),
            values=np.asarray(values, dtype="<f4"),
        )
        _replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_pt(path: Path, names: List[str], values: np.ndarray, force: bool) -> None:
    try:
        import torch
    except ImportError as error:
        raise VoicePackError(
            f"{path}: writing voices.pt needs PyTorch; use --format npz or --format json"
        ) from error

    _prepare_output(path, force)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.stem + ".", suffix=".pt", dir=str(path.parent)
    )
    os.close(descriptor)
    try:
        tensors = {
            name: torch.from_numpy(np.ascontiguousarray(row, dtype=np.float32))
            for name, row in zip(names, values)
        }
        torch.save(tensors, temporary)
        _replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_json(path: Path, names: List[str], values: np.ndarray, force: bool) -> None:
    payload = {
        name: [float(value) for value in row]
        for name, row in zip(names, values)
    }
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    _atomic_bytes(path, text.encode("utf-8"), force)


def _write_litert(
    output: Path,
    target: VoiceSet,
    names: List[str],
    values: np.ndarray,
    force: bool,
) -> Tuple[Path, Path]:
    if target.kind != "litert" or target.manifest is None:
        raise VoicePackError(
            "--format litert needs a target bundle with holos_litert.json"
        )

    if output.exists() and not output.is_dir():
        raise VoicePackError(f"{output}: LiteRT output must be a directory")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / LITERT_MANIFEST
    bin_path = output / target.bin_name
    _prepare_output(manifest_path, force)
    _prepare_output(bin_path, force)

    manifest = copy.deepcopy(target.manifest)
    manifest.setdefault("voices", {})
    manifest["voices"]["file"] = target.bin_name
    manifest["voices"]["names"] = names
    manifest["voices"]["dim"] = VOICE_DIM

    _atomic_bytes(bin_path, values.astype("<f4", copy=False).tobytes(order="C"), force)
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    _atomic_bytes(manifest_path, manifest_text.encode("utf-8"), force)
    return manifest_path, bin_path


def _summary(target: VoiceSet, pack: VoiceSet, names: List[str], skipped: int) -> str:
    lines = [
        f"target: {target.source} ({len(target.names)} voices, {target.kind})",
        f"pack:   {pack.source} ({len(pack.names)} voices, {pack.kind})",
        f"result: {len(names)} voices",
    ]
    if skipped:
        lines.append(f"skipped duplicates: {skipped}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("pack", type=Path, help="voice pack directory or voice file")
    parser.add_argument("target", type=Path, help="model bundle directory or voice file")
    parser.add_argument("-o", "--output", type=Path, required=True,
                        help="output file, or a directory for --format litert")
    parser.add_argument("--replace", action="store_true",
                        help="use only the pack voices instead of appending them")
    parser.add_argument("--prefix", default="",
                        help="prefix added to every pack voice name, e.g. 'cc0/'")
    parser.add_argument("--skip-duplicates", action="store_true",
                        help="skip pack voices whose names already exist in the target")
    parser.add_argument("--format", choices=FORMATS, default="auto",
                        help="output format (default: from the output extension, "
                             "then the target format)")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing output files")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan without writing anything")
    args = parser.parse_args(argv)

    try:
        pack = load_voices(args.pack)
        target = load_voices(args.target)
        names, values, skipped = merge_voices(
            target, pack, args.replace, args.prefix, args.skip_duplicates
        )
        kind = _resolve_kind(args.format, args.output, target.kind)
        if kind not in ("npz", "pt", "json", "litert"):
            raise VoicePackError(f"unsupported output format: {kind}")
        output = _output_path(args.output, kind)

        print(_summary(target, pack, names, skipped))
        print(f"format: {kind}")
        print(f"output: {output}")
        sys.stdout.flush()
        if args.dry_run:
            print("dry run: nothing written")
            return 0

        if kind == "npz":
            _write_npz(output, names, values, args.force)
        elif kind == "pt":
            _write_pt(output, names, values, args.force)
        elif kind == "json":
            _write_json(output, names, values, args.force)
        else:
            manifest_path, bin_path = _write_litert(
                output, target, names, values, args.force
            )
            print(f"wrote: {manifest_path}")
            print(f"wrote: {bin_path}")
            print("replace these two files in the LiteRT bundle")
            return 0

        print(f"wrote: {output}")
        return 0
    except VoicePackError as error:
        sys.stdout.flush()
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
