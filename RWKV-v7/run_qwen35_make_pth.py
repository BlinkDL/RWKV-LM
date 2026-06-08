#!/usr/bin/env python3
"""Save HF Qwen3.5 weights as a raw text-only state_dict .pth file.

This script keeps original tensor dtypes, drops vision/MTP tensors, and strips
the text-model wrapper prefix from exported keys.

Example:

    python run_qwen35_make_pth.py --model Qwen/Qwen3.5-0.8B --output /path/to/qwen35_0_8b_text.pth
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


VISION_PREFIXES = (
    "visual.",
    "model.visual.",
    "vision_tower.",
    "model.vision_tower.",
    "multi_modal_projector.",
    "model.multi_modal_projector.",
)
LANGUAGE_MODEL_PREFIX = "model.language_model."
MTP_PREFIX = "mtp."


def main() -> None:
    args = parse_args()
    config, state_dict = load_hf_state_dict(args.model, local_files_only=args.local_files_only)
    text_state, skipped_vision, skipped_mtp = export_text_state(state_dict)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(text_state, output)
    write_metadata(
        output,
        config=config,
        state_dict=text_state,
        source_model=args.model,
        skipped_vision_tensors=skipped_vision,
        skipped_mtp_tensors=skipped_mtp,
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "kept_tensors": len(text_state),
                "skipped_vision_tensors": skipped_vision,
                "skipped_mtp_tensors": skipped_mtp,
                "dtypes": dtype_counts(text_state),
                "numel": sum(t.numel() for t in text_state.values()),
            },
            indent=2,
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B", help="HF model id or local HF model directory.")
    parser.add_argument("--output", required=True, help="Output .pth path.")
    parser.add_argument("--local-files-only", action="store_true", help="Only use files already present in the HF cache.")
    return parser.parse_args()


def load_hf_state_dict(
    model_name: str,
    local_files_only: bool = False,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    root = resolve_model_dir(model_name, local_files_only=local_files_only)
    config_path = root / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing config.json in {root}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return config, load_weight_files(root)


def resolve_model_dir(model_name: str, local_files_only: bool = False) -> Path:
    path = Path(model_name)
    if path.exists():
        return path
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=model_name,
            local_files_only=local_files_only,
            allow_patterns=(
                "config.json",
                "*.safetensors",
                "*.safetensors.index.json",
                "*.bin",
                "*.bin.index.json",
            ),
        )
    )


def load_weight_files(root: Path) -> dict[str, torch.Tensor]:
    for index_name, loader in (
        ("model.safetensors.index.json", load_safetensors_index),
        ("pytorch_model.bin.index.json", load_torch_index),
    ):
        index_path = root / index_name
        if index_path.exists():
            return loader(root, index_path)

    safetensors = sorted(root.glob("*.safetensors"))
    if safetensors:
        return load_safetensors_files(safetensors)
    bins = sorted(root.glob("*.bin"))
    if bins:
        return load_torch_files(bins)
    raise FileNotFoundError(f"no HF weight files found in {root}")


def load_safetensors_index(root: Path, index_path: Path) -> dict[str, torch.Tensor]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    return load_safetensors_files([root / name for name in sorted(set(index["weight_map"].values()))])


def load_torch_index(root: Path, index_path: Path) -> dict[str, torch.Tensor]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    return load_torch_files([root / name for name in sorted(set(index["weight_map"].values()))])


def load_safetensors_files(files: list[Path]) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    state: dict[str, torch.Tensor] = {}
    for path in files:
        state.update(load_file(str(path), device="cpu"))
    return {key: value.detach().cpu().contiguous() for key, value in state.items()}


def load_torch_files(files: list[Path]) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for path in files:
        loaded = torch.load(path, map_location="cpu")
        if isinstance(loaded, dict) and "state_dict" in loaded:
            loaded = loaded["state_dict"]
        if not isinstance(loaded, dict):
            raise TypeError(f"unsupported torch weight file: {path}")
        for key, value in loaded.items():
            if isinstance(value, torch.Tensor):
                state[str(key)] = value.detach().cpu().contiguous()
    return state


def export_text_state(state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], int, int]:
    out: dict[str, torch.Tensor] = {}
    skipped_vision = 0
    skipped_mtp = 0
    for key, value in state_dict.items():
        if is_vision_key(key):
            skipped_vision += 1
            continue
        key = strip_language_model_prefix(key)
        if is_mtp_key(key):
            skipped_mtp += 1
            continue
        if key in out:
            raise ValueError(f"duplicate exported key after prefix stripping: {key}")
        out[key] = value
    return out, skipped_vision, skipped_mtp


def is_vision_key(key: str) -> bool:
    return key.startswith(VISION_PREFIXES)


def strip_language_model_prefix(key: str) -> str:
    if key.startswith(LANGUAGE_MODEL_PREFIX):
        return key[len(LANGUAGE_MODEL_PREFIX) :]
    return key


def is_mtp_key(key: str) -> bool:
    return key.startswith(MTP_PREFIX)


def dtype_counts(state_dict: dict[str, torch.Tensor]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in state_dict.values():
        name = str(value.dtype).replace("torch.", "")
        counts[name] = counts.get(name, 0) + 1
    return counts


def write_metadata(
    output: Path,
    *,
    config: dict[str, Any],
    state_dict: dict[str, torch.Tensor],
    source_model: str,
    skipped_vision_tensors: int,
    skipped_mtp_tensors: int,
) -> None:
    metadata = {
        "source_model": source_model,
        "kept_tensors": len(state_dict),
        "skipped_vision_tensors": skipped_vision_tensors,
        "skipped_mtp_tensors": skipped_mtp_tensors,
        "stripped_prefix": LANGUAGE_MODEL_PREFIX,
        "numel": sum(t.numel() for t in state_dict.values()),
        "dtypes": dtype_counts(state_dict),
        "config": config,
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
