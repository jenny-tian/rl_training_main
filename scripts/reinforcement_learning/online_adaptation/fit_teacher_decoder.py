"""Fit a fixed latent steering decoder from a validated teacher dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rl_training.online_adaptation.teacher_dataset import load_teacher_dataset, select_safe_rows
from rl_training.online_adaptation.teacher_steering import (
    fit_low_rank_teacher_decoder,
    save_teacher_decoder,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--code-dim", type=int, required=True)
    parser.add_argument("--action-limit", type=float, nargs="+", required=True)
    parser.add_argument("--sealed-seed", type=int, action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.code_dim <= 0:
        raise ValueError("--code-dim must be positive")
    limits = torch.tensor(args.action_limit, dtype=torch.float32)
    dataset = select_safe_rows(
        load_teacher_dataset(args.dataset, sealed_seeds=tuple(args.sealed_seed))
    )
    residual = dataset.residual().to(dtype=limits.dtype)
    if residual.shape[1] != limits.shape[0]:
        raise ValueError("--action-limit count must match the teacher action dimension")
    fit = fit_low_rank_teacher_decoder(residual, limits, args.code_dim)
    save_teacher_decoder(
        Path(args.output),
        fit,
        limits,
        source_metadata={
            "dataset_schema": dataset.metadata.get("dataset_schema", "rgls_teacher_dataset_v1"),
            "dataset_metadata": dataset.metadata,
            "safe_rows": dataset.size,
        },
    )
    print(
        f"saved {args.output}: rows={dataset.size} code_dim={args.code_dim} "
        f"normalized_mse={fit.normalized_mse:.6g} explained_variance={fit.explained_variance:.6g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
