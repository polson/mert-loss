from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torchaudio

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mert_loss import mert_mse


def candidate_score(path: Path) -> tuple[int, str]:
    name = path.name.lower()
    if name == "mixture.wav":
        return (0, str(path))
    if name == "drums.wav":
        return (1, str(path))
    return (2, str(path))


def rms(waveform: torch.Tensor) -> float:
    mono = waveform.mean(dim=0)
    return mono.float().pow(2).mean().sqrt().item()


def find_non_silent_clip(
    root: Path,
    duration_s: float,
    threshold: float,
    max_files: int | None = None,
) -> tuple[Path, torch.Tensor, int, int, float]:
    paths = sorted(root.rglob("*.wav"), key=candidate_score)
    if max_files is not None:
        paths = paths[:max_files]

    if not paths:
        raise FileNotFoundError(f"No .wav files found under {root}")

    for path in paths:
        info = torchaudio.info(str(path))
        chunk_frames = int(duration_s * info.sample_rate)
        if info.num_frames < chunk_frames:
            continue

        hop = max(1, chunk_frames // 2)
        max_offset = info.num_frames - chunk_frames

        for frame_offset in range(0, max_offset + 1, hop):
            waveform, sample_rate = torchaudio.load(
                str(path),
                frame_offset=frame_offset,
                num_frames=chunk_frames,
            )
            chunk_rms = rms(waveform)
            if chunk_rms >= threshold:
                return path, waveform, sample_rate, frame_offset, chunk_rms

    raise RuntimeError(
        f"Could not find a {duration_s:.1f}s non-silent clip under {root} "
        f"with RMS >= {threshold}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test mert_loss on a real audio clip.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/phil/Audiosep/Datasets/combined"),
        help="Root directory to scan for wav files.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Clip duration in seconds.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-3,
        help="Minimum RMS required for a chunk to count as non-silent.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=200,
        help="Maximum number of wav files to scan.",
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Only find and print a suitable clip without running MERT.",
    )
    args = parser.parse_args()

    path, waveform, sample_rate, frame_offset, chunk_rms = find_non_silent_clip(
        root=args.root,
        duration_s=args.duration,
        threshold=args.threshold,
        max_files=args.max_files,
    )

    start_s = frame_offset / sample_rate
    end_s = start_s + args.duration

    print(f"Selected file: {path}")
    print(f"Sample rate:   {sample_rate}")
    print(f"Shape:         {tuple(waveform.shape)}  # [C, T]")
    print(f"Clip window:   {start_s:.2f}s -> {end_s:.2f}s")
    print(f"Clip RMS:      {chunk_rms:.6f}")

    if args.scan_only:
        return

    if sample_rate != 44_100:
        raise RuntimeError(
            f"This smoke test expects a 44.1 kHz file to match the convenience API, got {sample_rate}."
        )

    # torchaudio.load returns [C, T]. The library API supports [T], [B, T], or [B, C, T],
    # so wrap the clip in a batch dimension to preserve the channel dimension.
    target = waveform.unsqueeze(0)

    torch.manual_seed(0)
    pred_same = target.clone().requires_grad_(True)
    pred_noisy = (target * 0.98 + 0.01 * torch.randn_like(target)).clamp(-1.0, 1.0)
    pred_noisy.requires_grad_(True)

    # --- VRAM measurement setup ---
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        vram_before = torch.cuda.memory_allocated() / (1024 ** 2)
    # --- end VRAM setup ---

    print("\nRunning MERT loss...")
    try:
        same_loss = mert_mse(pred_same, target)
        noisy_loss = mert_mse(pred_noisy, target)
    except ImportError as exc:
        raise SystemExit(
            "Missing runtime dependency while loading MERT. "
            "Install the project dependencies first, e.g. `pip install -e .`"
        ) from exc

    same_loss.backward()
    noisy_loss.backward()

    # --- VRAM measurement report ---
    if torch.cuda.is_available():
        vram_after = torch.cuda.memory_allocated() / (1024 ** 2)
        vram_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        vram_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
        print(f"\n--- VRAM Usage ({device_name}) ---")
        print(f"Before forward:  {vram_before:8.2f} MB")
        print(f"After backward:  {vram_after:8.2f} MB")
        print(f"Peak allocated:  {vram_peak:8.2f} MB")
        print(f"Peak reserved:   {vram_reserved:8.2f} MB")
    # --- end VRAM report ---

    print(f"same clip loss:   {same_loss.item():.8f}")
    print(f"noisy clip loss:  {noisy_loss.item():.8f}")
    print(f"same grad ok:     {pred_same.grad is not None}")
    print(f"noisy grad ok:    {pred_noisy.grad is not None}")


if __name__ == "__main__":
    main()
