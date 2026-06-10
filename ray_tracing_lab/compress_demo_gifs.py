"""Generate README preview GIFs while keeping full-size originals."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageSequence

ASSETS = Path(__file__).resolve().parent / "assets"
ORIGINAL = ASSETS / "original"

FILES = (
    "ray_tracing_demo.gif",
    "optional_features_demo.gif",
)


def compress_gif(src: Path, dst: Path, *, scale: float = 0.55, step: int = 2, colors: int = 128) -> None:
    im = Image.open(src)
    base_duration = im.info.get("duration", 40)
    frames: list[Image.Image] = []
    durations: list[int] = []

    for index, frame in enumerate(ImageSequence.Iterator(im)):
        if index % step:
            continue
        rgba = frame.convert("RGBA")
        width, height = rgba.size
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        resized = rgba.resize(new_size, Image.Resampling.LANCZOS)
        paletted = resized.convert("P", palette=Image.ADAPTIVE, colors=colors)
        frames.append(paletted)
        durations.append(int(frame.info.get("duration", base_duration) * step))

    if not frames:
        raise RuntimeError(f"No frames extracted from {src}")

    frames[0].save(
        dst,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=im.info.get("loop", 0),
        optimize=True,
        disposal=2,
    )


def main() -> None:
    ORIGINAL.mkdir(parents=True, exist_ok=True)

    for name in FILES:
        original_dst = ORIGINAL / name
        if not original_dst.exists():
            raise FileNotFoundError(f"Missing original GIF: {original_dst}")

        preview_dst = ASSETS / name.replace(".gif", "_preview.gif")
        compress_gif(original_dst, preview_dst)
        print(
            f"{name}: original {original_dst.stat().st_size / 1024 / 1024:.2f} MB -> "
            f"preview {preview_dst.stat().st_size / 1024 / 1024:.2f} MB"
        )


if __name__ == "__main__":
    main()
