"""Generate an animated GIF for Week2 MVP demo."""

from __future__ import annotations

from pathlib import Path
import tempfile

import imageio.v2 as imageio

from week2.main import export_frames


FRAME_COUNT = 90
FPS = 30
ANGLE_STEP = 4.0
OUTPUT_GIF = Path("assets/week2/mvp_demo.gif")


def make_gif() -> Path:
    """Export frames and assemble them into a GIF file."""
    OUTPUT_GIF.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="week2_frames_") as tmp_dir:
        frame_paths = export_frames(
            output_dir=Path(tmp_dir),
            frame_count=FRAME_COUNT,
            angle_step=ANGLE_STEP,
        )
        frames = [imageio.imread(frame_path) for frame_path in frame_paths]
        imageio.mimsave(OUTPUT_GIF, frames, fps=FPS, loop=0)

    return OUTPUT_GIF


if __name__ == "__main__":
    gif_path = make_gif()
    print(f"GIF generated: {gif_path}")
