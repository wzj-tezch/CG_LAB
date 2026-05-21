from pathlib import Path

from PIL import Image, ImageSequence

SRC = Path(r"D:\桌面\屏幕录制 2026-05-21 180539.gif")
DST = Path(__file__).resolve().parent / "assets" / "interactive_demo_20260521.gif"


def main():
    frames = []
    durations = []
    with Image.open(SRC) as im:
        w, h = im.size
        target_w = 720
        target_h = max(1, int(h * target_w / w))
        for i, frame in enumerate(ImageSequence.Iterator(im)):
            if i % 3 != 0:
                continue
            rgb = frame.convert("RGB").resize((target_w, target_h), Image.Resampling.LANCZOS)
            pal = rgb.quantize(colors=64, method=Image.Quantize.MEDIANCUT)
            frames.append(pal)
            durations.append(frame.info.get("duration", im.info.get("duration", 80)))

    if not frames:
        raise RuntimeError("No frames decoded from source GIF")

    frames[0].save(
        DST,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(f"saved {DST} ({DST.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
