"""Draw the frame around the llm.c comparison video (docs/bench-vs-llmc.*).

Writes 1920x1080 PNGs into the output directory, in the README demo's look
(scripts/bench_gif_frame.py, whose glow and font helpers this reuses):

- compare-bg.png: the whole canvas. Two windows side by side, each with a soft
  shadow and hairline; a line across the top stating the terms of the
  comparison; under each window a caption naming the trainer and the machine.
- compare-bg-left.png, compare-bg-right.png: its halves, the MarginFill of
  each pane's vhs tape (docs/bench/compare-llmc.tape, compare-llmm.tape), so
  the two recordings stack back into the canvas seamlessly.
- compare-speed.png: a transparent overlay with a "<N>x speed" pill, laid over
  the GIF only (the MP4 plays in real time).

The geometry has to match the compare tapes (Width, Height, Margin,
BorderRadius).

Usage: python compare_frame.py <out-dir> "<machine>" <threads> <gif speed>
"""

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.bench_gif_frame import (  # noqa: E402
    ACCENT,
    BASE,
    CAPTION,
    EMBER,
    VIOLET,
    font,
)

W, H = 1920, 1080
PANE = W // 2
MARGIN = 56  # docs/bench/compare-*.tape: Set Margin
RADIUS = 16  # docs/bench/compare-*.tape: Set BorderRadius
TEXT_SIZE = 26


def glow(center, radius, color, strength) -> Image.Image:
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    cx, cy = center
    ImageDraw.Draw(layer).ellipse(
        (cx - radius, cy - radius, cx + radius, cy + radius),
        fill=(*color, int(255 * strength)),
    )
    return layer.filter(ImageFilter.GaussianBlur(radius * 0.45))


def windows() -> list[tuple[int, int, int, int]]:
    return [(x0 + MARGIN, MARGIN, x0 + PANE - MARGIN, H - MARGIN) for x0 in (0, PANE)]


def background(machine: str, threads: str) -> Image.Image:
    img = Image.new("RGBA", (W, H), (*BASE, 255))
    img = Image.alpha_composite(img, glow((W * 0.06, H * 1.05), W * 0.36, EMBER, 0.42))
    img = Image.alpha_composite(img, glow((W * 0.97, -H * 0.08), W * 0.3, VIOLET, 0.3))

    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    for x0, y0, x1, y1 in windows():
        ImageDraw.Draw(shadow).rounded_rectangle(
            (x0, y0 + 14, x1, y1 + 14), RADIUS, fill=(0, 0, 0, 200)
        )
    img = Image.alpha_composite(img, shadow.filter(ImageFilter.GaussianBlur(22)))

    draw = ImageDraw.Draw(img)
    for x0, y0, x1, y1 in windows():
        draw.rounded_rectangle(
            (x0 - 1, y0 - 1, x1, y1),
            RADIUS + 1,
            outline=(255, 255, 255, 34),
            width=1,
        )

    f = font(TEXT_SIZE)
    terms = (
        f"same B=4 T=64 run · same {threads} CPU threads · "
        "recorded separately, synced at launch"
    )
    draw.text((W / 2, MARGIN / 2 + 2), terms, font=f, fill=CAPTION, anchor="mm")
    baseline = H - MARGIN // 2 + 10
    for (x0, _, _, _), name, detail in zip(
        windows(),
        ("llm.c", "llm.mojo"),
        (f"C, OpenMP · {machine}", f"Mojo · {machine}"),
    ):
        draw.text((x0 + 4, baseline), name, font=f, fill=ACCENT, anchor="ls")
        draw.text(
            (x0 + 4 + f.getlength(name + "  "), baseline),
            detail,
            font=f,
            fill=CAPTION,
            anchor="ls",
        )
    return img.convert("RGB")


def speed_badge(speed: str) -> Image.Image:
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    f = font(TEXT_SIZE - 2)
    label = f"{speed}x speed"
    width = f.getlength(label)
    right, cy = W - MARGIN - 4, H - MARGIN // 2
    box = (right - width - 36, cy - 20, right, cy + 20)
    draw.rounded_rectangle(box, 20, fill=(*ACCENT, 40), outline=(*ACCENT, 170), width=2)
    draw.text(
        ((box[0] + box[2]) / 2, cy), label, font=f, fill=(255, 196, 170), anchor="mm"
    )
    return img


def main() -> None:
    out, machine, threads, speed = Path(sys.argv[1]), *sys.argv[2:5]
    out.mkdir(parents=True, exist_ok=True)
    bg = background(machine, threads)
    bg.save(out / "compare-bg.png")
    bg.crop((0, 0, PANE, H)).save(out / "compare-bg-left.png")
    bg.crop((PANE, 0, W, H)).save(out / "compare-bg-right.png")
    speed_badge(speed).save(out / "compare-speed.png")


if __name__ == "__main__":
    main()
