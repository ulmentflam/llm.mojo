"""Draw the static frame around the README demo terminal (docs/bench.gif).

Writes two 1600x900 PNGs into the output directory:

- bench-bg.png: vhs's MarginFill. A dark field lit by an ember glow (bottom
  left) and a cooler violet one (top right), a soft drop shadow where the
  terminal window will sit, and a caption naming the machine in the bottom
  margin. vhs paints the window over it.
- badge-2x.png: a transparent overlay with a pill in the bottom margin
  saying the run plays at double speed with its compile waits cut, shown
  over exactly that part.

The geometry has to match docs/bench.tape (Width, Height, Margin,
BorderRadius). The caption is read from the host, so it always names the
machine that recorded the run.

Usage: python bench_gif_frame.py <out-dir> "<caption>"
"""

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1600, 900
MARGIN = 72  # docs/bench.tape: Set Margin
RADIUS = 16  # docs/bench.tape: Set BorderRadius

BASE = (9, 10, 15)
EMBER = (255, 94, 40)
VIOLET = (124, 58, 237)
CAPTION = (148, 155, 170)
ACCENT = (255, 106, 50)


def font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        Path.home() / "Library/Fonts/IBMPlexMono-Regular.otf",
        Path("/System/Library/Fonts/Menlo.ttc"),
    ]
    return ImageFont.truetype(str(next(p for p in candidates if p.exists())), size)


def glow(
    center: tuple[float, float], radius: float, color, strength: float
) -> Image.Image:
    # A blurred disc is a cheap, smooth radial falloff.
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    cx, cy = center
    ImageDraw.Draw(layer).ellipse(
        (cx - radius, cy - radius, cx + radius, cy + radius),
        fill=(*color, int(255 * strength)),
    )
    return layer.filter(ImageFilter.GaussianBlur(radius * 0.45))


def background(caption: str) -> Image.Image:
    img = Image.new("RGBA", (W, H), (*BASE, 255))
    img = Image.alpha_composite(img, glow((W * 0.08, H * 1.02), W * 0.42, EMBER, 0.42))
    img = Image.alpha_composite(
        img, glow((W * 0.96, -H * 0.05), W * 0.36, VIOLET, 0.30)
    )

    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle(
        (MARGIN, MARGIN + 14, W - MARGIN, H - MARGIN + 14), RADIUS, fill=(0, 0, 0, 200)
    )
    img = Image.alpha_composite(img, shadow.filter(ImageFilter.GaussianBlur(22)))

    # A hairline just outside the window so its edge reads on the dark field.
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(
        (MARGIN - 1, MARGIN - 1, W - MARGIN, H - MARGIN),
        RADIUS + 1,
        outline=(255, 255, 255, 34),
        width=1,
    )
    baseline = H - MARGIN // 2 + 9
    draw.text(
        (MARGIN + 4, baseline), "llm.mojo", font=font(24), fill=ACCENT, anchor="ls"
    )
    draw.text(
        (MARGIN + 4 + font(24).getlength("llm.mojo  "), baseline),
        caption,
        font=font(24),
        fill=CAPTION,
        anchor="ls",
    )
    return img.convert("RGB")


def badge() -> Image.Image:
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    label = "2x speed · compile waits cut"
    f = font(22)
    width = f.getlength(label)
    right, cy = W - MARGIN - 4, H - MARGIN // 2
    box = (right - width - 32, cy - 19, right, cy + 19)
    draw.rounded_rectangle(box, 19, fill=(*ACCENT, 40), outline=(*ACCENT, 170), width=2)
    draw.text(
        ((box[0] + box[2]) / 2, cy), label, font=f, fill=(255, 196, 170), anchor="mm"
    )
    return img


def main() -> None:
    out, caption = Path(sys.argv[1]), sys.argv[2]
    out.mkdir(parents=True, exist_ok=True)
    background(caption).save(out / "bench-bg.png")
    badge().save(out / "badge-2x.png")


if __name__ == "__main__":
    main()
