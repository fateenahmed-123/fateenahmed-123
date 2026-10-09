"""Render ascii.rest pieces (via tools/export.ts) into the profile's animated images.

usage: python3 tools/render.py banner|avatar
"""
import base64, json, subprocess, sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"


def frames(piece, fps, seconds, t0=0.0):
    out = subprocess.run(
        ["node", str(ROOT / "tools/export.ts"), piece, str(fps), str(seconds), str(t0)],
        check=True, capture_output=True, text=True,
    ).stdout
    return [json.loads(line) for line in out.splitlines()]


def hexrgb(s):
    return tuple(int(s[i:i + 2], 16) for i in (1, 3, 5))


def disc(size, cover):
    # a filled circle covering `cover` of a size x size cell
    if cover <= 0:
        return np.zeros((size, size), bool)
    r = size / 2 * cover ** 0.5
    c = (np.arange(size) + 0.5) - size / 2
    return (c[:, None] ** 2 + c[None, :] ** 2) <= r * r


# --- banner: ocean sunset, halftone dots in palette colour ----------------
def banner():
    meta = json.loads(subprocess.run(
        ["node", "-e", "import('./tools/pieces/ocean-sunset.ts').then(m=>console.log(JSON.stringify(m.meta)))"],
        cwd=ROOT, check=True, capture_output=True, text=True).stdout)
    W, H = meta["cols"], meta["rows"]
    CELL, FPS, LOOP, FADE, T0 = 6, 12, 8.0, 1.5, 20.0
    # F(T0 - FADE) .. F(T0 + LOOP): the tail of the loop dissolves into its head
    fs = frames("ocean-sunset", FPS, LOOP + FADE, T0 - FADE)
    nf, nfade = int(LOOP * FPS), int(FADE * FPS)
    steps = {c: i for i, c in enumerate(" ·•●")}
    masks = np.stack([disc(CELL, c) for c in (0, 0.3, 0.6, 1)])  # (4, CELL, CELL)

    def cells(rec):
        st = np.array([steps[c] for c in rec["text"].replace("\n", "")], np.uint8).reshape(H, W)
        col = np.frombuffer(base64.b64decode(rec["color"]), np.uint8).reshape(H, W)
        return st, col

    rng = np.random.default_rng(7)
    thresh = rng.random((H, W))
    pal = [hexrgb(meta["ground"])] + [hexrgb(c) for c in meta["palette"]]
    flat_pal = [v for c in pal for v in c] + [0] * (768 - 3 * len(pal))
    ys, xs = np.mgrid[0:H * CELL, 0:W * CELL]
    images = []
    for i in range(nf):
        st, col = cells(fs[nfade + i])
        if i >= nf - nfade:  # dither-dissolve toward the loop's first frame
            k = (i - (nf - nfade) + 1) / (nfade + 1)
            st0, col0 = cells(fs[i - (nf - nfade)])
            pick = thresh < k
            st, col = np.where(pick, st0, st), np.where(pick, col0, col)
        S = np.repeat(np.repeat(st, CELL, 0), CELL, 1)
        C = np.repeat(np.repeat(col, CELL, 0), CELL, 1)
        on = masks[S, ys % CELL, xs % CELL]
        idx = np.where(on, C + 1, 0).astype(np.uint8)
        im = Image.frombytes("P", (idx.shape[1], idx.shape[0]), idx.tobytes())
        im.putpalette(flat_pal)
        images.append(im)
    save(images, "banner", 1000 / FPS)


# --- avatar: three-body, glyphs drawn as glowing marks on a square --------
def avatar():
    COLS, ROWS, FPS = 65, 15, 30
    LOOP = 11 / 3  # three equal bodies: a third of a lap returns the same picture
    SIZE, SS = 460, 3  # output px, supersampling
    fs = frames("three-body", FPS, LOOP)
    big = SIZE * SS
    cw = big * 0.98 / COLS  # the eight spans nearly the full width
    ch = cw * 2
    ox, oy = (big - cw * COLS) / 2, (big - ch * ROWS) / 2
    ground = np.array(hexrgb("#0b0817"), float)
    orbit_c = np.array(hexrgb("#90327c"), float)
    trail_c = np.array(hexrgb("#f2804f"), float)
    tail_c = np.array(hexrgb("#b23c7c"), float)
    head_c = np.array(hexrgb("#fff5d8"), float)
    glow_c = np.array(hexrgb("#ffb44a"), float)
    # glyph -> list of (dy within cell 0..1, radius in cell widths, colour)
    glyph = {
        "'": [(0.3, 0.24, orbit_c)], "·": [(0.5, 0.24, orbit_c)], ".": [(0.7, 0.24, orbit_c)],
        ":": [(0.3, 0.26, trail_c), (0.7, 0.26, trail_c)],
        "o": [(0.5, 0.42, trail_c)], "O": [(0.5, 0.75, head_c)],
    }
    yy, xx = np.mgrid[0:big, 0:big].astype(np.float32)
    images = []
    for n, rec in enumerate(fs):
        rgb = np.broadcast_to(ground, (big, big, 3)).copy()
        heads = []
        lines = rec["text"].split("\n")
        for r, line in enumerate(lines):
            for c, g in enumerate(line):
                if g not in glyph:
                    continue
                for dy, rad, colr in glyph[g]:
                    # trail dots fade out the farther they are from a head, orbit dots stay dim
                    if g == "·" and colr is orbit_c and _near_trail(lines, r, c):
                        colr = tail_c
                    cx, cy = ox + (c + 0.5) * cw, oy + (r + dy) * ch
                    R = rad * cw
                    x0, x1 = int(cx - R) - 1, int(cx + R) + 2
                    y0, y1 = int(cy - R) - 1, int(cy + R) + 2
                    d = np.hypot(xx[y0:y1, x0:x1] - cx, yy[y0:y1, x0:x1] - cy)
                    m = (d <= R)[..., None]
                    rgb[y0:y1, x0:x1] = np.where(m, colr, rgb[y0:y1, x0:x1])
                    if g == "O":
                        heads.append((cx, cy))
        # a warm glow round each body, under the marks
        glow = np.zeros((big, big), np.float32)
        for cx, cy in heads:
            glow += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (cw * 1.3) ** 2))
        glow = (np.clip(glow, 0, 1) * 0.5)[..., None]
        dark = (np.abs(rgb - ground).sum(-1, keepdims=True) < 1)
        rgb = np.where(dark, rgb + glow * (glow_c - rgb), rgb)
        im = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).resize((SIZE, SIZE), Image.LANCZOS)
        images.append(im)
    images[len(images) // 5].save(ASSETS / "avatar.png")
    pal = images[0].quantize(64, method=Image.MEDIANCUT)
    save([im.quantize(palette=pal, dither=Image.NONE) for im in images], "avatar", 1000 / FPS)


def _near_trail(lines, r, c):
    # a "·" that is part of a comet's tail sits beside a ":" on the same row
    row = lines[r]
    return any(0 <= c + d < len(row) and row[c + d] == ":" for d in (-2, -1, 1, 2))


def save(images, name, ms):
    images[0].save(ASSETS / f"{name}.gif", save_all=True, append_images=images[1:],
                   duration=round(ms), loop=0, optimize=True, disposal=1)
    print(name, (ASSETS / f"{name}.gif").stat().st_size // 1024, "KB,", len(images), "frames")


if __name__ == "__main__":
    {"banner": banner, "avatar": avatar}[sys.argv[1]]()
