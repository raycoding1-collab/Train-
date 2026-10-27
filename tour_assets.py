"""Web sizes of one tour panorama: python3 tour_assets.py <master.png> <spot id>

tour/pano/<id>.webp       full resolution (laptops and desktops)
tour/pano/<id>_mid.webp   3072 x 1536 (phones, and the first sharp frame while the full one loads)
tour/pano/<id>_thumb.jpg  512 x 256 (places sheet and map)
"""
import os
import sys

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))


def main(master, sid):
    out = os.path.join(HERE, "tour", "pano")
    os.makedirs(out, exist_ok=True)
    im = Image.open(master).convert("RGB")
    im.save(os.path.join(out, f"{sid}.webp"), quality=84, method=6)
    im.resize((3072, 1536), Image.LANCZOS).save(os.path.join(out, f"{sid}_mid.webp"), quality=82, method=6)
    im.resize((512, 256), Image.LANCZOS).save(os.path.join(out, f"{sid}_thumb.jpg"), quality=82, optimize=True)
    sizes = [os.path.getsize(os.path.join(out, f)) / 1e6 for f in (f"{sid}.webp", f"{sid}_mid.webp")]
    print(f"[assets] {sid}: full {sizes[0]:.1f} MB, mid {sizes[1]:.1f} MB", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
