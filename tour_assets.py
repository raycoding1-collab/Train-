"""Web sizes of one tour panorama: python3 tour_assets.py <master.png> <spot id>

tour/pano/<id>.webp       full resolution (laptops and desktops)
tour/pano/<id>_mid.webp   3072 x 1536 (phones, and the first sharp frame while the full one loads)
tour/pano/<id>_thumb.jpg  512 x 256 (places sheet and map)
tour/ready.json           the spots whose panorama and depth map both exist, for the viewer
"""
import json
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
    write_ready()


def write_ready():
    spots = json.load(open(os.path.join(HERE, "tour", "spots.json")))["spots"]
    ready = [s["id"] for s in spots if all(os.path.exists(os.path.join(HERE, "tour", p)) for p in
                                           (f"pano/{s['id']}.webp", f"pano/{s['id']}_mid.webp", f"depth/{s['id']}.png"))]
    json.dump(ready, open(os.path.join(HERE, "tour", "ready.json"), "w"))


if __name__ == "__main__":
    if sys.argv[1:] == ["--ready"]:
        write_ready()
    else:
        main(sys.argv[1], sys.argv[2])
