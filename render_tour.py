"""Render the photoreal 360-degree tour (tour/index.html) with Cycles.

At every spot in tour/spots.json:
  * an equirectangular colour image (path traced, day light, no people), handed to tour_assets.py
    for the web sizes: tour/pano/<id>.webp (full), <id>_mid.webp (3072 wide), <id>_thumb.jpg
  * a depth map (distance from the eye) for the gliding transitions between spots: tour/depth/<id>.png,
    inverse depth in 16 bits split over the red (high byte) and green (low byte) channels

    python render_tour.py --pass depth                 # all depth maps (about a minute each)
    python render_tour.py --pass color --resume        # all colour images, in the order of spots.json
    python render_tour.py --pass color --only pool-north --res 1536 768 --samples 8   # quick look

The image centre faces +X (Blender) and the image runs clockwise seen from above, like the existing
panoramas; the viewer maps it the same way. Finished spots are skipped with --resume.
"""
import argparse
import json
import math
import os
import struct
import subprocess
import sys
import time
import zlib

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import station_modes  # noqa: E402

NEAR, FAR = 0.3, 3000.0          # inverse-depth encoding range in metres; sky reads as FAR
HIDDEN = ("People",)             # the tour is walked alone
VOLUMES = ("Atmosphere", "AtmosphereInner")
SEE_THROUGH = ("Glass",)         # transparent in the depth maps; water keeps its surface


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend", default=os.path.join(HERE, "station.blend"))
    ap.add_argument("--spots", default=os.path.join(HERE, "tour", "spots.json"))
    ap.add_argument("--pass", dest="kind", choices=("color", "depth"), required=True)
    ap.add_argument("--only", default="", help="comma-separated spot ids")
    ap.add_argument("--res", type=int, nargs=2, default=(6144, 3072))
    ap.add_argument("--depth-res", type=int, nargs=2, default=(1024, 512))
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--master", default=os.path.join(HERE, "renders", "tour"), help="lossless PNG masters")
    ap.add_argument("--resume", action="store_true")
    return ap.parse_args(argv)


def pano_camera(scene, eye):
    cd = bpy.data.cameras.get("TourCam") or bpy.data.cameras.new("TourCam")
    cd.type, cd.panorama_type = "PANO", "EQUIRECTANGULAR"
    cd.clip_start, cd.clip_end = 0.05, 6000.0
    cam = bpy.data.objects.get("TourCam") or bpy.data.objects.new("TourCam", cd)
    if cam.name not in scene.collection.objects:
        scene.collection.objects.link(cam)
    cam.location = eye
    cam.rotation_euler = (math.pi / 2, 0.0, -math.pi / 2)
    scene.camera = cam


def write_png_rgb(path, rgb):
    """Minimal PNG writer (8-bit RGB) so the depth maps need nothing beyond numpy."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)  # noqa: E731
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def setup_depth(scene, args):
    """Every surface emits its distance from the camera, except glass, which the eye looks through (so what lies
    behind it keeps its own depth); volumes, sky and denoising off; one exact sample per pixel.
    Runs in its own process, so the materials are rewritten in place."""
    for m in bpy.data.materials:
        nt = m.node_tree
        if nt is None:
            continue
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        if m.name in SEE_THROUGH:
            nt.links.new(nt.nodes.new("ShaderNodeBsdfTransparent").outputs["BSDF"], out.inputs["Surface"])
            continue
        cam, em = nt.nodes.new("ShaderNodeCameraData"), nt.nodes.new("ShaderNodeEmission")
        nt.links.new(cam.outputs["View Distance"], em.inputs["Color"])
        nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    for n in VOLUMES:
        if n in bpy.data.objects:
            bpy.data.objects[n].hide_render = True
    world = bpy.data.worlds.new("TourDepthWorld")          # the sky emits nothing: it reads as FAR
    world.color = (0, 0, 0)
    if world.node_tree:
        for n in world.node_tree.nodes:
            if n.type == "BACKGROUND":
                n.inputs["Strength"].default_value = 0.0
    scene.world = world
    c = scene.cycles
    c.samples, c.use_adaptive_sampling, c.use_denoising = 1, False, False
    c.max_bounces, c.transparent_max_bounces = 0, 16
    c.filter_width = 0.01
    c.seed = 0
    scene.render.resolution_x, scene.render.resolution_y = args.depth_res
    im = scene.render.image_settings
    im.media_type = "IMAGE"
    im.file_format, im.color_depth, im.exr_codec = "OPEN_EXR", "32", "ZIP"


def depth_pass(scene, spot, eye, args):
    out = os.path.join(HERE, "tour", "depth", f"{spot['id']}.png")
    if args.resume and os.path.exists(out):
        print(f"[depth] {spot['id']} already done", flush=True)
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    exr = os.path.join(args.master, f"{spot['id']}_depth.exr")
    pano_camera(scene, eye)
    scene.render.filepath = exr
    t = time.time()
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(exr, check_existing=False)
    w, h = img.size
    px = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(px)
    bpy.data.images.remove(img)
    d = px[0::4].reshape(h, w)[::-1]                                  # rows top to bottom
    inv = np.where(d > 0.2, 1.0 / np.maximum(d, NEAR), 1.0 / FAR)   # no surface is that close to the eye: below it is sky
    q = np.clip((inv - 1 / FAR) / (1 / NEAR - 1 / FAR), 0, 1)
    q16 = np.round(q * 65535).astype(np.uint32)
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[..., 0], rgb[..., 1] = q16 >> 8, q16 & 255
    write_png_rgb(out, rgb)
    nadir = float(d[-1, w // 2])
    print(f"[depth] {spot['id']} -> {out} ({time.time() - t:.1f}s, floor below {nadir:.2f} m)", flush=True)


def color_pass(scene, spot, eye, args):
    sid = spot["id"]
    final = os.path.join(HERE, "tour", "pano", f"{sid}.webp")
    if args.resume and os.path.exists(final):
        print(f"[color] {sid} already done", flush=True)
        return
    master = os.path.join(args.master, f"{sid}.png")
    pano_camera(scene, eye)
    scene.render.filepath = master
    t = time.time()
    bpy.ops.render.render(write_still=True)
    print(f"[color] {sid} rendered ({time.time() - t:.1f}s)", flush=True)
    subprocess.run(["python3", os.path.join(HERE, "tour_assets.py"), master, sid], check=True)


def main():
    args = parse_args()
    bpy.ops.wm.open_mainfile(filepath=args.blend)
    scene = bpy.context.scene
    data = json.load(open(args.spots))
    only = set(filter(None, args.only.split(",")))
    spots = [s for s in data["spots"] if not only or s["id"] in only]
    os.makedirs(args.master, exist_ok=True)
    station_modes.apply_mode(scene, "day")
    for n in HIDDEN:
        if n in bpy.data.objects:
            bpy.data.objects[n].hide_render = True
    r = scene.render
    r.engine, r.resolution_percentage, r.use_motion_blur = "CYCLES", 100, False
    r.use_persistent_data = True                       # keep the synced scene between spots
    scene.cycles.device = "CPU"
    if args.kind == "depth":
        setup_depth(scene, args)
    else:
        r.resolution_x, r.resolution_y = args.res
        c = scene.cycles
        c.samples, c.use_adaptive_sampling, c.adaptive_threshold = args.samples, True, 0.02
        c.use_denoising, c.denoiser = True, "OPENIMAGEDENOISE"
        im = r.image_settings
        im.media_type = "IMAGE"
        im.file_format, im.color_depth, im.compression = "PNG", "8", 15
    for spot in spots:
        x, y, z = spot["pos"]
        eye = (x, y, z + data["eye"])
        (depth_pass if args.kind == "depth" else color_pass)(scene, spot, eye, args)


if __name__ == "__main__":
    main()
