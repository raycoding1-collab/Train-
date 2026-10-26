#!/usr/bin/env python3
"""Render Emerald Valley Station.

    blender -b station.blend --python render.py -- --cam CAM_Aerial --mode day --res 960 540 --samples 16 \
        --out renders/{cam}_{mode}.png
    python render.py --cam all --mode both          # with the `bpy` module; opens station.blend next to this file

Cameras: CAM_Aerial, CAM_Plaza, CAM_Valley, CAM_Platform, CAM_Roof, CAM_TrailUp, CAM_Fly (or "all").
Modes: day, dusk (or "both"). --anim renders the scene's frame range (1-240) of each camera to MP4.
Cycles on the CPU with OpenImageDenoise is the default; --engine EEVEE needs a GPU / display.
"""
import argparse
import math
import os
import shutil
import subprocess
import sys
import time

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import station_modes  # noqa: E402

CAMERAS = ["CAM_Aerial", "CAM_Plaza", "CAM_Valley", "CAM_Platform", "CAM_Roof", "CAM_TrailUp", "CAM_Fly"]


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blend", default=os.path.join(HERE, "station.blend"))
    ap.add_argument("--cam", default="CAM_Aerial", help="camera name, comma separated list, or 'all'")
    ap.add_argument("--mode", default="day", choices=["day", "dusk", "both"])
    ap.add_argument("--res", nargs=2, type=int, default=None, metavar=("W", "H"),
                    help="output size (default 1920 1080, or 3072 1536 with --pano)")
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--out", default=os.path.join(HERE, "renders", "{cam}_{mode}.png"))
    ap.add_argument("--anim", action="store_true", help="render the frame range to MP4")
    ap.add_argument("--frames", nargs=2, type=int, metavar=("START", "END"), help="frame range for --anim")
    ap.add_argument("--frame", type=int, help="frame for stills (default: the camera's still_frame)")
    ap.add_argument("--engine", default="CYCLES", choices=["CYCLES", "EEVEE"])
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = all)")
    ap.add_argument("--sun-az", type=float, help="override sun azimuth (compass degrees)")
    ap.add_argument("--sun-el", type=float, help="override sun elevation (degrees)")
    ap.add_argument("--no-motion-blur", action="store_true")
    ap.add_argument("--pano", action="store_true",
                    help="render 360-degree equirectangular panoramas at the PANO_* spots (or those named by --cam)")
    args = ap.parse_args(argv)
    if args.res is None:
        args.res = [3072, 1536] if args.pano else [1920, 1080]
    return args


def setup_engine(scene, args):
    r = scene.render
    r.resolution_x, r.resolution_y = args.res
    r.resolution_percentage = 100
    r.threads_mode = "FIXED" if args.threads else "AUTO"
    if args.threads:
        r.threads = args.threads
    if args.no_motion_blur:
        r.use_motion_blur = False
    if args.engine == "EEVEE":
        r.engine = "BLENDER_EEVEE"
        scene.eevee.taa_render_samples = args.samples
    else:
        r.engine = "CYCLES"
        scene.cycles.device = "CPU"
        scene.cycles.samples = args.samples
        scene.cycles.use_denoising = True
        scene.cycles.denoiser = "OPENIMAGEDENOISE"


def output_path(pattern, cam, mode, ext=None):
    path = pattern.format(cam=cam, mode=mode)
    if ext:
        path = os.path.splitext(path)[0] + ext
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return os.path.abspath(path)


def set_image_format(scene, path):
    im = scene.render.image_settings
    im.media_type = "IMAGE"
    if path.lower().endswith((".jpg", ".jpeg")):
        im.file_format, im.quality = "JPEG", 92
    else:
        im.file_format, im.color_depth = "PNG", "8"


def render_still(scene, cam, mode, args):
    ob = bpy.data.objects[cam]
    scene.camera = ob
    scene.frame_set(args.frame or int(ob.get("still_frame", 1)))
    scene.render.filepath = output_path(args.out, cam, mode)
    set_image_format(scene, scene.render.filepath)
    t = time.time()
    bpy.ops.render.render(write_still=True)
    print(f"[render] {cam} {mode} frame {scene.frame_current} -> {scene.render.filepath} ({time.time() - t:.1f}s)",
          flush=True)


def render_pano(scene, spot, mode, args):
    """Equirectangular 360-degree image from an eye-level spot; the image centre faces +x (three.js convention)."""
    ob = bpy.data.objects[spot]
    cam_data = bpy.data.cameras.get("PanoCam") or bpy.data.cameras.new("PanoCam")
    cam_data.type, cam_data.panorama_type = "PANO", "EQUIRECTANGULAR"
    cam_data.clip_start, cam_data.clip_end = 0.05, 6000.0
    cam = bpy.data.objects.get("PanoCam") or bpy.data.objects.new("PanoCam", cam_data)
    if cam.name not in scene.collection.objects:
        scene.collection.objects.link(cam)
    cam.location = ob.matrix_world.translation
    cam.rotation_euler = (math.pi / 2, 0.0, -math.pi / 2)
    scene.camera = cam
    scene.frame_set(args.frame or int(ob.get("still_frame", 1)))
    scene.render.use_motion_blur = False
    scene.render.filepath = output_path(args.out, spot, mode)
    set_image_format(scene, scene.render.filepath)
    t = time.time()
    bpy.ops.render.render(write_still=True)
    print(f"[pano] {spot} {mode} -> {scene.render.filepath} ({time.time() - t:.1f}s)", flush=True)


def render_anim(scene, cam, mode, args):
    """Render PNG frames (resumable), then encode MP4 with ffmpeg (Blender's own encoder as fallback)."""
    scene.camera = bpy.data.objects[cam]
    start, end = args.frames or (scene.frame_start, scene.frame_end)
    mp4 = output_path(args.out, cam, mode, ".mp4")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        scene.frame_start, scene.frame_end = start, end
        scene.render.image_settings.media_type = "VIDEO"
        scene.render.image_settings.file_format = "FFMPEG"
        scene.render.ffmpeg.format, scene.render.ffmpeg.codec = "MPEG4", "H264"
        scene.render.ffmpeg.constant_rate_factor = "HIGH"
        scene.render.filepath = mp4
        bpy.ops.render.render(animation=True)
        return
    frames_dir = os.path.splitext(mp4)[0] + "_frames"
    os.makedirs(frames_dir, exist_ok=True)
    set_image_format(scene, ".png")
    scene.render.use_persistent_data = True     # keep the synced scene between frames
    for f in range(start, end + 1):
        path = os.path.join(frames_dir, f"{f:04d}.png")
        if os.path.exists(path):
            continue
        scene.frame_set(f)
        scene.render.filepath = path
        t = time.time()
        bpy.ops.render.render(write_still=True)
        print(f"[anim] {cam} {mode} frame {f}/{end} ({time.time() - t:.1f}s)", flush=True)
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", str(scene.render.fps), "-start_number", str(start),
                    "-i", os.path.join(frames_dir, "%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                    "-movflags", "+faststart", mp4], check=True)
    print(f"[anim] wrote {mp4}", flush=True)


def main():
    args = parse_args()
    if not bpy.data.filepath:
        bpy.ops.wm.open_mainfile(filepath=args.blend)
    scene = bpy.context.scene
    setup_engine(scene, args)
    if args.pano:
        spots = sorted(o.name for o in bpy.data.objects if o.name.startswith("PANO_"))
        cams = spots if args.cam in ("all", "CAM_Aerial") else args.cam.split(",")
    else:
        cams = CAMERAS if args.cam == "all" else args.cam.split(",")
    modes = ["day", "dusk"] if args.mode == "both" else [args.mode]
    for mode in modes:
        station_modes.apply_mode(scene, mode, args.sun_az, args.sun_el)
        for cam in cams:
            if args.pano:
                render_pano(scene, cam, mode, args)
            elif args.anim:
                render_anim(scene, cam, mode, args)
            else:
                render_still(scene, cam, mode, args)


if __name__ == "__main__":
    main()
