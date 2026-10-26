#!/usr/bin/env python3
"""Render the far-distance tree impostors for the web viewer: each library plant is rendered by Cycles from the side
(orthographic, transparent film, soft sky light) into one cell of web/trees.png, an atlas of 4 x 2 cells of 256 x 512.

    blender52/bin/python render_sprites.py      # then run export_web.py, which builds the crossed-quad impostors
"""
import os
import sys

import bpy  # noqa: I001  (must precede numpy-heavy modules with the PyPI bpy module)
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SPRITES = ["TreeA", "TreeB", "TreeC", "TreeD", "PineA", "PineB", "ShrubA", "ShrubB"]
CELL_W, CELL_H, COLS = 256, 512, 4


def frame(ob):
    """Orthographic frame of a plant standing at the origin: (scale, bottom margin) so the cell holds it base-down."""
    w = max(ob.dimensions.x, ob.dimensions.y)
    S = max(ob.dimensions.z, 2 * w) * 1.04
    return S, 0.01 * S


def main():
    bpy.ops.wm.open_mainfile(filepath=os.path.join(HERE, "station.blend"))
    sc = bpy.data.scenes.new("Sprites")
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 48
    sc.cycles.use_denoising = True
    sc.render.film_transparent = True
    sc.render.resolution_x, sc.render.resolution_y = CELL_W, CELL_H
    sc.view_settings.view_transform = "Standard"         # plain colours; the viewer applies its own tone mapping
    world = bpy.data.worlds.new("SpriteSky")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.62, 0.68, 0.74, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.1
    sc.world = world
    sun = bpy.data.objects.new("SpriteSun", bpy.data.lights.new("SpriteSun", "SUN"))
    sun.data.energy, sun.data.angle = 1.6, 0.4
    sun.rotation_euler = (0.35, 0.0, 0.0)                  # mostly from above: crowns lit on top, darker underneath
    cam = bpy.data.objects.new("SpriteCam", bpy.data.cameras.new("SpriteCam"))
    cam.data.type = "ORTHO"
    cam.rotation_euler = (np.pi / 2, 0, 0)                 # looking along +y
    for o in (sun, cam):
        sc.collection.objects.link(o)
    sc.camera = cam
    atlas = np.zeros((2 * CELL_H, COLS * CELL_W, 4), np.float32)
    tmp = os.path.join(HERE, "renders", "_sprite.png")
    for k, name in enumerate(SPRITES):
        src = bpy.data.objects[name]
        ob = bpy.data.objects.new(f"Sprite_{name}", src.data)
        sc.collection.objects.link(ob)
        S, m = frame(src)
        cam.data.ortho_scale = S
        cam.location = (0, -60, S / 2 - m)
        sc.render.filepath = tmp
        bpy.ops.render.render(write_still=True, scene=sc.name)
        img = bpy.data.images.load(tmp, check_existing=False)
        px = np.array(img.pixels[:], np.float32).reshape(CELL_H, CELL_W, 4)[::-1]     # top row first
        bpy.data.images.remove(img)
        a = px[..., 3:4]
        if (a > 0.5).any():                             # bleed colour into transparent pixels: no dark mip fringes
            px[..., :3] = np.where(a > 0.02, px[..., :3], px[..., :3][a[..., 0] > 0.5].mean(0))
        r, c = divmod(k, COLS)
        atlas[r * CELL_H:(r + 1) * CELL_H, c * CELL_W:(c + 1) * CELL_W] = px
        sc.collection.objects.unlink(ob)
        print(f"[sprite] {name}: {src.dimensions.z:.1f} m tall, frame {S:.1f} m", flush=True)
    os.remove(tmp)
    out = bpy.data.images.new("trees", COLS * CELL_W, 2 * CELL_H, alpha=True)
    out.pixels[:] = atlas[::-1].ravel()
    out.filepath_raw = os.path.join(HERE, "web", "trees.png")
    out.file_format = "PNG"
    out.save()
    print("[sprite] wrote web/trees.png", flush=True)


if __name__ == "__main__":
    sys.exit(main())
