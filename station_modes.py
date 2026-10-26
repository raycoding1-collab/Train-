"""Lighting modes for Emerald Valley Station, shared by build_station.py and render.py.

A mode sets the sky + sun, the artificial lights (every material node named
"LIGHTS*" and every light object carrying a "dusk_energy" property), exposure,
and the compositor's haze / bloom.
"""
import math

import bpy
from mathutils import Vector

MODES = {
    "day": dict(
        sun_az=305.0, sun_el=36.0, sun_strength=4.2, sun_color=(1.0, 0.97, 0.92),
        sky_strength=0.32, air=1.0, aerosol=1.2, ozone=1.0,
        lights=0.0, exposure=-0.35,
        haze=(0.70, 0.78, 0.88), haze_amount=0.16, bloom=0.0,
        cloud_color=(1.0, 1.0, 1.0), cloud_cover=0.56,
    ),
    "dusk": dict(
        sun_az=287.0, sun_el=1.5, sun_strength=2.0, sun_color=(1.0, 0.48, 0.22),
        sky_strength=0.22, air=1.0, aerosol=3.0, ozone=1.2,
        lights=1.0, exposure=0.15,
        haze=(0.45, 0.33, 0.32), haze_amount=0.14, bloom=0.55,
        cloud_color=(1.5, 0.75, 0.6), cloud_cover=0.52,
    ),
}


def sun_vector(az_deg, el_deg):
    """Unit vector pointing *towards* the sun. Azimuth is a compass bearing: 0 = +Y, 90 = +X."""
    az, el = math.radians(az_deg), math.radians(el_deg)
    return Vector((math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)))


def apply_mode(scene, mode, sun_az=None, sun_el=None):
    p = dict(MODES[mode])
    if sun_az is not None:
        p["sun_az"] = sun_az
    if sun_el is not None:
        p["sun_el"] = sun_el

    nodes = scene.world.node_tree.nodes
    sky = nodes["Sky"]
    sky.sun_elevation = math.radians(p["sun_el"])
    sky.sun_rotation = math.radians(p["sun_az"])
    sky.air_density = p["air"]
    sky.aerosol_density = p["aerosol"]
    sky.ozone_density = p["ozone"]
    nodes["Background"].inputs["Strength"].default_value = p["sky_strength"]
    if "CloudColor" in nodes:
        nodes["CloudColor"].outputs[0].default_value = (*p["cloud_color"], 1.0)
        nodes["CloudCover"].outputs[0].default_value = p["cloud_cover"]

    sun = bpy.data.objects["Sun"]
    sun.rotation_euler = sun_vector(p["sun_az"], p["sun_el"]).to_track_quat("Z", "Y").to_euler()
    sun.data.energy = p["sun_strength"]
    sun.data.color = p["sun_color"]

    for mat in bpy.data.materials:
        if mat.node_tree:
            for n in mat.node_tree.nodes:
                if n.name.startswith("LIGHTS"):
                    n.outputs[0].default_value = p["lights"]
    for ob in scene.objects:
        if ob.type == "LIGHT" and "dusk_energy" in ob:
            ob.data.energy = ob["dusk_energy"] * p["lights"]
            ob.hide_render = p["lights"] <= 0.0

    scene.view_settings.exposure = p["exposure"]

    comp = scene.compositing_node_group
    if comp:
        haze = comp.nodes.get("Haze")
        if haze:
            color_b = next(s for s in haze.inputs if s.identifier == "B_Color")
            color_b.default_value = (*p["haze"], 1.0)
        amount = comp.nodes.get("HazeAmount")
        if amount:
            amount.inputs[1].default_value = p["haze_amount"]
        bloom = comp.nodes.get("Bloom")
        if bloom:
            bloom.mute = p["bloom"] <= 0.0
            bloom.inputs["Strength"].default_value = p["bloom"]
    scene["mode"] = mode
