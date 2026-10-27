"""Lighting modes for Emerald Valley Station, shared by build_station.py and render.py.

A mode sets the sky + sun, the artificial lights (every material node named
"LIGHTS*" and every light object carrying a "dusk_energy" property), exposure,
the compositor's haze / bloom, and a scattering atmosphere that turns sunlight through the
roof lattice and the oculi into visible beams.
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
        atmo=(0.0, 0.55, (1.0, 1.0, 1.0)), inner=(0.0035, 0.6),
        cloud_color=(1.0, 1.0, 1.0), cloud_cover=0.56,
    ),
    "dusk": dict(
        sun_az=287.0, sun_el=1.5, sun_strength=2.0, sun_color=(1.0, 0.48, 0.22),
        sky_strength=0.22, air=1.0, aerosol=3.0, ozone=1.2,
        lights=1.0, exposure=0.15,
        haze=(0.45, 0.33, 0.32), haze_amount=0.14, bloom=0.55,
        atmo=(0.0, 0.72, (1.0, 0.92, 0.84)), inner=(0.007, 0.75),
        cloud_color=(1.5, 0.75, 0.6), cloud_cover=0.52,
    ),
}


def sun_vector(az_deg, el_deg):
    """Unit vector pointing *towards* the sun. Azimuth is a compass bearing: 0 = +Y, 90 = +X."""
    az, el = math.radians(az_deg), math.radians(el_deg)
    return Vector((math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)))


def _volume_box(scene, name, size, center, mat_name):
    import bmesh
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=size, verts=bm.verts)
    bmesh.ops.translate(bm, vec=center, verts=bm.verts)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    scene.collection.objects.link(ob)
    mat = bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    mat.node_tree.nodes.clear()
    me.materials.append(mat)
    return ob, mat.node_tree


def set_atmosphere(scene, density, anisotropy, color, inner=(0.0, 0.5)):
    """Scattering air, single scattering: a thin layer over the whole site for aerial depth, and denser air under the
    roof (fading out towards the open sides, kept below the lowest roof edge) so sunlight through the lattice and
    the oculi shows as beams."""
    ob = bpy.data.objects.get("Atmosphere")
    if ob is None:
        ob, nt = _volume_box(scene, "Atmosphere", (3200, 2400, 152), (0, 0, 64), "AtmosphereVolume")
        vol = nt.nodes.new("ShaderNodeVolumePrincipled")
        vol.name = "Vol"
        nt.links.new(vol.outputs[0], nt.nodes.new("ShaderNodeOutputMaterial").inputs["Volume"])
    vol = ob.active_material.node_tree.nodes["Vol"]
    vol.inputs["Density"].default_value = density
    vol.inputs["Anisotropy"].default_value = anisotropy
    vol.inputs["Color"].default_value = (*color, 1.0)
    ob.hide_render = density <= 0.0

    inside = bpy.data.objects.get("AtmosphereInner")
    if inside is None:
        size = (224.0, 116.0, 30.0)                  # roof footprint; z from -8 m to 22 m, under the lowest roof edge
        inside, nt = _volume_box(scene, "AtmosphereInner", size, (0, 0, 7.0), "AtmosphereInnerVolume")
        gen = nt.nodes.new("ShaderNodeTexCoord")
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        nt.links.new(gen.outputs["Generated"], sep.inputs[0])

        def edge(axis, length, fade):               # 0 at the box side, 1 once `fade` metres inside
            m = nt.nodes.new("ShaderNodeMath")
            m.operation = "PINGPONG"
            nt.links.new(sep.outputs[axis], m.inputs[0])
            m.inputs[1].default_value = 0.5            # min(g, 1 - g)
            r = nt.nodes.new("ShaderNodeMapRange")
            r.interpolation_type = "SMOOTHERSTEP"
            nt.links.new(m.outputs[0], r.inputs["Value"])
            r.inputs["From Max"].default_value = fade / length
            return r.outputs["Result"]

        mx = nt.nodes.new("ShaderNodeMath")
        mx.operation = "MULTIPLY"
        nt.links.new(edge("X", size[0], 22.0), mx.inputs[0])
        nt.links.new(edge("Y", size[1], 20.0), mx.inputs[1])
        top = edge("Z", size[2], 3.0)
        mz = nt.nodes.new("ShaderNodeMath")
        mz.operation = "MULTIPLY"
        nt.links.new(mx.outputs[0], mz.inputs[0])
        nt.links.new(top, mz.inputs[1])
        dens = nt.nodes.new("ShaderNodeMath")
        dens.operation = "MULTIPLY"
        dens.name = "Density"
        nt.links.new(mz.outputs[0], dens.inputs[0])
        vol = nt.nodes.new("ShaderNodeVolumePrincipled")
        vol.name = "Vol"
        nt.links.new(dens.outputs[0], vol.inputs["Density"])
        nt.links.new(vol.outputs[0], nt.nodes.new("ShaderNodeOutputMaterial").inputs["Volume"])
        inside.active_material.cycles.volume_step_rate = 4.0      # the field is smooth: long steps are enough
    nt = inside.active_material.node_tree
    nt.nodes["Density"].inputs[1].default_value = inner[0]
    nt.nodes["Vol"].inputs["Anisotropy"].default_value = inner[1]
    nt.nodes["Vol"].inputs["Color"].default_value = (*color, 1.0)
    inside.hide_render = inner[0] <= 0.0


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
    set_atmosphere(scene, *p.get("atmo", (0.0, 0.0, (1.0, 1.0, 1.0))), inner=p.get("inner", (0.0, 0.5)))
    scene["mode"] = mode
