#!/usr/bin/env python3
"""Export Emerald Valley Station for the first-person web viewer in web/.

    blender -b station.blend --python export_web.py
    python export_web.py            # with the `bpy` module; opens station.blend next to this file

Writes web/station.json (a manifest) and web/*.bin (gzip-compressed geometry) for web/index.html.
Architecture is exported as built; the terrain is resampled with baked colours; vegetation becomes
GPU instances of low-poly stand-ins derived from each plant's own leaf clusters (two levels of detail).
Coordinates are converted to three.js conventions (y up): (x, y, z) -> (x, z, -y).
"""
import gzip
import json
import math
import os
import sys

import bpy  # noqa: I001  (must precede build_station / bmesh with the PyPI bpy module)
import time
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_station as bs  # noqa: E402
import render_sprites  # noqa: E402

OUT = os.path.join(HERE, "web")
BLEND = os.path.join(HERE, "station.blend")

# objects exported as static geometry: name -> (collides with the walker, casts shadows)
STATIC = {
    "RoofFrame": (True, True), "RoofGlazing": (True, False), "RoofNodeLights": (False, False),
    "RoofGardenSoil": (True, True), "DendriformColumns": (True, True), "Decks": (True, True),
    "BridgeAndWalks": (True, True), "Balustrades": (True, False), "Lawns": (True, False), "Plaza": (True, True),
    "LightStrips": (False, False), "People": (False, True), "VineCurtains": (False, True), "Stream": (False, False),
    "Trail": (False, False), "SignText": (True, False), "SignText.001": (False, False),
}
COLOR_MATS = {"Leaf", "VineLeaf", "Needle", "People", "Flower"}
LEAF_RAMPS = {
    "Leaf": [(0.0, (0.035, 0.1, 0.018)), (0.55, (0.08, 0.2, 0.035)), (1.0, (0.2, 0.3, 0.055))],
    "VineLeaf": [(0.0, (0.05, 0.14, 0.025)), (0.7, (0.11, 0.26, 0.04)), (1.0, (0.3, 0.36, 0.07))],
    "Needle": [(0.0, (0.02, 0.06, 0.03)), (1.0, (0.06, 0.13, 0.05))],
}


# Baked shading: ambient occlusion by ray casting against the static model, plus canopy shade on the ground,
# so the real-time view gets the contact shadows and forest-floor darkness of the path-traced renders.
NO_AO = {"Glass", "SolarGlass", "SteelMesh", "StreamWater", "PoolWater", "LED", "NodeLight", "LampHead", "SignGlow",
         "Display", "People", "VineLeaf"}
TREES = []      # (x, y, crown radius, density) of every plant instance in Blender space, filled by export_vegetation


def hemisphere(k):
    i = np.arange(k) + 0.5
    phi = i * np.pi * (3 - np.sqrt(5))
    r = np.sqrt(i / k)                          # cosine-weighted
    return np.c_[r * np.cos(phi), r * np.sin(phi), np.sqrt(1 - r * r)]


def bake_ao(bvh, P, N, k=12, dist=7.0):
    """Fraction of the hemisphere (cosine weighted, distance faded) that sees open sky within `dist`."""
    H = hemisphere(k)
    up = np.where(np.abs(N[:, 1:2]) < 0.9, [[0.0, 1.0, 0.0]], [[1.0, 0.0, 0.0]])
    T = np.cross(up, N)
    T /= np.linalg.norm(T, axis=1, keepdims=True).clip(1e-9)
    B = np.cross(N, T)
    rot = np.random.default_rng(5).uniform(0, 2 * np.pi, len(P))
    c, s = np.cos(rot), np.sin(rot)
    O = (P + N * 0.05).tolist()
    occ = np.zeros(len(P))
    for j in range(k):
        hx, hy = H[j, 0] * c - H[j, 1] * s, H[j, 0] * s + H[j, 1] * c
        D = (T * hx[:, None] + B * hy[:, None] + N * H[j, 2]).tolist()
        cast = bvh.ray_cast
        for i in range(len(O)):
            hit = cast(Vector(O[i]), Vector(D[i]), dist)
            if hit[0] is not None:
                occ[i] += 1.0 - 0.5 * hit[3] / dist
    return np.clip(1 - occ / k, 0, 1)


def canopy_shade(x, y, cell=2.0):
    """Ground darkening under tree crowns: summed crown discs, box blurred, sampled at (x, y) in Blender space."""
    x0, y0 = -1500.0, -1100.0
    D = np.zeros((int(3000 / cell) + 2, int(2200 / cell) + 2), np.float32)
    for tx, ty, R, k in TREES:
        i0, i1 = max(int((tx - R - x0) / cell), 0), min(int((tx + R - x0) / cell) + 1, D.shape[0])
        j0, j1 = max(int((ty - R - y0) / cell), 0), min(int((ty + R - y0) / cell) + 1, D.shape[1])
        if i1 <= i0 or j1 <= j0:
            continue
        xs = x0 + (np.arange(i0, i1) + 0.5) * cell
        ys = y0 + (np.arange(j0, j1) + 0.5) * cell
        D[i0:i1, j0:j1] += k * np.clip(1 - ((xs[:, None] - tx) ** 2 + (ys[None, :] - ty) ** 2) / (R * R), 0, 1)
    for ax in (0, 1):                             # 5-cell box blur
        c = np.cumsum(np.pad(D, [(3, 2) if a == ax else (0, 0) for a in (0, 1)], mode="edge"), axis=ax)
        D = (np.take(c, range(5, c.shape[ax]), axis=ax) - np.take(c, range(0, c.shape[ax] - 5), axis=ax)) / 5
    i = np.clip(((x - x0) / cell).astype(int), 0, D.shape[0] - 1)
    j = np.clip(((y - y0) / cell).astype(int), 0, D.shape[1] - 1)
    return np.clip(1 / (1 + 1.4 * D[i, j]), 0.38, 1)


def to_three(p):
    p = np.asarray(p, np.float64)
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], -1)


def srgb8_to_lin(c):
    c = np.asarray(c, np.float64) / 255
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin_to_srgb8(c):
    c = np.clip(np.asarray(c, np.float64), 0, 1)
    s = np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)
    return np.round(s * 255).astype(np.uint8)


def ramp(values, stops):
    pos = np.array([p for p, _ in stops])
    cols = np.array([c for _, c in stops])
    return np.stack([np.interp(values, pos, cols[:, k]) for k in range(3)], -1)


class Pack:
    """One gzip-compressed binary file of 4-byte aligned typed arrays."""

    def __init__(self, name):
        self.name, self.parts, self.size = name, [], 0

    def add(self, arr):
        b = np.ascontiguousarray(arr).tobytes()
        pad = (-self.size) % 4
        self.parts.append(b"\0" * pad + b)
        off = self.size + pad
        self.size = off + len(b)
        return off

    def write(self):
        data = gzip.compress(b"".join(self.parts), 9, mtime=0)   # reproducible bytes
        with open(os.path.join(OUT, self.name), "wb") as f:
            f.write(data)
        return len(data)


def mesh_record(pack, name, material, P, N, C=None, UV=None, idx=None, **extra):
    """Quantise one triangle mesh (three.js coordinates) into `pack`."""
    lo, hi = P.min(0), P.max(0)
    scale = np.maximum(hi - lo, 1e-6)
    q = np.round((P - lo) / scale * 65535).astype(np.uint16)
    n = np.round(np.clip(N, -1, 1) * 127).astype(np.int8)
    rec = dict(name=name, material=material, file=pack.name, count=len(P),
               pos=pack.add(q), posMin=lo.round(5).tolist(), posScale=scale.round(5).tolist(), nrm=pack.add(n))
    if C is not None:
        rec["col"] = pack.add(np.asarray(C, np.uint8))
    if UV is not None:
        rec["uv"] = pack.add(np.asarray(UV, np.float32))
    idx = np.asarray(idx).ravel()
    big = len(P) > 65535
    rec["idx"] = pack.add(idx.astype(np.uint32 if big else np.uint16))
    rec["idxCount"], rec["idxType"] = len(idx), "u32" if big else "u16"
    rec.update(extra)
    return rec


def extract(ob, dg, local=False, keep_faces=None):
    """Evaluated triangles of an object grouped by material: {material: (P, N, C or None, I)} in three.js space."""
    try:
        ev = ob.evaluated_get(dg)
    except Exception:            # library objects outside the view layer: use the mesh as stored
        ev = ob
    me = ev.to_mesh()
    me.calc_loop_triangles()
    nt = len(me.loop_triangles)
    tl = np.empty(nt * 3, np.int32)
    me.loop_triangles.foreach_get("loops", tl)
    tp = np.empty(nt, np.int32)
    me.loop_triangles.foreach_get("polygon_index", tp)
    tm = np.empty(nt, np.int32)
    me.loop_triangles.foreach_get("material_index", tm)
    lv = np.empty(len(me.loops), np.int32)
    me.loops.foreach_get("vertex_index", lv)
    co = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3).astype(np.float64)
    cn = np.empty(len(me.loops) * 3, np.float32)
    me.corner_normals.foreach_get("vector", cn)
    cn = cn.reshape(-1, 3).astype(np.float64)
    fattr = {}
    for a in me.attributes:
        if a.domain == "FACE" and a.name in ("lv", "pcol", "fc"):
            if a.data_type == "FLOAT":
                v = np.empty(len(me.polygons), np.float32)
                a.data.foreach_get("value", v)
            else:
                v = np.empty(len(me.polygons) * 4, np.float32)
                a.data.foreach_get("color", v)
                v = v.reshape(-1, 4)[:, :3]
            fattr[a.name] = v
    mats = [m.name if m else "" for m in me.materials]
    ev.to_mesh_clear()
    if not local:
        M = np.array(ob.matrix_world)
        co = co @ M[:3, :3].T + M[:3, 3]
        Nm = np.linalg.inv(M[:3, :3]).T
        cn = cn @ Nm.T
    cn /= np.linalg.norm(cn, axis=1, keepdims=True).clip(1e-9)
    if keep_faces is not None:
        keep = keep_faces(tp)
        tl, tp, tm = tl.reshape(-1, 3)[keep].ravel(), tp[keep], tm[keep]
    out = {}
    for k, mname in enumerate(mats):
        sel = tm == k
        if not sel.any():
            continue
        loops = tl.reshape(-1, 3)[sel].ravel()
        polys = np.repeat(tp[sel], 3)
        vid = lv[loops]
        nq = np.round(cn[loops] * 127).astype(np.int64)
        C = None
        if mname in COLOR_MATS:
            if "pcol" in fattr and mname == "People":
                C = lin_to_srgb8(fattr["pcol"][polys])
            elif "fc" in fattr and mname == "Flower":
                C = lin_to_srgb8(fattr["fc"][polys])
            elif "lv" in fattr and mname in LEAF_RAMPS:
                C = lin_to_srgb8(ramp(fattr["lv"][polys], LEAF_RAMPS[mname]))
        key = np.c_[vid, nq] if C is None else np.c_[vid, nq, C.astype(np.int64)]
        uniq, inv = np.unique(key, axis=0, return_inverse=True)
        P = to_three(co[uniq[:, 0]])
        N = to_three(uniq[:, 1:4] / 127.0)
        out[mname] = (P, N, None if C is None else uniq[:, 4:7].astype(np.uint8), inv.reshape(-1, 3))
    return out


# ------------------------------------------------------------------ terrain
def export_terrain():
    X = 1500 * np.sinh(3.6 * np.linspace(-1, 1, 401)) / np.sinh(3.6)
    Y = 1100 * np.sinh(3.6 * np.linspace(-1, 1, 281)) / np.sinh(3.6)
    XX, YY = np.meshgrid(X, Y, indexing="ij")
    H = bs.terrain_h(XX, YY)
    gx, gy = np.gradient(H, X, Y)
    nrm = bs.norm(np.stack([-gx, -gy, np.ones_like(H)], -1))
    d = np.abs(YY - bs.stream_y(XX))
    n1 = bs.NZ_FINE(XX * 0.6 + 40, YY * 0.6)
    grass = ramp(np.clip(0.5 + 0.45 * n1, 0, 1),
                 [(0.0, (0.07, 0.15, 0.03)), (0.5, (0.13, 0.22, 0.05)), (1.0, (0.22, 0.26, 0.08))])
    wet = (1 - bs.smoothstep(3.0, 10.0, d))[..., None]
    col = grass * (1 - wet) + np.array([0.05, 0.1, 0.03]) * wet
    bed = (1 - bs.smoothstep(2.8, 4.2, d))[..., None]
    col = col * (1 - bed) + np.array([0.3, 0.26, 0.2]) * bed
    slope = (1 - nrm[..., 2])[..., None]
    rock = bs.smoothstep(0.32, 0.5, slope)
    col = col * (1 - rock) + np.array([0.33, 0.32, 0.3]) * rock
    P = to_three(np.stack([XX, YY, H], -1).reshape(-1, 3))
    N = to_three(nrm.reshape(-1, 3))
    UV = np.stack([XX, YY], -1).reshape(-1, 2) / 6.0
    F = bs.grid_faces(len(X), len(Y))
    I = np.concatenate([F[:, [0, 1, 2]], F[:, [0, 2, 3]]])
    col = col * canopy_shade(XX, YY)[..., None]
    return P, N, col.reshape(-1, 3), UV, I


# ------------------------------------------------------------------ vegetation stand-ins
def ico_mesh(sub):
    V, F = bs.ico(sub)
    return V, F


def blob_standin(proto, detail, rng):
    """Low-poly tree/shrub: its LeafCore blobs (ico(2) chunks of 162 verts) re-made as coarse icospheres + trunk."""
    me = proto.data
    mats = [m.name for m in me.materials]
    nv = len(me.vertices)
    co = np.empty(nv * 3, np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    pm = np.empty(len(me.polygons), np.int32)
    me.polygons.foreach_get("material_index", pm)
    lt = np.empty(len(me.polygons), np.int32)
    me.polygons.foreach_get("loop_total", lt)
    lv = np.empty(len(me.loops), np.int32)
    me.loops.foreach_get("vertex_index", lv)
    loop_mat = np.repeat(pm, lt)

    def verts_of(mat):
        return np.unique(lv[loop_mat == mats.index(mat)]) if mat in mats else np.array([], int)

    core_v = verts_of("LeafCore")     # blob cores are ico(2) chunks of consecutive vertices
    chunk = len(bs.ico(2)[0])
    blobs = []
    for start in range(0, len(core_v), chunk):
        pts = co[core_v[start:start + chunk]]
        if len(pts) < chunk // 2:
            continue
        c = pts.mean(0)
        r = (pts.max(0) - pts.min(0)) / 2 * 1.22
        blobs.append((c, r))
    blobs.sort(key=lambda b: -np.prod(b[1]))
    keep = {"near": 18, "far": 8}[detail]
    blobs = blobs[:keep]
    Vs, Fs, Cs = [], [], []
    base = 0
    Vi, Fi = ico_mesh(2 if detail == "near" else 1)
    for c, r in blobs:
        jitter = 1 + 0.08 * rng.standard_normal(len(Vi))
        V = c + Vi * r * jitter[:, None]
        shade = 0.55 + 0.45 * (Vi[:, 2] * 0.5 + 0.5)
        tint = ramp(np.full(len(Vi), rng.uniform(0.3, 0.8)), LEAF_RAMPS["Leaf"]) * shade[:, None]
        Vs.append(V)
        Fs.append(Fi + base)
        Cs.append(tint)
        base += len(V)
    leaf = (np.concatenate(Vs), np.concatenate(Fs), np.concatenate(Cs)) if Vs else None
    cards = None
    if detail == "near" and "Leaf" in mats and proto.name[:5] in ("TreeA", "TreeB", "TreeC", "TreeD", "Grove", "Emerg", "Shrub"):
        # a sparse set of the tree's own leaf cards, enlarged, so near crowns read as leaves rather than blobs
        landmark = proto.name.startswith(("Grove", "Emerg"))      # only a few of these: give them more leaves
        shrub = proto.name.startswith("Shrub")
        polys = np.where((pm == mats.index("Leaf")) & (lt == 4))[0][::3 if landmark else 6 if shrub else 9]
        ls = np.empty(len(me.polygons), np.int32)
        me.polygons.foreach_get("loop_start", ls)
        quad = co[lv[ls[polys][:, None] + np.arange(4)]]
        ctr = quad.mean(1, keepdims=True)
        quad = ctr + (quad - ctr) * (1.35 if landmark else 1.7 if shrub else 1.9)
        shade = np.empty(len(me.polygons), np.float32)
        me.attributes["lv"].data.foreach_get("value", shade)
        col = np.repeat(ramp(shade[polys], LEAF_RAMPS["Leaf"]), 4, 0)
        n = len(polys)
        cards = (quad.reshape(-1, 3), (4 * np.arange(n)[:, None] + np.arange(4)), col)
        Vs2, Cs2 = leaf[0], leaf[2] * 0.55       # darker cores under the cards
        leaf = (Vs2, leaf[1], Cs2)
    trunk = None
    if "Bark" in mats:
        bark = co[verts_of("Bark")]
        top = min((c[2] - r[2] * 0.4 for c, r in blobs), default=bark[:, 2].max())
        rad = max(0.08, np.percentile(np.hypot(bark[bark[:, 2] < 1.0, 0], bark[bark[:, 2] < 1.0, 1]), 60)) \
            if (bark[:, 2] < 1.0).any() else 0.2
        V, F = bs.tube([(0, 0, -0.3), (0, 0, top)], [rad, rad * 0.6], 6 if detail == "near" else 5, caps=False)
        trunk = (V, F[0])
    return leaf, trunk, cards


def conifer_standin(proto, detail):
    me = proto.data
    co = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    r = np.hypot(co[:, 0], co[:, 1])
    z = co[:, 2]
    zmin, zmax = np.percentile(z[r > 0.4], 1), z.max()
    tiers = 5 if detail == "near" else 3
    sides = 8 if detail == "near" else 5
    prof = []
    for k in range(tiers):
        z0 = zmin + (zmax - zmin) * k / tiers
        z1 = zmin + (zmax - zmin) * (k + 1) / tiers
        sel = (z >= z0) & (z < z1)
        rr = np.percentile(r[sel], 95) if sel.sum() > 5 else 0.3
        prof += [(rr * 1.05, z0), (rr * 0.45, z1 - (z1 - z0) * 0.25)]
    prof.append((0.0, zmax))
    V, F = bs.lathe(np.array(prof), sides, caps=False)
    shade = 0.6 + 0.4 * (V[:, 2] - zmin) / (zmax - zmin)
    C = ramp(np.full(len(V), 0.5), LEAF_RAMPS["Needle"]) * shade[:, None]
    tV, tF = bs.tube([(0, 0, -0.3), (0, 0, zmin + 0.5)], 0.25, 5, caps=False)
    return (V, F[0], C), (tV, tF[0])


def tuft_standin(proto, detail, rng):
    """Grass / flower clumps: a few blades plus small coloured flower heads."""
    name = proto.name
    nb = 7 if detail == "near" else 4
    Vs, Fs, Cs = [], [], []
    base = 0
    for k in range(nb):
        a = rng.uniform(0, 2 * np.pi)
        h = rng.uniform(0.3, 0.55) * (0.6 if name == "GrassLow" else 1.0)
        o = np.array([np.cos(a), np.sin(a), 0]) * rng.uniform(0, 0.1)
        s = np.array([-np.sin(a), np.cos(a), 0]) * 0.02
        tip = o + np.array([np.cos(a) * h * 0.35, np.sin(a) * h * 0.35, h])
        Vs.append(np.array([o - s, o + s, tip]))
        Fs.append(np.array([[0, 1, 2]]) + base)
        Cs.append(ramp(np.full(3, rng.uniform(0.3, 0.9)), LEAF_RAMPS["Leaf"]) * np.array([[0.6], [0.6], [1.0]]))
        base += 3
    heads = {"Daisies": [(0.95, 0.95, 0.92)], "Cosmos": [(0.85, 0.28, 0.55), (0.96, 0.62, 0.76)],
             "Lavender": [(0.42, 0.3, 0.68)]}.get(name)
    if heads:
        Vi, Fi = ico_mesh(0)
        for k in range(6 if detail == "near" else 3):
            a = rng.uniform(0, 2 * np.pi)
            c = np.array([np.cos(a) * 0.1, np.sin(a) * 0.1, rng.uniform(0.3, 0.5)])
            rad = (0.012, 0.012, 0.04) if name == "Lavender" else (0.03, 0.03, 0.012)
            Vs.append(c + Vi * rad)
            Fs.append(Fi + base)
            Cs.append(np.tile(heads[k % len(heads)], (len(Vi), 1)))
            base += len(Vi)
    return np.concatenate(Vs), np.concatenate(Fs), np.concatenate(Cs)


def fern_standin(detail, rng):
    n = 8 if detail == "near" else 5
    Vs, Fs = [], []
    for k in range(n):
        a = 2 * np.pi * k / n
        d = np.array([np.cos(a), np.sin(a), 0])
        s = np.array([-np.sin(a), np.cos(a), 0])
        L = rng.uniform(0.8, 1.2)
        Vs.append(np.array([s * 0.05, d * L * 0.5 + s * 0.12 + [0, 0, 0.35 * L], d * L + [0, 0, 0.05],
                            d * L * 0.5 - s * 0.12 + [0, 0, 0.35 * L], -s * 0.05]))
        Fs.append(np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4]]) + 5 * k)
    V = np.concatenate(Vs)
    return V, np.concatenate(Fs), ramp(np.full(len(V), 0.6), LEAF_RAMPS["Leaf"])


def rock_standin(proto, detail):
    me = proto.data
    co = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    Vi, Fi = ico_mesh(1 if detail == "near" else 0)
    ext = (co.max(0) - co.min(0)) / 2
    return co.mean(0) + Vi * ext * (1 + 0.12 * np.sin(Vi @ [3.1, 1.7, 2.3]))[:, None], Fi


def triangulate(F):
    F = np.asarray(F)
    return np.concatenate([F[:, [0, 1, 2]], F[:, [0, 2, 3]]]) if F.shape[1] == 4 else F


def flat_normals_record(pack, name, material, V, F, C=None):
    """Unindexed (flat shaded) record for low-poly stand-ins, in three.js space."""
    F = triangulate(F)
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True).clip(1e-9)
    P = to_three(tri.reshape(-1, 3))
    N = to_three(np.repeat(n, 3, 0))
    col = None if C is None else lin_to_srgb8(np.asarray(C)[F].reshape(-1, 3))
    return mesh_record(pack, name, material, P, N, col, None, np.arange(len(P)).reshape(-1, 3))


def smooth_record(pack, name, material, V, F, C=None):
    F = triangulate(F)
    tri_n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, F[:, k], tri_n)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True).clip(1e-9)
    col = None if C is None else lin_to_srgb8(C)
    return mesh_record(pack, name, material, to_three(V), to_three(vn), col, None, F)


def sprite_record(pack, name, ob):
    """Far impostor: three crossed quads carrying the plant's Cycles render from web/trees.png (render_sprites.py)."""
    S, m = render_sprites.frame(ob)
    r, c = divmod(render_sprites.SPRITES.index(name), render_sprites.COLS)
    u0, u1 = c / render_sprites.COLS, (c + 1) / render_sprites.COLS
    vt, vb = 1 - r / 2, 1 - (r + 1) / 2                    # textures are flipped: v = 1 is the top row
    P, UV, I = [], [], []
    for a in np.radians([0, 60, 120]):
        h = np.array([np.cos(a), np.sin(a), 0]) * S / 4
        P += [-h + [0, 0, -m], h + [0, 0, -m], h + [0, 0, S - m], -h + [0, 0, S - m]]
        UV += [[u0, vb], [u1, vb], [u1, vt], [u0, vt]]
        b = len(P) - 4
        I += [[b, b + 1, b + 2], [b, b + 2, b + 3]]
    P = to_three(np.array(P))
    N = np.tile([0.0, 1.0, 0.0], (len(P), 1))           # lit as from above, like a canopy
    return mesh_record(pack, f"{name}:sprite", "Sprite", P, N, np.full((len(P), 3), 255, np.uint8),
                       np.array(UV, np.float32), np.array(I))


def export_vegetation(pack, dg, scene):
    rng = np.random.default_rng(3)
    lib = {o.name: o for o in bpy.data.collections["LIB"].all_objects}
    protos = {}
    for name, ob in lib.items():
        for detail in ("near", "far"):
            recs = []
            if detail == "far" and name in render_sprites.SPRITES and os.path.exists(os.path.join(OUT, "trees.png")):
                protos.setdefault(name, {})[detail] = [sprite_record(pack, name, ob)]
                continue
            if name.startswith("Pine"):
                (V, F, C), (tV, tF) = conifer_standin(ob, detail)
                recs += [smooth_record(pack, f"{name}:{detail}:crown", "StandinLeaf", V, F, C),
                         flat_normals_record(pack, f"{name}:{detail}:trunk", "Bark", tV, tF)]
            elif name.startswith("Rock"):
                V, F = rock_standin(ob, detail)
                recs.append(flat_normals_record(pack, f"{name}:{detail}", "Rock", V, F))
            elif name.startswith("Fern"):
                V, F, C = fern_standin(detail, rng)
                recs.append(flat_normals_record(pack, f"{name}:{detail}", "StandinLeaf", V, F, C))
            elif name.startswith("Grass") or name in ("Daisies", "Cosmos", "Lavender"):
                V, F, C = tuft_standin(ob, detail, rng)
                recs.append(flat_normals_record(pack, f"{name}:{detail}", "StandinLeaf", V, F, C))
            else:
                leaf, trunk, cards = blob_standin(ob, detail, rng)
                if leaf:
                    recs.append(smooth_record(pack, f"{name}:{detail}:crown", "StandinLeaf", *leaf))
                if cards:
                    recs.append(flat_normals_record(pack, f"{name}:{detail}:leaves", "Leaf", *cards))
                if trunk:
                    recs.append(flat_normals_record(pack, f"{name}:{detail}:trunk", "Bark", *trunk))
            protos.setdefault(name, {})[detail] = recs
        # hero detail: the very mesh the Cycles renders use, for plants close to the viewer
        thin = (lambda tp: tp % 3 == 0) if name == "Lavender" else None     # 6,480 florets: a third is plenty
        protos[name]["hero"] = [mesh_record(pack, f"{name}:hero:{mat}", mat, P, N, C, None, I)
                                for mat, (P, N, C, I) in extract(ob, dg, local=True, keep_faces=thin).items()
                                if mat != "LeafCore"]       # the leaves are dense enough; the solid core shows its facets from below

    crown = {}
    for name, ob in lib.items():
        if name.startswith(("Tree", "Pine", "Grove", "Emerg", "Shrub")):
            co = np.empty(len(ob.data.vertices) * 3, np.float32)
            ob.data.vertices.foreach_get("co", co)
            crown[name] = float(np.percentile(np.hypot(co[0::3], co[1::3]), 90))
    groups = {}
    for inst in dg.object_instances:
        if not inst.is_instance or not inst.parent or inst.instance_object.name not in lib:
            continue
        M = np.array(inst.matrix_world)
        s = float(np.linalg.norm(M[:3, 0]))
        rot = math.atan2(M[1, 0], M[0, 0])
        groups.setdefault((inst.parent.name, inst.instance_object.name), []).append(
            (*M[:3, 3], rot, s, (inst.random_id % 10007) / 10007.0))
    # plaza and emergent trees are plain objects sharing a library mesh
    for ob in scene.objects:
        if ob.type == "MESH" and ob.data.name in lib and ob.name not in lib:
            M = np.array(ob.matrix_world)
            groups.setdefault(("Placed", ob.data.name), []).append(
                (*M[:3, 3], ob.rotation_euler.z, ob.scale.x, rng.random()))
    inst_recs = []
    for (parent, proto), rows in sorted(groups.items()):
        a = np.array(rows, np.float64)
        if proto in crown:
            k = 0.9 if proto.startswith("Pine") else 0.45 if proto.startswith("Shrub") else 0.8
            TREES.extend((x, y, max(0.6, s * crown[proto]), k) for x, y, s in a[:, [0, 1, 4]])
        pos = to_three(a[:, :3])
        data = np.c_[pos, a[:, 3:6]].astype(np.float32)
        inst_recs.append(dict(group=parent, proto=proto, file=pack.name, data=pack.add(data), count=len(a)))
    return protos, inst_recs


# ------------------------------------------------------------------ cameras and trains
def camera_views(scene):
    views = []
    for name in ("CAM_Plaza", "CAM_Platform", "CAM_Valley", "CAM_TrailUp", "CAM_Roof", "CAM_Aerial"):
        ob = bpy.data.objects[name]
        M = np.array(ob.matrix_world)
        fwd = -M[:3, 2]
        views.append(dict(name=name, pos=to_three(M[:3, 3]).round(3).tolist(), dir=to_three(fwd).round(4).tolist()))
    return views


PANO_LABELS = {"PANO_Plaza": "Plaza", "PANO_Platform": "Platform", "PANO_SkyBridge": "Sky bridge",
               "PANO_CanopyWalk": "Canopy walk", "PANO_Stream": "Stream trail", "PANO_StairTower": "Stair tower",
               "PANO_Lookout": "Lookout", "PANO_Roof": "Above the roof"}


def pano_spots():
    return [dict(name=n, label=lbl, pos=to_three(np.array(bpy.data.objects[n].matrix_world.translation)).round(3).tolist(),
                 file=f"pano/{n}_day.jpg", fileDusk=f"pano/{n}_dusk.jpg")
            for n, lbl in PANO_LABELS.items() if n in bpy.data.objects]


def map_data():
    """Plan outlines for the viewer's mini-map, as [x, z] pairs in three.js space."""
    def xz(pts):
        pts = np.asarray(pts, float)
        return np.c_[pts[:, 0], -pts[:, 1]].round(1).tolist()

    def rect(x0, x1, y0, y1):
        return xz([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])

    u = np.linspace(0, 1, 90)
    roof = np.concatenate([bs.roof_uv_xyz(u, np.zeros_like(u)), bs.roof_uv_xyz(u[::-1], np.ones_like(u))])[:, :2]
    x = np.linspace(-900, 900, 361)
    return dict(
        roof=xz(roof), stream=xz(np.c_[x, bs.stream_y(x)]), pool=xz(bs.stadium(*bs.POOL, n=48)),
        walks=[xz(p[::3, :2]) for p in bs.walk_paths()], plaza=rect(-104, 104, 38, 118), terrace=rect(-104, 104, -59, -38),
        lawns=[rect(*r) for r in bs.LAWNS],
        guideways=[xz([[-1000, s * bs.GUIDE_Y], [1000, s * bs.GUIDE_Y]]) for s in (1, -1)],
        oculi=[[x, -y, round(float(bs.roof_z_at(np.array(x), np.array(y))), 2), a, b] for x, y, a, b in bs.OCULI],
    )


def main():
    if not bpy.data.filepath:
        bpy.ops.wm.open_mainfile(filepath=BLEND)
    scene = bpy.context.scene
    scene.frame_set(1)
    dg = bpy.context.evaluated_depsgraph_get()
    os.makedirs(OUT, exist_ok=True)
    packs = {k: Pack(f"{k}.bin") for k in ("site", "station", "plants")}
    protos, instances = export_vegetation(packs["plants"], dg, scene)
    tP, tN, tC, tUV, tI = export_terrain()
    items = []
    for name, (collide, shadow) in STATIC.items():
        ob = bpy.data.objects.get(name)
        if ob is None:
            continue
        keep = None
        if name == "VineCurtains":      # a quarter of the leaves is plenty in real time
            keep = lambda tp: (tp % 4 == 0)  # noqa: E731
        pack = packs["site"] if name in ("Stream", "Trail", "People", "VineCurtains") else packs["station"]
        for mat, (P, N, C, I) in extract(ob, dg, keep_faces=keep).items():
            items.append([pack, name, mat, P, N, C, I, collide, shadow])
    # occluders: the opaque architecture plus the terrain around the station
    near = (np.abs(tP[:, 0]) < 400) & (np.abs(tP[:, 2]) < 300)
    tri = tI[near[tI].all(1)]
    remap = np.cumsum(near) - 1
    occ_P, occ_F, base = [tP[near]], [remap[tri]], int(near.sum())
    for it in items:
        if it[2] not in NO_AO:
            occ_P.append(it[3]); occ_F.append(it[6] + base); base += len(it[3])
    bvh = BVHTree.FromPolygons(np.concatenate(occ_P).tolist(), np.concatenate(occ_F).tolist(), all_triangles=True)
    t0 = time.time()
    box = (np.abs(tP[:, 0]) < 260) & (np.abs(tP[:, 2]) < 200)
    tC[box] *= (0.3 + 0.7 * bake_ao(bvh, tP[box], tN[box], k=12, dist=25.0))[:, None]
    meshes = [mesh_record(packs["site"], "Terrain", "Terrain", tP, tN, lin_to_srgb8(tC), tUV, tI, collide=True, receiveOnly=True)]
    for pack, name, mat, P, N, C, I, collide, shadow in items:
        UV = None
        if mat == "SolarGlass":
            UV = np.c_[P[:, 0], -P[:, 2]] / 1.19
        elif mat in ("Paving", "Terrazzo", "Timber", "Lawn", "Gravel"):
            UV = np.c_[P[:, 0], -P[:, 2]] / {"Paving": 1.6, "Terrazzo": 4.0, "Timber": 2.0, "Lawn": 6.0,
                                              "Gravel": 3.0}[mat]
        if mat not in NO_AO:
            ao = 0.25 + 0.75 * bake_ao(bvh, P, N)
            C = lin_to_srgb8(np.repeat(ao[:, None], 3, 1)) if C is None else \
                lin_to_srgb8(srgb8_to_lin(C) * ao[:, None])
        meshes.append(mesh_record(pack, f"{name}:{mat}", mat, P, N, C, UV, I, collide=collide, castShadow=shadow))
    print(f"baked ambient occlusion in {time.time() - t0:.0f}s")
    train = []
    for mat, (P, N, C, I) in extract(bpy.data.objects["TrainA"], dg, local=True).items():
        train.append(mesh_record(packs["station"], f"Train:{mat}", mat, P, N, C, None, I, castShadow=True))
    manifest = dict(
        meshes=meshes, train=train, prototypes=protos, instances=instances, views=camera_views(scene), panos=pano_spots(), map=map_data(),
        trains=dict(z=bs.TRAIN_Z, y=bs.GUIDE_Y, platformStop=-70.0, length=6 * 26.0 + 5 * 0.9),
        files={},
    )
    for k, p in packs.items():
        size = p.write()
        manifest["files"][p.name] = size
        print(f"{p.name}: {p.size / 1e6:.1f} MB raw -> {size / 1e6:.1f} MB gzip")
    with open(os.path.join(OUT, "station.json"), "w") as f:
        json.dump(manifest, f, separators=(",", ":"))
    tris = sum(m["idxCount"] for m in meshes) // 3
    print(f"static triangles: {tris:,}; instance groups: {len(instances)}; "
          f"instances: {sum(i['count'] for i in instances):,}")


if __name__ == "__main__":
    main()
