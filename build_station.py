#!/usr/bin/env python3
"""Emerald Valley Station (翠谷站) - procedural Blender 5.2 scene.

A futuristic green rail hub: an undulating timber gridshell roof on branching
"dendriform" columns, a forest valley with a stream between two elevated maglev
decks, a canopy walk, vine curtains, roof gardens, solar glazing and a
reflecting pool.

    blender -b --factory-startup --python build_station.py
    python build_station.py          # with the `bpy` module from PyPI

Writes station.blend next to this file.
"""
import math
import os
import sys
import time

import bpy  # noqa: I001  (must precede bmesh when using the PyPI bpy module)
import bmesh
import numpy as np
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import station_modes  # noqa: E402

OUT = os.path.join(HERE, "station.blend")
T0 = time.time()

GRADE = 13.9        # finished level of plaza, terraces and platforms
GUIDE_Y = 24.0      # maglev guideways run along x at y = +-GUIDE_Y
TRAIN_Z = 13.15     # train object origin (underside of skirt)
FRAMES = 240


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


# =============================================================== numpy helpers
def smoothstep(e0, e1, x):
    t = np.clip((np.asarray(x, float) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def norm(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True).clip(1e-12)


class Noise:
    """Smooth deterministic noise from a sum of randomly oriented sine waves (2D or 3D)."""

    def __init__(self, seed, octaves=5, base=1 / 90, waves=4, dim=2):
        r = np.random.default_rng(seed)
        self.k, self.ph, self.amp = [], [], []
        amp = 1.0
        for o in range(octaves):
            for _ in range(waves):
                d = norm(r.normal(size=dim))
                self.k.append(d * base * 2 ** o * 2 * np.pi)
                self.ph.append(r.uniform(0, 2 * np.pi))
                self.amp.append(amp / waves)
            amp *= 0.5
        self.scale = 1.6 / sum(self.amp)

    def __call__(self, *coords):
        coords = np.broadcast_arrays(*[np.asarray(c, float) for c in coords])
        out = np.zeros(coords[0].shape)
        for k, ph, a in zip(self.k, self.ph, self.amp):
            out += a * np.sin(sum(ki * c for ki, c in zip(k, coords)) + ph)
        return out * self.scale


COLOR_ATTRS = {"pcol", "fc"}


class MeshBuilder:
    """Accumulates numpy geometry (with per-face materials/attributes) into one Blender mesh."""

    def __init__(self):
        self.chunks, self.nv, self.mats = [], 0, []

    def add(self, verts, faces, mat=None, smooth=False, pattrs=None, mat_index=None, **fattrs):
        """`mat` is one material, or a list indexed per face by `mat_index`."""
        verts = np.asarray(verts, float).reshape(-1, 3)
        groups = faces if isinstance(faces, (list, tuple)) else [faces]
        groups = [np.atleast_2d(np.asarray(g, np.int64)) for g in groups if np.size(g)]
        nf = sum(len(g) for g in groups)
        if not len(verts) or not nf:
            return
        gidx = []
        for m in (mat if isinstance(mat, (list, tuple)) else [mat]):
            if m is not None and m not in self.mats:
                self.mats.append(m)
            gidx.append(self.mats.index(m) if m is not None else 0)
        gidx = np.array(gidx, np.int32)
        mi = gidx[np.asarray(mat_index)] if mat_index is not None else np.full(nf, gidx[0], np.int32)
        fa = {}
        for k, v in fattrs.items():
            v = np.asarray(v, np.float32)
            if k in COLOR_ATTRS:
                if v.shape[-1] == 3:
                    v = np.concatenate([v, np.ones(v.shape[:-1] + (1,), np.float32)], -1)
                v = np.broadcast_to(v.reshape(-1, 4), (nf, 4)) if v.size == 4 else v
            else:
                v = np.broadcast_to(v, (nf,))
            fa[k] = v
        pa = {k: np.broadcast_to(np.asarray(v, np.float32), (len(verts),)) for k, v in (pattrs or {}).items()}
        self.chunks.append(dict(v=verts, f=[g + self.nv for g in groups], nf=nf, mi=mi, smooth=smooth, fa=fa, pa=pa))
        self.nv += len(verts)

    def build(self, name, coll=None):
        if not self.chunks:
            return None
        V = np.concatenate([c["v"] for c in self.chunks])
        groups = [g for c in self.chunks for g in c["f"]]
        sizes = np.concatenate([np.full(len(g), g.shape[1], np.int32) for g in groups])
        loops = np.concatenate([g.ravel() for g in groups]).astype(np.int32)
        starts = np.zeros(len(sizes), np.int32)
        starts[1:] = np.cumsum(sizes)[:-1]
        if loops.min() < 0 or loops.max() >= len(V):
            raise ValueError(f"{name}: face index out of range")
        me = bpy.data.meshes.new(name)
        me.vertices.add(len(V))
        me.vertices.foreach_set("co", V.astype(np.float32).ravel())
        me.loops.add(len(loops))
        me.loops.foreach_set("vertex_index", loops)
        me.polygons.add(len(sizes))
        me.polygons.foreach_set("loop_start", starts)
        me.polygons.foreach_set("loop_total", sizes)
        me.polygons.foreach_set("material_index", np.concatenate([c["mi"] for c in self.chunks]))
        me.polygons.foreach_set("use_smooth",
                                np.concatenate([np.full(c["nf"], c["smooth"], bool) for c in self.chunks]))
        for k in sorted({k for c in self.chunks for k in c["fa"]}):
            color = k in COLOR_ATTRS
            parts = [c["fa"].get(k, np.zeros((c["nf"], 4) if color else c["nf"], np.float32)) for c in self.chunks]
            at = me.attributes.new(k, "FLOAT_COLOR" if color else "FLOAT", "FACE")
            at.data.foreach_set("color" if color else "value", np.ascontiguousarray(np.concatenate(parts)).ravel())
        for k in sorted({k for c in self.chunks for k in c["pa"]}):
            parts = [c["pa"].get(k, np.zeros(len(c["v"]), np.float32)) for c in self.chunks]
            me.attributes.new(k, "FLOAT", "POINT").data.foreach_set("value", np.ascontiguousarray(np.concatenate(parts)))
        me.update(calc_edges=True)
        for m in self.mats:
            me.materials.append(m)
        ob = bpy.data.objects.new(name, me)
        (coll or bpy.context.scene.collection).objects.link(ob)
        return ob


def grid_faces(nu, nv, wrap_u=False, wrap_v=False, flip=False):
    i, j = np.meshgrid(np.arange(nu if wrap_u else nu - 1), np.arange(nv if wrap_v else nv - 1), indexing="ij")
    i1, j1 = (i + 1) % nu, (j + 1) % nv
    a, b, c, d = i * nv + j, i1 * nv + j, i1 * nv + j1, i * nv + j1
    return np.stack([a, d, c, b] if flip else [a, b, c, d], -1).reshape(-1, 4)


def rmf(P):
    """Rotation-minimising frames along a polyline."""
    T = norm(np.gradient(P, axis=0))
    ref = np.array([0.0, 0.0, 1.0]) if abs(T[0, 2]) < 0.95 else np.array([1.0, 0.0, 0.0])
    n = norm(np.cross(T[0], ref))
    N = np.empty_like(P)
    for i in range(len(P)):
        n = n - T[i] * np.dot(n, T[i])
        n /= np.linalg.norm(n)
        N[i] = n
    return T, N, np.cross(T, N)


def tube(P, R, sides=8, caps=True):
    P = np.asarray(P, float)
    R = np.broadcast_to(np.asarray(R, float), (len(P),))
    T, N, B = rmf(P)
    th = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    ring = np.cos(th)[None, :, None] * N[:, None] + np.sin(th)[None, :, None] * B[:, None]
    V = (P[:, None] + R[:, None, None] * ring).reshape(-1, 3)
    F = [grid_faces(len(P), sides, wrap_v=True, flip=True)]
    if caps:
        F += [np.arange(sides)[::-1][None], ((len(P) - 1) * sides + np.arange(sides))[None]]
    return V, F


def sweep(path, prof, closed_prof=True, closed_path=False, caps=True, up=(0, 0, 1)):
    """Sweep a 2D profile (lateral, vertical) along a path; lateral = left of travel."""
    path, prof = np.asarray(path, float), np.asarray(prof, float)
    if closed_path:
        T = norm(np.roll(path, -1, 0) - np.roll(path, 1, 0))
    else:
        T = norm(np.gradient(path, axis=0))
    S = norm(np.cross(np.asarray(up, float)[None], T))
    U = np.cross(T, S)
    V = path[:, None] + prof[None, :, 0, None] * S[:, None] + prof[None, :, 1, None] * U[:, None]
    F = [grid_faces(len(path), len(prof), wrap_u=closed_path, wrap_v=closed_prof)]
    if caps and closed_prof and not closed_path:
        m = len(prof)
        F += [np.arange(m)[None], ((len(path) - 1) * m + np.arange(m)[::-1])[None]]
    return V.reshape(-1, 3), F


BOX_O = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                  [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
BOX_F = np.array([[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])


def boxes(C, S, rotz=None):
    C = np.atleast_2d(np.asarray(C, float))
    S = np.broadcast_to(np.asarray(S, float), C.shape)
    o = BOX_O[None] * S[:, None] / 2
    if rotz is not None:
        c, s = np.cos(rotz)[:, None], np.sin(rotz)[:, None]
        o = np.stack([o[..., 0] * c - o[..., 1] * s, o[..., 0] * s + o[..., 1] * c, o[..., 2]], -1)
    V = C[:, None] + o
    F = BOX_F[None] + 8 * np.arange(len(C))[:, None, None]
    return V.reshape(-1, 3), F.reshape(-1, 4)


BEAM_F = np.array([[0, 1, 2, 3], [7, 6, 5, 4], [0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0]])


def beams(P0, P1, up, w, h):
    """Rectangular members between point pairs, depth oriented along `up`."""
    P0, P1 = np.atleast_2d(P0), np.atleast_2d(P1)
    d = norm(P1 - P0)
    s = norm(np.cross(d, np.broadcast_to(up, d.shape)))
    u = np.cross(s, d)
    cs = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float)
    off = cs[None, :, 0, None] * (w / 2) * s[:, None] + cs[None, :, 1, None] * (h / 2) * u[:, None]
    V = np.concatenate([P0[:, None] + off, P1[:, None] + off], 1)
    F = BEAM_F[None] + 8 * np.arange(len(P0))[:, None, None]
    return V.reshape(-1, 3), F.reshape(-1, 4)


def lathe(rz, sides=24, center=(0, 0, 0), caps=True):
    rz = np.asarray(rz, float)
    th = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    V = np.stack([rz[:, 0, None] * np.cos(th), rz[:, 0, None] * np.sin(th),
                  np.broadcast_to(rz[:, 1, None], (len(rz), sides))], -1).reshape(-1, 3) + center
    F = [grid_faces(len(rz), sides, wrap_v=True)]
    if caps:
        F += [np.arange(sides)[::-1][None], ((len(rz) - 1) * sides + np.arange(sides))[None]]
    return V, F


_ICO = {}


def ico(sub):
    if sub not in _ICO:
        bm = bmesh.new()
        bmesh.ops.create_icosphere(bm, subdivisions=sub, radius=1.0)
        V = np.array([v.co[:] for v in bm.verts])
        F = np.array([[v.index for v in f.verts] for f in bm.faces])
        bm.free()
        _ICO[sub] = (V, F)
    return _ICO[sub]


def catmull(points, step=1.0):
    P = np.asarray(points, float)
    P = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1:i + 3]
        n = max(2, int(np.linalg.norm(p2 - p1) / step))
        t = np.linspace(0, 1, n, endpoint=False)[:, None]
        out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(P[-2][None])
    return np.vstack(out)


def bezier(A, B, C1, C2, n=10):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 3 * A + 3 * (1 - t) ** 2 * t * C1 + 3 * (1 - t) * t ** 2 * C2 + t ** 3 * B


def dist_polyline_xy(px, py, poly):
    P = np.stack([np.ravel(px), np.ravel(py)], -1)
    A, B = poly[:-1, :2], poly[1:, :2]
    ab = B - A
    out = np.empty(len(P))
    for s in range(0, len(P), 8000):
        p = P[s:s + 8000, None]
        t = np.clip(((p - A) * ab).sum(-1) / (ab * ab).sum(-1).clip(1e-9), 0, 1)
        out[s:s + 8000] = np.linalg.norm(p - (A + t[..., None] * ab), axis=-1).min(1)
    return out.reshape(np.shape(px))


def new_object(name, data, coll=None, loc=(0, 0, 0), rot=(0, 0, 0), scale=1.0):
    ob = bpy.data.objects.new(name, data)
    (coll or bpy.context.scene.collection).objects.link(ob)
    ob.location, ob.rotation_euler = loc, rot
    ob.scale = (scale, scale, scale) if np.isscalar(scale) else scale
    return ob


def collection(name, parent=None):
    c = bpy.data.collections.new(name)
    (parent or bpy.context.scene.collection).children.link(c)
    return c


# =============================================================== site layout
NZ_FINE = Noise(11, octaves=5, base=1 / 55)
NZ_HILL = Noise(23, octaves=6, base=1 / 420)


def stream_y(x):
    x = np.asarray(x, float)
    return 6.0 * np.sin(x / 41.0 + 0.4) + 3.0 * np.sin(x / 17.0 + 1.3)


def trail_y(x):
    return stream_y(x) + 9.5 + 1.5 * np.sin(np.asarray(x, float) / 29.0)


def water_z(x):
    return -0.95 - 0.003 * np.asarray(x, float)


def box_mask(x, y, x0, x1, y0, y1, soft):
    return (smoothstep(x0 - soft, x0, x) * (1 - smoothstep(x1, x1 + soft, x))
            * smoothstep(y0 - soft, y0, y) * (1 - smoothstep(y1, y1 + soft, y)))


def terrace_mask(x, y):
    return np.maximum(box_mask(x, y, -104, 104, 37.0, 120, 9), box_mask(x, y, -104, 104, -60, -37.0, 9))


def terrain_raw(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ay, d = np.abs(y), np.abs(y - stream_y(x))
    h = 9.0 * smoothstep(7, 30, d) + 4.5 * smoothstep(28, 72, ay)
    h += 46 * smoothstep(120, 380, ay) * (0.62 + 0.38 * NZ_HILL(x, y))
    h += 55 * smoothstep(520, 1250, np.abs(x)) * (0.65 + 0.35 * NZ_HILL(y + 300, x))
    h += NZ_FINE(x, y) * (0.3 + 1.7 * smoothstep(25, 90, ay))
    h += -0.003 * x
    h -= 1.8 * (1 - smoothstep(2.2, 5.2, d))
    return h


def terrain_h(x, y):
    m = terrace_mask(x, y)
    return terrain_raw(x, y) * (1 - m) + (GRADE - 0.1) * m


# Station roof: a diagrid gridshell over a (u, v) parameter rectangle.
ROOF_L, ROOF_W0, NU, NV = 216.0, 106.0, 90, 44


def roof_w(u):
    return ROOF_W0 * (1 + 0.05 * np.sin(2 * np.pi * np.asarray(u) + 0.5))


def roof_uv_xyz(u, v):
    u, v = np.asarray(u, float), np.asarray(v, float)
    x = (u - 0.5) * ROOF_L
    y = (v - 0.5) * roof_w(u)
    arch = np.clip(np.sin(np.pi * v), 0, 1) ** 0.75
    wave = np.sin(2 * np.pi * 2.0 * u + 1.7 * (v - 0.5) + 0.3)
    lift = 4.5 * (smoothstep(0.82, 1.0, u) + smoothstep(0.18, 0.0, u))
    z = 22.0 + 11.5 * arch + 3.0 * wave * (0.3 + 0.7 * arch) + lift * np.sqrt(arch)
    return np.stack(np.broadcast_arrays(x, y, z), -1)


def roof_wave(u, v):
    return np.sin(2 * np.pi * 2.0 * u + 1.7 * (v - 0.5) + 0.3)


def roof_uv_of(x, y):
    u = np.asarray(x, float) / ROOF_L + 0.5
    return u, np.asarray(y, float) / roof_w(u) + 0.5


def roof_z_at(x, y):
    return roof_uv_xyz(*roof_uv_of(x, y))[..., 2]


def roof_frame(u, v, e=1e-4):
    P = roof_uv_xyz(u, v)
    N = norm(np.cross(roof_uv_xyz(u + e, v) - roof_uv_xyz(u - e, v), roof_uv_xyz(u, v + e) - roof_uv_xyz(u, v - e)))
    return P, N


VALLEY_COLS = [(x, float(stream_y(x)) - 8.5) for x in (-81.0, -27.0, 27.0, 81.0)]
SIDE_COLS = [(x, s * 43.0) for s in (1, -1) for x in (-90.0, -54.0, -18.0, 18.0, 54.0, 90.0)]
EMERGENT = [(x, float(stream_y(x)) - 9.0) for x in (-54.0, 54.0)]
OCULI = [(x, y, 11.0, 9.5) for x, y in EMERGENT]
WALK_WEST = [(-6, 11, 19.6), (-16, 12.5, 19.3), (-32, 15, 18.6), (-50, 15.5, 17.6), (-66, 12.5, 16.6),
             (-82, 13.5, 15.6), (-96, 14.5, 14.6), (-104, 14.0, 14.2)]
WALK_EAST = [(6, 11, 19.6), (16, 12.5, 19.3), (32, 15, 18.6), (50, 15.5, 17.8), (68, 13, 17.0),
             (86, 13.5, 16.2), (98, 9, 15.8), (104, 3.5, 15.5), (108.5, -0.5, 15.4)]
TOWER = (-104.0, 12.6)
LOOKOUT = (111.0, -4.8, 15.4)   # cantilevered lookout on the south bank at the east mouth
BRIDGE_X = 6.0      # sky bridge spans x in [-6, 6]
BRIDGE_Z = 19.6
POOL = (0.0, 80.0, 44.0, 9.5)   # stadium: centre x, y, half length, end radius
LAWNS = [(-101.0, -58.0, 54.0, 115.0), (58.0, 101.0, 54.0, 115.0)]   # x0, x1, y0, y1 on the north plaza


def stadium(cx, cy, hl, r, n=120):
    t = np.linspace(0, 1, n, endpoint=False)
    straight = 2 * (hl - r)
    per = 2 * straight + 2 * np.pi * r
    s = t * per
    pts = []
    for si in s:
        if si < straight:
            pts.append((cx - (hl - r) + si, cy - r))
        elif si < straight + np.pi * r:
            a = -np.pi / 2 + (si - straight) / r
            pts.append((cx + (hl - r) + r * np.cos(a), cy + r * np.sin(a)))
        elif si < 2 * straight + np.pi * r:
            pts.append((cx + (hl - r) - (si - straight - np.pi * r), cy + r))
        else:
            a = np.pi / 2 + (si - 2 * straight - np.pi * r) / r
            pts.append((cx - (hl - r) + r * np.cos(a), cy + r * np.sin(a)))
    return np.array(pts)


def in_stadium(x, y, cx, cy, hl, r, pad=0.0):
    dx = np.clip(np.abs(np.asarray(x) - cx) - (hl - r), 0, None)
    return np.hypot(dx, np.asarray(y) - cy) < r + pad


# =============================================================== materials
M = {}


def _sock(sockets, key):
    if isinstance(key, int):
        return sockets[key]
    for s in sockets:
        if (s.name == key or s.identifier == key) and s.enabled:
            return s
    raise KeyError(key)


def node(nt, kind, inputs=None, name=None, **props):
    n = nt.nodes.new(kind)
    if name:
        n.name = n.label = name
    for k, v in props.items():
        setattr(n, k, v)
    for k, v in (inputs or {}).items():
        sock = _sock(n.inputs, k)
        if isinstance(v, bpy.types.NodeSocket):
            nt.links.new(v, sock)
        else:
            if isinstance(v, (tuple, list)) and len(v) == 3 and sock.type == "RGBA":
                v = (*v, 1.0)
            sock.default_value = v
    return n


def out(n, key=0):
    return _sock(n.outputs, key)


def new_mat(name):
    m = bpy.data.materials.new(name)
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    o = nt.nodes.new("ShaderNodeOutputMaterial")
    M[name] = m
    return m, nt, o


def math_node(nt, op, a, b=None, c=None, **kw):
    ins = {k: v for k, v in enumerate((a, b, c)) if v is not None}
    return out(node(nt, "ShaderNodeMath", ins, operation=op, **kw))


def lights_value(nt):
    return out(node(nt, "ShaderNodeValue", name="LIGHTS"))


def principled(name, color, rough=0.5, metal=0.0, **inputs):
    m, nt, o = new_mat(name)
    p = node(nt, "ShaderNodeBsdfPrincipled", {"Base Color": color, "Roughness": rough, "Metallic": metal, **inputs})
    nt.links.new(out(p), o.inputs["Surface"])
    return m, nt, p, o


def noise_tex(nt, scale, detail=4.0, coord=None, rough=0.55):
    ins = {"Scale": scale, "Detail": detail, "Roughness": rough}
    if coord is not None:
        ins["Vector"] = coord
    return node(nt, "ShaderNodeTexNoise", ins)


def ramp(nt, fac, stops):
    r = node(nt, "ShaderNodeValToRGB", {"Fac": fac})
    els = r.color_ramp.elements
    while len(els) < len(stops):
        els.new(0.5)
    for el, (pos, col) in zip(els, stops):
        el.position = pos
        el.color = (*col, 1.0) if len(col) == 3 else col
    return out(r, "Color")


def bump(nt, height, strength=0.2, dist=0.1):
    return out(node(nt, "ShaderNodeBump", {"Height": height, "Strength": strength, "Distance": dist}))


def obj_coord(nt):
    return out(node(nt, "ShaderNodeTexCoord"), "Object")


def glass_like(nt, surface, shadow_color=(0.9, 0.95, 0.93)):
    """Let shadow/diffuse rays pass straight through thin glazing (no caustics needed)."""
    lp = node(nt, "ShaderNodeLightPath")
    fac = math_node(nt, "MAXIMUM", out(lp, "Is Shadow Ray"), out(lp, "Is Diffuse Ray"))
    tr = node(nt, "ShaderNodeBsdfTransparent", {"Color": shadow_color})
    return out(node(nt, "ShaderNodeMixShader", {0: fac, 1: surface, 2: out(tr)}))


def build_materials():
    # glulam timber
    m, nt, p, o = principled("Timber", (0.6, 0.4, 0.22), 0.5)
    co = obj_coord(nt)
    wave = node(nt, "ShaderNodeTexWave", {"Vector": co, "Scale": 1.6, "Distortion": 6.0, "Detail": 3.0})
    nt.links.new(ramp(nt, out(wave, "Fac"), [(0.0, (0.46, 0.27, 0.13)), (1.0, (0.74, 0.52, 0.31))]), p.inputs["Base Color"])
    nt.links.new(bump(nt, out(wave, "Fac"), 0.08, 0.02), p.inputs["Normal"])

    principled("SteelWhite", (0.82, 0.83, 0.81), 0.28, 0.0, **{"Coat Weight": 0.25})
    principled("SteelDark", (0.08, 0.09, 0.1), 0.35, 0.8)
    principled("Corten", (0.28, 0.12, 0.05), 0.75, 0.2)

    # thin clear glazing
    m, nt, o = new_mat("Glass")
    g = node(nt, "ShaderNodeBsdfPrincipled", {"Base Color": (0.88, 0.96, 0.93), "Roughness": 0.02,
                                              "Transmission Weight": 1.0, "IOR": 1.5, "Thin Wall": True})
    nt.links.new(glass_like(nt, out(g)), o.inputs["Surface"])

    # semi-transparent PV glazing: opaque cells, clear gaps
    m, nt, o = new_mat("SolarGlass")
    co = obj_coord(nt)
    br = node(nt, "ShaderNodeTexBrick", {"Vector": co, "Scale": 3.0, "Mortar Size": 0.035, "Mortar Smooth": 0.0,
                                         "Brick Width": 0.5, "Row Height": 0.5}, offset=0.0, squash=1.0)
    big = node(nt, "ShaderNodeTexBrick", {"Vector": co, "Scale": 0.42, "Mortar Size": 0.06, "Mortar Smooth": 0.0,
                                          "Brick Width": 0.5, "Row Height": 0.5}, offset=0.0)
    gap = math_node(nt, "MAXIMUM", out(br, "Fac"), out(big, "Fac"))   # brick Fac is 1 in the mortar lines
    cell = node(nt, "ShaderNodeBsdfPrincipled", {"Base Color": (0.015, 0.025, 0.06), "Roughness": 0.18,
                                                 "Metallic": 0.3, "Coat Weight": 1.0, "Coat Roughness": 0.03})
    clear = node(nt, "ShaderNodeBsdfPrincipled", {"Base Color": (0.88, 0.95, 0.93), "Roughness": 0.02,
                                                  "Transmission Weight": 1.0, "Thin Wall": True})
    surf = out(node(nt, "ShaderNodeMixShader", {0: gap, 1: out(cell), 2: out(clear)}))
    tr = node(nt, "ShaderNodeBsdfTransparent", {"Color": (0.9, 0.95, 0.93)})
    shadow = out(node(nt, "ShaderNodeMixShader", {0: gap, 1: out(cell), 2: out(tr)}))
    lp = node(nt, "ShaderNodeLightPath")
    nt.links.new(out(node(nt, "ShaderNodeMixShader", {0: out(lp, "Is Shadow Ray"), 1: surf, 2: shadow})), o.inputs["Surface"])

    # concretes
    for name, base, lo, hi in (("Concrete", 0.62, (0.55, 0.54, 0.51), (0.68, 0.67, 0.63)),
                               ("ConcreteDark", 0.45, (0.38, 0.38, 0.37), (0.5, 0.5, 0.48))):
        m, nt, p, o = principled(name, (base, base, base), 0.7)
        nz = noise_tex(nt, 0.6, 8.0, obj_coord(nt))
        nt.links.new(ramp(nt, out(nz, "Fac"), [(0.3, lo), (0.7, hi)]), p.inputs["Base Color"])
        nt.links.new(bump(nt, out(nz, "Fac"), 0.05, 0.05), p.inputs["Normal"])

    # plaza paving: big granite slabs
    m, nt, p, o = principled("Paving", (0.36, 0.34, 0.32), 0.45)
    co = obj_coord(nt)
    br = node(nt, "ShaderNodeTexBrick", {"Vector": co, "Scale": 1.0, "Brick Width": 1.6, "Row Height": 0.8,
                                         "Mortar Size": 0.012, "Color1": (0.38, 0.355, 0.33), "Color2": (0.27, 0.26, 0.25),
                                         "Mortar": (0.12, 0.12, 0.12), "Bias": 0.0}, offset=0.5)
    nz = noise_tex(nt, 12.0, 6.0, co)
    col = node(nt, "ShaderNodeMix", {"Factor": 0.15, "A": out(br, "Color"), "B": out(nz, "Color")},
               data_type="RGBA", blend_type="OVERLAY")
    nt.links.new(out(col, "Result"), p.inputs["Base Color"])
    nt.links.new(bump(nt, math_node(nt, "SUBTRACT", 1.0, out(br, "Fac")), 0.3, 0.01), p.inputs["Normal"])
    nt.links.new(math_node(nt, "MULTIPLY_ADD", out(nz, "Fac"), 0.3, 0.3), p.inputs["Roughness"])

    principled("Granite", (0.035, 0.037, 0.04), 0.25)
    principled("Mullion", (0.1, 0.11, 0.12), 0.35, 0.8)
    m, nt, p, o = principled("Terrazzo", (0.33, 0.32, 0.3), 0.22)
    vor = node(nt, "ShaderNodeTexVoronoi", {"Vector": obj_coord(nt), "Scale": 28.0})
    nt.links.new(ramp(nt, out(vor, "Distance"), [(0.0, (0.55, 0.53, 0.5)), (0.12, (0.3, 0.29, 0.28)),
                                                 (0.5, (0.36, 0.35, 0.33))]), p.inputs["Base Color"])
    m, nt, p, o = principled("Lawn", (0.1, 0.2, 0.04), 0.8)
    nz = noise_tex(nt, 0.25, 6.0, obj_coord(nt))
    nt.links.new(ramp(nt, out(nz, "Fac"), [(0.35, (0.07, 0.15, 0.03)), (0.65, (0.15, 0.25, 0.05))]), p.inputs["Base Color"])
    principled("StoneLight", (0.72, 0.7, 0.66), 0.5)

    # reflecting pool
    m, nt, p, o = principled("PoolWater", (0.004, 0.008, 0.008), 0.0, **{"IOR": 1.33})
    nz = noise_tex(nt, 0.9, 2.0, obj_coord(nt))
    nt.links.new(bump(nt, out(nz, "Fac"), 0.015, 0.1), p.inputs["Normal"])

    # stream water
    m, nt, o = new_mat("StreamWater")
    co = obj_coord(nt)
    mp = node(nt, "ShaderNodeMapping", {"Vector": co, "Scale": (0.35, 1.0, 1.0)})
    nz = noise_tex(nt, 1.6, 6.0, out(mp))
    w = node(nt, "ShaderNodeBsdfPrincipled", {"Base Color": (0.55, 0.72, 0.62), "Roughness": 0.02,
                                              "Transmission Weight": 1.0, "IOR": 1.33})
    nt.links.new(bump(nt, out(nz, "Fac"), 0.25, 0.1), w.inputs["Normal"])
    nt.links.new(out(w), o.inputs["Surface"])
    vol = node(nt, "ShaderNodeVolumeAbsorption", {"Color": (0.6, 0.8, 0.65), "Density": 0.2})
    nt.links.new(out(vol), o.inputs["Volume"])

    # terrain: grass / moss / soil / rock by slope + wetness
    m, nt, p, o = principled("Terrain", (0.12, 0.2, 0.05), 0.85)
    co = obj_coord(nt)
    geo = node(nt, "ShaderNodeNewGeometry")
    nz_z = out(node(nt, "ShaderNodeSeparateXYZ", {0: out(geo, "Normal")}), "Z")
    slope = math_node(nt, "SUBTRACT", 1.0, nz_z)
    n1 = noise_tex(nt, 0.035, 6.0, co)
    n2 = noise_tex(nt, 0.6, 8.0, co)
    grass = ramp(nt, out(n1, "Fac"), [(0.35, (0.09, 0.17, 0.035)), (0.5, (0.16, 0.24, 0.06)), (0.65, (0.24, 0.27, 0.09))])
    wet = out(node(nt, "ShaderNodeAttribute", {}, attribute_name="wet"), "Fac")
    moss = node(nt, "ShaderNodeMix", {"Factor": wet, "A": grass, "B": (0.05, 0.1, 0.03)}, data_type="RGBA")
    soilf = math_node(nt, "MULTIPLY", smooth_fac(nt, slope, 0.12, 0.3), out(n2, "Fac"), use_clamp=True)
    soil = node(nt, "ShaderNodeMix", {"Factor": math_node(nt, "MULTIPLY", soilf, 1.6, use_clamp=True),
                                      "A": out(moss, "Result"), "B": (0.2, 0.15, 0.1)}, data_type="RGBA")
    pebble = node(nt, "ShaderNodeTexVoronoi", {"Vector": co, "Scale": 6.0})
    bedf = out(node(nt, "ShaderNodeAttribute", {}, attribute_name="bed"), "Fac")
    soil = node(nt, "ShaderNodeMix", {"Factor": bedf, "A": out(soil, "Result"),
                                      "B": ramp(nt, out(pebble, "Distance"),
                                                [(0.0, (0.2, 0.17, 0.13)), (0.5, (0.42, 0.37, 0.3))])},
                data_type="RGBA")
    rockf = smooth_fac(nt, slope, 0.32, 0.5)
    rock = node(nt, "ShaderNodeMix", {"Factor": rockf, "A": out(soil, "Result"),
                                      "B": ramp(nt, out(n2, "Fac"), [(0.3, (0.27, 0.26, 0.24)), (0.7, (0.42, 0.41, 0.38))])},
                data_type="RGBA")
    nt.links.new(out(rock, "Result"), p.inputs["Base Color"])
    nt.links.new(bump(nt, out(n2, "Fac"), 0.25, 0.2), p.inputs["Normal"])

    m, nt, p, o = principled("Gravel", (0.45, 0.41, 0.35), 0.9)
    vor = node(nt, "ShaderNodeTexVoronoi", {"Vector": obj_coord(nt), "Scale": 9.0})
    nt.links.new(ramp(nt, out(vor, "Distance"), [(0.0, (0.3, 0.27, 0.23)), (0.6, (0.55, 0.51, 0.45))]), p.inputs["Base Color"])
    nt.links.new(bump(nt, out(vor, "Distance"), 0.4, 0.05), p.inputs["Normal"])

    m, nt, p, o = principled("Bark", (0.16, 0.12, 0.09), 0.85)
    nz = noise_tex(nt, 6.0, 6.0, obj_coord(nt))
    nt.links.new(ramp(nt, out(nz, "Fac"), [(0.3, (0.1, 0.075, 0.055)), (0.7, (0.26, 0.21, 0.16))]), p.inputs["Base Color"])
    nt.links.new(bump(nt, out(nz, "Fac"), 0.4, 0.05), p.inputs["Normal"])

    def foliage(name, stops, transl=0.28, hue=0.035):
        m, nt, o = new_mat(name)
        lv = out(node(nt, "ShaderNodeAttribute", {}, attribute_name="lv", attribute_type="GEOMETRY"), "Fac")
        oi = node(nt, "ShaderNodeObjectInfo")
        base = ramp(nt, lv, stops)
        hs = node(nt, "ShaderNodeHueSaturation", {
            "Hue": math_node(nt, "MULTIPLY_ADD", out(oi, "Random"), 2 * hue, 0.5 - hue),
            "Value": math_node(nt, "MULTIPLY_ADD", out(oi, "Random"), 0.35, 0.82), "Color": base})
        col = out(hs)
        p = node(nt, "ShaderNodeBsdfPrincipled", {"Base Color": col, "Roughness": 0.5, "Specular IOR Level": 0.35})
        tr = node(nt, "ShaderNodeBsdfTranslucent", {"Color": out(node(nt, "ShaderNodeHueSaturation",
                                                                        {"Saturation": 1.25, "Value": 1.6, "Color": col}))})
        nt.links.new(out(node(nt, "ShaderNodeMixShader", {0: transl, 1: out(p), 2: out(tr)})), o.inputs["Surface"])
        return m

    foliage("Leaf", [(0.0, (0.035, 0.1, 0.018)), (0.55, (0.08, 0.2, 0.035)), (1.0, (0.2, 0.3, 0.055))])
    foliage("Needle", [(0.0, (0.02, 0.06, 0.03)), (1.0, (0.06, 0.13, 0.05))], transl=0.18, hue=0.02)
    foliage("VineLeaf", [(0.0, (0.05, 0.14, 0.025)), (0.7, (0.11, 0.26, 0.04)), (1.0, (0.3, 0.36, 0.07))], hue=0.025)
    principled("LeafCore", (0.025, 0.055, 0.015), 0.9)

    m, nt, o = new_mat("Flower")
    fc = out(node(nt, "ShaderNodeAttribute", {}, attribute_name="fc"), "Color")
    p = node(nt, "ShaderNodeBsdfPrincipled", {"Base Color": fc, "Roughness": 0.5, "Subsurface Weight": 0.2})
    nt.links.new(out(p), o.inputs["Surface"])

    m, nt, p, o = principled("Rock", (0.35, 0.34, 0.31), 0.75)
    co = obj_coord(nt)
    nz = noise_tex(nt, 1.2, 8.0, co)
    crack = node(nt, "ShaderNodeTexVoronoi", {"Vector": co, "Scale": 2.5}, feature="DISTANCE_TO_EDGE")
    geo = node(nt, "ShaderNodeNewGeometry")
    up = out(node(nt, "ShaderNodeSeparateXYZ", {0: out(geo, "Normal")}), "Z")
    rc = ramp(nt, out(nz, "Fac"), [(0.3, (0.24, 0.23, 0.21)), (0.7, (0.46, 0.45, 0.41))])
    mo = node(nt, "ShaderNodeMix", {"Factor": smooth_fac(nt, up, 0.55, 0.85), "A": rc, "B": (0.06, 0.12, 0.03)}, data_type="RGBA")
    nt.links.new(out(mo, "Result"), p.inputs["Base Color"])
    rough = math_node(nt, "ADD", out(nz, "Fac"), smooth_fac(nt, out(crack, "Distance"), 0.0, 0.06))
    nt.links.new(bump(nt, rough, 0.6, 0.08), p.inputs["Normal"])

    m, nt, p, o = principled("Soil", (0.1, 0.16, 0.05), 0.95)
    nz = noise_tex(nt, 2.5, 6.0, obj_coord(nt))
    nt.links.new(ramp(nt, out(nz, "Fac"), [(0.35, (0.08, 0.07, 0.04)), (0.55, (0.09, 0.16, 0.04)), (0.75, (0.16, 0.22, 0.06))]),
                 p.inputs["Base Color"])

    # train
    m, nt, p, o = principled("TrainBody", (0.88, 0.9, 0.9), 0.2, 0.0, **{"Coat Weight": 0.8, "Coat Roughness": 0.05})
    principled("TrainStripe", (0.0, 0.32, 0.18), 0.25, 0.6, **{"Coat Weight": 0.8})
    principled("TrainSkirt", (0.06, 0.07, 0.075), 0.4, 0.5)
    m, nt, p, o = principled("TrainWindow", (0.01, 0.012, 0.015), 0.05, 0.0, **{"Coat Weight": 1.0,
                                                                              "Emission Color": (1.0, 0.82, 0.6)})
    nt.links.new(math_node(nt, "MULTIPLY", lights_value(nt), 7.0), p.inputs["Emission Strength"])
    m, nt, p, o = principled("Headlight", (0.04, 0.045, 0.05), 0.05, 0.0, **{"Coat Weight": 1.0,
                                                                          "Emission Color": (1.0, 0.95, 0.85)})
    nt.links.new(math_node(nt, "MULTIPLY", lights_value(nt), 25.0), p.inputs["Emission Strength"])

    # architectural lighting (off by day)
    for name, color, k in (("LED", (1.0, 0.72, 0.42), 14.0), ("NodeLight", (1.0, 0.78, 0.5), 45.0),
                           ("LampHead", (1.0, 0.8, 0.55), 30.0), ("SignGlow", (0.85, 1.0, 0.9), 6.0)):
        m, nt, p, o = principled(name, (0.85, 0.85, 0.83), 0.3, 0.0, **{"Emission Color": color})
        nt.links.new(math_node(nt, "MULTIPLY", lights_value(nt), k), p.inputs["Emission Strength"])
    m, nt, p, o = principled("Display", (0.01, 0.02, 0.015), 0.1, 0.0, **{"Emission Color": (0.35, 1.0, 0.6),
                                                                          "Emission Strength": 1.5})

    m, nt, o = new_mat("People")
    pc = out(node(nt, "ShaderNodeAttribute", {}, attribute_name="pcol"), "Color")
    nt.links.new(out(node(nt, "ShaderNodeBsdfPrincipled", {"Base Color": pc, "Roughness": 0.6})), o.inputs["Surface"])

    principled("Tactile", (0.75, 0.55, 0.04), 0.5)
    principled("Rubber", (0.03, 0.03, 0.03), 0.6)

    m, nt, o = new_mat("SteelMesh")
    s = node(nt, "ShaderNodeBsdfPrincipled", {"Base Color": (0.5, 0.52, 0.53), "Metallic": 0.9, "Roughness": 0.35})
    tr = node(nt, "ShaderNodeBsdfTransparent")
    nt.links.new(out(node(nt, "ShaderNodeMixShader", {0: 0.72, 1: out(s), 2: out(tr)})), o.inputs["Surface"])


def smooth_fac(nt, x, lo, hi):
    return out(node(nt, "ShaderNodeMapRange", {"Value": x, "From Min": lo, "From Max": hi},
                    interpolation_type="SMOOTHSTEP"), "Result")


# =============================================================== plant library
def leaf_cards(rng, centers, normals, size, aspect=0.55, droop=0.18, spread=0.45):
    """Leaves folded along the midrib: two quads each. Returns verts, faces and a per-face shade value."""
    n = len(centers)
    nrm = norm(normals + rng.normal(0, spread, (n, 3)))
    t = norm(np.cross(nrm, norm(rng.normal(0, 1, (n, 3)))))
    s = np.cross(nrm, t)
    L = (size * rng.uniform(0.7, 1.3, n))[:, None]
    W = L * aspect
    fold = W * 0.22
    sag = nrm * L * droop
    V = np.stack([centers - t * L * 0.5,
                  centers - t * L * 0.1 + s * W * 0.5 - nrm * fold,
                  centers + t * L * 0.25 + s * W * 0.3 - nrm * fold - sag * 0.4,
                  centers + t * L * 0.5 - sag,
                  centers + t * L * 0.25 - s * W * 0.3 - nrm * fold - sag * 0.4,
                  centers - t * L * 0.1 - s * W * 0.5 - nrm * fold], 1)
    base = 6 * np.arange(n)[:, None]
    F = np.concatenate([base + [0, 1, 2, 3], base + [0, 3, 4, 5]])
    shade = rng.random(n)
    return V.reshape(-1, 3), F, np.concatenate([shade, shade])


def tree_broadleaf(seed, height, crown_r, crown_h, n_blobs, r0, leaf=0.42, dens=7.0, flat=0.0):
    """Broadleaf tree whose crown is many overlapping leaf clusters filling an ellipsoid (reads as one mass)."""
    rng = np.random.default_rng(seed)
    mb = MeshBuilder()
    top_z = height - crown_h * 0.5
    zs = np.linspace(-0.4, top_z, 9)
    q = (zs - zs[0]) / (zs[-1] - zs[0])
    lean = rng.normal(0, 0.03 * height, 2)
    P = np.c_[lean[0] * q ** 1.6, lean[1] * q ** 1.6, zs]
    R = r0 * (1 - 0.55 * q) * (1 + 0.6 * np.exp(-np.maximum(zs, 0) * 2.5))
    mb.add(*tube(P, R, 10), mat=M["Bark"], smooth=True)
    cz = height - crown_h * 0.5
    nb = int(n_blobs * 1.8)
    u = norm(rng.normal(0, 1, (nb, 3)))
    rad = np.sqrt(rng.uniform(0.3, 1.0, nb))
    hz = crown_h * 0.5 * (1 - 0.35 * flat)
    C = np.c_[lean[0] + u[:, 0] * rad * crown_r * 0.68, lean[1] + u[:, 1] * rad * crown_r * 0.68,
              cz + u[:, 2] * rad * hz * 0.62]
    C[0] = (lean[0], lean[1], height - crown_h * 0.28)
    RB = crown_r * rng.uniform(0.3, 0.45, nb)
    Vi, Fi = ico(2)
    for c, rb in zip(C, RB):
        k = int(rng.integers(5, 9))
        start = P[min(k, len(P) - 1)]
        end = start + (c - start) * 0.85
        L = np.linalg.norm(end - start)
        path = bezier(start, end, start + np.array([0, 0, L * 0.35]), end - (end - start) * 0.3, 7)
        mb.add(*tube(path, np.linspace(R[min(k, len(R) - 1)] * 0.6, r0 * 0.1, 7), 6), mat=M["Bark"], smooth=True)
        disp = 1 + 0.18 * np.sin(Vi @ rng.normal(0, 3.0, 3)) * np.sin(Vi @ rng.normal(0, 3.0, 3))
        mb.add(c + Vi * disp[:, None] * rb * 0.8 * np.array([1, 1, 0.85]), Fi, mat=M["LeafCore"], smooth=True)
        n = int(4 * np.pi * rb * rb * dens)
        dirs = norm(rng.normal(0, 1, (n, 3)))
        dirs = dirs[dirs[:, 2] > -0.7]
        pts = c + dirs * rb * rng.uniform(0.8, 1.15, (len(dirs), 1)) * np.array([1, 1, 0.85])
        outward = norm(pts - [lean[0], lean[1], cz])
        V, F, lv = leaf_cards(rng, pts, dirs + outward * 0.6 + np.array([0, 0, 0.45]), leaf)
        mb.add(V, F, mat=M["Leaf"], lv=lv)
    return mb


def tree_emergent(seed, height=38.0):
    """Giant valley tree: a wide lower crown under the roof and an upper crown rising through an oculus."""
    rng = np.random.default_rng(seed)
    mb = MeshBuilder()
    zs = np.linspace(-0.6, height - 4.0, 12)
    q = (zs - zs[0]) / (zs[-1] - zs[0])
    P = np.c_[0.4 * np.sin(q * 2.0), 0.3 * np.sin(q * 1.3 + 0.5), zs]
    R = 1.0 * (1 - 0.6 * q) * (1 + 1.1 * np.exp(-np.maximum(zs, 0) * 0.9))
    mb.add(*tube(P, R, 16), mat=M["Bark"], smooth=True)
    for k in range(6):          # buttress roots
        a = k * np.pi / 3 + rng.normal(0, 0.15)
        d = np.array([np.cos(a), np.sin(a), 0.0])
        root = np.array([d * t * 3.2 + [0, 0, 2.6 * (1 - t) ** 2 - 0.3] for t in np.linspace(0, 1, 7)])
        mb.add(*tube(root, np.linspace(0.7, 0.15, 7), 8), mat=M["Bark"], smooth=True)
    blobs = []
    for k in range(8):
        a = 2 * np.pi * k / 8 + rng.normal(0, 0.2)
        r = rng.uniform(4.8, 6.8)
        blobs.append(((r * np.cos(a), r * np.sin(a), rng.uniform(21.5, 25.5)), rng.uniform(2.8, 3.5), rng.uniform(14, 20)))
    for k in range(5):
        a = 2 * np.pi * k / 5 + rng.normal(0, 0.3)
        r = rng.uniform(1.5, 3.6)
        blobs.append(((r * np.cos(a), r * np.sin(a), rng.uniform(31.0, 35.0)), rng.uniform(2.8, 3.4), rng.uniform(25, 29)))
    blobs.append(((0.0, 0.0, height - 1.5), 3.4, height - 6.0))
    Vi, Fi = ico(2)
    for c, rb, zb in blobs:
        c = np.array(c)
        start = np.array([np.interp(zb, zs, P[:, 0]), np.interp(zb, zs, P[:, 1]), zb])
        end = start + (c - start) * 0.85
        L = np.linalg.norm(end - start)
        path = bezier(start, end, start + np.array([0, 0, L * 0.4]), end - (end - start) * 0.3, 9)
        mb.add(*tube(path, np.linspace(0.42, 0.1, 9), 8), mat=M["Bark"], smooth=True)
        disp = 1 + 0.18 * np.sin(Vi @ rng.normal(0, 3.0, 3)) * np.sin(Vi @ rng.normal(0, 3.0, 3))
        mb.add(c + Vi * disp[:, None] * rb * 0.8 * np.array([1, 1, 0.75]), Fi, mat=M["LeafCore"], smooth=True)
        n = int(4 * np.pi * rb * rb * 6.5)
        dirs = norm(rng.normal(0, 1, (n, 3)))
        dirs = dirs[dirs[:, 2] > -0.6]
        pts = c + dirs * rb * rng.uniform(0.85, 1.08, (len(dirs), 1)) * np.array([1, 1, 0.75])
        V, F, lv = leaf_cards(rng, pts, dirs + np.array([0, 0, 0.5]), 0.5)
        mb.add(V, F, mat=M["Leaf"], lv=lv)
    return mb


def tree_conifer(seed, height, base_r, r0=0.25):
    rng = np.random.default_rng(seed)
    mb = MeshBuilder()
    zs = np.linspace(-0.4, height * 0.98, 6)
    mb.add(*tube(np.c_[np.zeros(6), np.zeros(6), zs], r0 * (1 - 0.8 * zs / height).clip(0.2), 8), mat=M["Bark"], smooth=True)
    z0 = height * 0.16
    q = np.linspace(0, 1, 44)
    tiers = 9
    saw = 1 - (q * tiers % 1.0)
    r = base_r * (1 - q) ** 0.9 * (0.72 + 0.28 * saw)
    z = z0 + q * (height - z0)
    mb.add(*lathe(np.c_[r * 0.85, z], 14), mat=M["LeafCore"], smooth=True)
    area = np.pi * base_r * (height - z0)
    n = int(area * 6.0)
    qq = 1 - np.sqrt(rng.random(n))
    rr = base_r * (1 - qq) ** 0.9 * (0.72 + 0.28 * (1 - (qq * tiers % 1.0)))
    a = rng.uniform(0, 2 * np.pi, n)
    pts = np.c_[rr * np.cos(a), rr * np.sin(a), z0 + qq * (height - z0)]
    nrm = np.c_[np.cos(a), np.sin(a), np.full(n, 0.5)]
    V, F, lv = leaf_cards(rng, pts, nrm, 0.6, aspect=0.32, droop=0.3, spread=0.3)
    mb.add(V, F, mat=M["Needle"], lv=lv)
    return mb


def shrub(seed, h, r, leaf=0.16, dens=40.0):
    rng = np.random.default_rng(seed)
    mb = MeshBuilder()
    nb = 4
    C = np.c_[rng.normal(0, r * 0.35, (nb, 2)), rng.uniform(h * 0.35, h * 0.6, nb)]
    for c in C:
        rb = r * rng.uniform(0.5, 0.7)
        V, F = ico(2)
        mb.add(c + V * rb * 0.8 * np.array([1, 1, 0.8]), F, mat=M["LeafCore"], smooth=True)
        n = int(4 * np.pi * rb * rb * dens)
        dirs = norm(rng.normal(0, 1, (n, 3)))
        V, F, lv = leaf_cards(rng, c + dirs * rb * rng.uniform(0.85, 1.1, (n, 1)) * np.array([1, 1, 0.8]),
                              dirs + [0, 0, 0.4], leaf)
        mb.add(V, F, mat=M["Leaf"], lv=lv * 0.8 + 0.2)
    return mb


def fern(seed, n=14):
    rng = np.random.default_rng(seed)
    mb = MeshBuilder()
    for k in range(n):
        az = 2 * np.pi * k / n + rng.normal(0, 0.15)
        L = rng.uniform(0.7, 1.25)
        t = np.linspace(0, 1, 9)
        rise = rng.uniform(0.9, 1.3)
        C = np.c_[np.cos(az) * L * t, np.sin(az) * L * t, L * (rise * t - 1.05 * t ** 2)]
        w = 0.17 * L * np.sin(np.pi * np.minimum(t * 1.1, 1.0)) ** 0.7
        side = np.array([-np.sin(az), np.cos(az), 0.0])
        V = np.concatenate([C + side * w[:, None], C - side * w[:, None]])
        F = np.stack([np.arange(8), np.arange(1, 9), 9 + np.arange(1, 9), 9 + np.arange(8)], -1)
        mb.add(V, F, mat=M["Leaf"], lv=rng.uniform(0.3, 0.9))
    return mb


def grass_tuft(seed, n=34, h=0.55, flowers=0):
    rng = np.random.default_rng(seed)
    mb = MeshBuilder()
    for k in range(n):
        a = rng.uniform(0, 2 * np.pi)
        base = np.array([np.cos(a), np.sin(a), 0]) * rng.uniform(0, 0.12)
        lean = rng.uniform(0.15, 0.6)
        hh = h * rng.uniform(0.6, 1.2)
        t = np.linspace(0, 1, 4)
        C = base + np.c_[np.cos(a) * lean * hh * t ** 2, np.sin(a) * lean * hh * t ** 2, hh * t]
        side = np.array([-np.sin(a), np.cos(a), 0]) * 0.012 * (1 - t)[:, None] * 2
        V = np.concatenate([C + side, C - side])
        F = np.stack([np.arange(3), np.arange(1, 4), 4 + np.arange(1, 4), 4 + np.arange(3)], -1)
        mb.add(V, F, mat=M["Leaf"], lv=rng.uniform(0.5, 1.0))
        if k < flowers:
            Vi, Fi = ico(1)
            col = [(0.95, 0.85, 0.2), (0.95, 0.95, 0.92), (0.55, 0.3, 0.75), (0.9, 0.45, 0.25)][k % 4]
            mb.add(C[-1] + Vi * 0.035, Fi, mat=M["Flower"], fc=col)
    return mb


NZ3 = Noise(5, octaves=3, base=0.25, waves=5, dim=3)


def rock(seed):
    rng = np.random.default_rng(seed)
    V, F = ico(3)
    off = rng.normal(0, 4, 3)
    r = 1 + 0.3 * NZ3(*(V * 1.3 + off).T) + 0.12 * np.abs(NZ3(*(V * 3.5 - off).T))
    V = V * r[:, None] * np.array([1.0, rng.uniform(0.6, 0.9), rng.uniform(0.45, 0.7)]) - [0, 0, 0.18]
    mb = MeshBuilder()
    mb.add(V, F, mat=M["Rock"])
    return mb


def build_library():
    lib = collection("LIB")
    colls = {k: collection(f"LIB_{k}", lib) for k in ("Forest", "Conifer", "Grove", "Under", "Garden", "Rocks")}
    P = {}

    def put(key, name, mb):
        ob = mb.build(name, colls[key] if key else lib)
        P[name] = ob
        return ob

    put("Forest", "TreeA", tree_broadleaf(1, 15.0, 5.6, 10.5, 9, 0.36))
    put("Forest", "TreeB", tree_broadleaf(2, 11.0, 4.3, 7.8, 7, 0.28))
    put("Forest", "TreeC", tree_broadleaf(3, 18.0, 5.0, 12.5, 9, 0.38))
    put("Forest", "TreeD", tree_broadleaf(4, 9.5, 4.0, 6.5, 6, 0.24, flat=0.4))
    put("Grove", "GroveA", tree_broadleaf(5, 11.0, 4.4, 8.2, 10, 0.27))
    put("Grove", "GroveB", tree_broadleaf(6, 8.5, 4.0, 6.2, 9, 0.23, flat=0.5))
    put("Grove", "GroveC", tree_broadleaf(7, 12.5, 3.9, 9.6, 10, 0.24))
    put("Conifer", "PineA", tree_conifer(8, 20.0, 3.6))
    put("Conifer", "PineB", tree_conifer(9, 14.0, 2.8))
    put("Under", "ShrubA", shrub(10, 1.6, 1.3))
    put("Under", "ShrubB", shrub(11, 1.0, 0.9))
    put("Under", "FernA", fern(12))
    put("Under", "FernB", fern(13, 10))
    put("Garden", "GrassA", grass_tuft(14))
    put("Garden", "GrassB", grass_tuft(15, 24, 0.8))
    put("Garden", "Flowers", grass_tuft(16, 26, 0.45, flowers=14))
    put("Garden", "ShrubS", shrub(17, 0.8, 0.6, leaf=0.12))
    for i in range(3):
        put("Rocks", f"Rock{i}", rock(20 + i))
    put(None, "Emergent", tree_emergent(30))
    bpy.context.view_layer.layer_collection.children["LIB"].exclude = True
    return P, colls


def scatter(name, target, coll, density, attr, seed, scale=(0.75, 1.3), dist_min=0.0, tilt=0.05, parent=None):
    """Geometry-nodes scatter of a library collection over `target`, weighted by a point attribute."""
    ng = bpy.data.node_groups.new(name, "GeometryNodeTree")
    ng.is_modifier = True
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nodes = ng.nodes
    gi = nodes.new("NodeGroupInput")
    go = nodes.new("NodeGroupOutput")
    _ = gi
    oi = node(ng, "GeometryNodeObjectInfo", {"Object": target}, transform_space="RELATIVE")
    na = node(ng, "GeometryNodeInputNamedAttribute", {"Name": attr}, data_type="FLOAT")
    sel = node(ng, "FunctionNodeCompare", {"A": out(na, "Attribute"), "B": 0.002}, data_type="FLOAT", operation="GREATER_THAN")
    # crop to faces that can receive points first: Poisson sampling otherwise seeds the whole terrain
    crop = node(ng, "GeometryNodeSeparateGeometry", {"Geometry": out(oi, "Geometry"), "Selection": out(sel)}, domain="FACE")
    dp = node(ng, "GeometryNodeDistributePointsOnFaces",
              {"Mesh": out(crop, "Selection"), "Distance Min": dist_min, "Density Max": density,
               "Density Factor": out(na, "Attribute"), "Seed": seed}, distribute_method="POISSON")
    ci = node(ng, "GeometryNodeCollectionInfo", {"Collection": coll, "Separate Children": True, "Reset Children": True})
    n_inst = len(coll.objects)
    ri = node(ng, "FunctionNodeRandomValue", {"Min": 0, "Max": n_inst - 1, "Seed": seed + 1}, data_type="INT")
    rr = node(ng, "FunctionNodeRandomValue", {"Min": (-tilt, -tilt, 0.0), "Max": (tilt, tilt, 2 * np.pi),
                                              "Seed": seed + 2}, data_type="FLOAT_VECTOR")
    rs = node(ng, "FunctionNodeRandomValue", {"Min": scale[0], "Max": scale[1], "Seed": seed + 3}, data_type="FLOAT")
    rot = node(ng, "FunctionNodeEulerToRotation", {"Euler": out(rr, "Value")})
    ip = node(ng, "GeometryNodeInstanceOnPoints", {"Points": out(dp, "Points"), "Instance": out(ci),
                                                   "Pick Instance": True, "Instance Index": out(ri, "Value"),
                                                   "Rotation": out(rot), "Scale": out(rs, "Value")})
    ng.links.new(out(ip), go.inputs[0])
    ob = new_object(name, bpy.data.meshes.new(name + "_src"), parent)
    mod = ob.modifiers.new("Scatter", "NODES")
    mod.node_group = ng
    return ob


# =============================================================== terrain & water
def walk_paths():
    return [catmull(WALK_WEST, 0.8), catmull(WALK_EAST, 0.8)]


def build_terrain(coll):
    X = 1500 * np.sinh(3.2 * np.linspace(-1, 1, 561)) / np.sinh(3.2)
    Y = 1100 * np.sinh(3.2 * np.linspace(-1, 1, 421)) / np.sinh(3.2)
    XX, YY = np.meshgrid(X, Y, indexing="ij")
    H = terrain_h(XX, YY)
    ay = np.abs(YY)
    d = np.abs(YY - stream_y(XX))
    terr = terrace_mask(XX, YY)
    trail_clear = np.where(np.abs(XX) < 430, smoothstep(1.6, 3.2, np.abs(YY - trail_y(XX))), 1.0)
    guide_clear = smoothstep(5.0, 8.5, np.abs(YY - GUIDE_Y)) * smoothstep(5.0, 8.5, np.abs(YY + GUIDE_Y))
    foot = box_mask(XX, YY, -110, 110, -57, 57, 5)
    clump = smoothstep(-0.35, 0.15, NZ_FINE(XX * 0.45 + 70, YY * 0.45))
    stream_clear = np.where(np.abs(XX) < 280, smoothstep(10.0, 15.0, d), smoothstep(5.5, 8.5, d))  # fly-through corridor
    base = (1 - terr) * guide_clear * trail_clear * clump * stream_clear * (ay < 900)
    forest = base * (1 - foot) * (1 - smoothstep(130, 190, ay)) * (1 - smoothstep(600, 800, np.abs(XX)))
    conifer = base * (1 - foot) * np.maximum(smoothstep(110, 170, ay), smoothstep(550, 750, np.abs(XX)))

    # grove under the roof, kept clear of structures and of the fly-through corridor over the stream
    grove = np.zeros_like(H)
    sel = (np.abs(XX) < 112) & (ay < 21)
    xs, ys = XX[sel], YY[sel]
    clear = np.ones_like(xs)
    for path in walk_paths():
        clear *= smoothstep(6.0, 8.0, dist_polyline_xy(xs, ys, path))
    for cx, cy in VALLEY_COLS + [TOWER, LOOKOUT[:2]]:
        clear *= smoothstep(3.5, 5.0, np.hypot(xs - cx, ys - cy))
    for cx, cy in EMERGENT:
        clear *= smoothstep(6.0, 8.0, np.hypot(xs - cx, ys - cy))
    clear *= smoothstep(BRIDGE_X + 3, BRIDGE_X + 5, np.abs(xs))
    clear *= smoothstep(9.5, 11.5, d[sel]) * trail_clear[sel] * smoothstep(19.0, 17.0, np.abs(ys))
    grove[sel] = clear
    under = (1 - terr) * smoothstep(4.3, 6.0, d) * np.where(np.abs(XX) < 430, smoothstep(1.3, 2.2, np.abs(YY - trail_y(XX))), 1)
    under *= (1 - smoothstep(45, 70, ay)) * (1 - smoothstep(300, 420, np.abs(XX))) * guide_clear
    rocks = smoothstep(2.6, 3.4, d) * (1 - smoothstep(6.0, 8.5, d)) * (1 - terr) * (np.abs(XX) < 600)
    deck_zone = box_mask(XX, ay, -114, 114, 24.0, 64.0, 2.0)     # platforms and terraces: nothing grows through
    under *= 1 - deck_zone
    conifer *= (ay < 700) * (np.abs(XX) < 1300)
    wet = 1 - smoothstep(3.0, 10.0, d)
    bed = 1 - smoothstep(2.8, 4.2, d)
    meadow = smoothstep(4.0, 5.0, d) * (1 - smoothstep(150, 190, np.abs(XX))) * (1 - smoothstep(19, 23, ay)) * (1 - terr)
    meadow *= np.where(np.abs(XX) < 430, smoothstep(1.3, 1.8, np.abs(YY - trail_y(XX))), 1)

    V = np.stack([XX, YY, H], -1).reshape(-1, 3)
    mb = MeshBuilder()
    mb.add(V, grid_faces(len(X), len(Y)), mat=M["Terrain"], smooth=True,
           pattrs={k: a.ravel() for k, a in dict(wet=wet, bed=bed, forest=forest, conifer=conifer, grove=grove,
                                                  under=under, rocks=rocks, meadow=meadow).items()})
    terrain = mb.build("Terrain", coll)

    # stream water ribbon
    xs = np.arange(-1000, 1000.1, 1.0)
    lat = np.linspace(-7.5, 7.5, 4)
    V = np.stack([np.repeat(xs, len(lat)), (stream_y(xs)[:, None] + lat[None]).ravel(),
                  np.repeat(water_z(xs), len(lat))], -1)
    mb = MeshBuilder()
    mb.add(V, grid_faces(len(xs), len(lat)), mat=M["StreamWater"], smooth=True)
    mb.build("Stream", coll)

    # gravel trail
    xs = np.arange(-430, 430.1, 1.0)
    lat = np.linspace(-1.2, 1.2, 5)
    yy = trail_y(xs)[:, None] + lat[None]
    xx = np.repeat(xs[:, None], len(lat), 1)
    V = np.stack([xx, yy, terrain_h(xx, yy) + 0.07], -1).reshape(-1, 3)
    mb = MeshBuilder()
    mb.add(V, grid_faces(len(xs), len(lat)), mat=M["Gravel"], smooth=True)
    mb.build("Trail", coll)
    return terrain


# =============================================================== roof gridshell
def build_roof(coll):
    I, J = np.meshgrid(np.arange(NU + 1), np.arange(NV + 1), indexing="ij")
    U, Vv = I / NU, J / NV
    P, N = roof_frame(U, Vv)
    P, N = P.reshape(-1, 3), N.reshape(-1, 3)

    def idx(i, j):
        return i * (NV + 1) + j

    def ell(x, y):
        return [((x - cx) / a) ** 2 + ((y - cy) / b) ** 2 for cx, cy, a, b in OCULI]

    # members of the diagrid + boundary ring
    E, EB = [], []
    for i in range(NU + 1):
        for j in range(NV + 1):
            if (i + j) % 2:
                continue
            for dj in (1, -1):
                if i + 1 <= NU and 0 <= j + dj <= NV:
                    E.append((idx(i, j), idx(i + 1, j + dj)))
            if j in (0, NV) and i + 2 <= NU:
                EB.append((idx(i, j), idx(i + 2, j)))
            if i in (0, NU) and j + 2 <= NV:
                EB.append((idx(i, j), idx(i, j + 2)))
    E = np.array(E)
    A, B = P[E[:, 0]].copy(), P[E[:, 1]].copy()
    upv = norm(N[E[:, 0]] + N[E[:, 1]])
    keep = np.ones(len(E), bool)
    for k, (cx, cy, a, b) in enumerate(OCULI):
        fa = ((A[:, 0] - cx) / a) ** 2 + ((A[:, 1] - cy) / b) ** 2
        fb = ((B[:, 0] - cx) / a) ** 2 + ((B[:, 1] - cy) / b) ** 2
        keep &= ~((fa < 1) & (fb < 1))
        for src, dst, fs in ((A, B, fb), (B, A, fa)):
            m = (fs < 1) & (((src[:, 0] - cx) / a) ** 2 + ((src[:, 1] - cy) / b) ** 2 >= 1)
            if not m.any():
                continue
            d = dst[m] - src[m]
            o = src[m]
            qa = (d[:, 0] / a) ** 2 + (d[:, 1] / b) ** 2
            qb = 2 * ((o[:, 0] - cx) * d[:, 0] / a ** 2 + (o[:, 1] - cy) * d[:, 1] / b ** 2)
            qc = ((o[:, 0] - cx) / a) ** 2 + ((o[:, 1] - cy) / b) ** 2 - 1
            t = (-qb - np.sqrt((qb * qb - 4 * qa * qc).clip(0))) / (2 * qa)
            dst[m] = o + d * t[:, None]
    frame = MeshBuilder()
    frame.add(*beams(A[keep], B[keep], upv[keep], 0.3, 0.52), mat=M["Timber"])
    lift = upv[keep] * 0.33     # aluminium cover caps over the glazing joints, so the lattice reads from above
    frame.add(*beams(A[keep] + lift, B[keep] + lift, upv[keep], 0.1, 0.05), mat=M["Mullion"])
    EB = np.array(EB)
    frame.add(*beams(P[EB[:, 0]], P[EB[:, 1]], norm(N[EB[:, 0]] + N[EB[:, 1]]), 0.6, 0.95), mat=M["Timber"])

    # steel node connectors and (dusk) node lights
    lat_ids = np.array([idx(i, j) for i in range(NU + 1) for j in range(NV + 1) if (i + j) % 2 == 0])
    inside = np.zeros(len(lat_ids), bool)
    for f in ell(P[lat_ids, 0], P[lat_ids, 1]):
        inside |= f < 1
    lights = MeshBuilder()
    for k in lat_ids[~inside]:
        frame.add(*tube([P[k] - N[k] * 0.3, P[k] + N[k] * 0.28], 0.24, 10), mat=M["SteelWhite"], smooth=True)
    li = lat_ids[~inside & (np.abs(P[lat_ids, 1]) < 47)]
    li = li[(li // (NV + 1)) % 2 == 0]
    for k in li:
        lights.add(*tube([P[k] - N[k] * 0.36, P[k] - N[k] * 0.3], 0.13, 10), mat=M["NodeLight"])

    # oculus ring beams
    for cx, cy, a, b in OCULI:
        th = np.linspace(0, 2 * np.pi, 96, endpoint=False)
        ex, ey = cx + a * np.cos(th), cy + b * np.sin(th)
        ring = np.c_[ex, ey, roof_z_at(ex, ey)]
        sq = [(-0.35, -0.5), (0.35, -0.5), (0.35, 0.5), (-0.35, 0.5)]
        frame.add(*sweep(ring, sq, closed_path=True), mat=M["SteelWhite"])

    # glazing panels, PV panels, roof-garden trays
    faces = []
    for i in range(NU + 1):
        for j in range(NV + 1):
            if (i + j) % 2 == 0:
                continue
            if 1 <= i <= NU - 1 and 1 <= j <= NV - 1:
                faces.append([idx(i - 1, j), idx(i, j - 1), idx(i + 1, j), idx(i, j + 1)])
            elif j == 0 and 1 <= i <= NU - 1:
                faces.append([idx(i - 1, 0), idx(i + 1, 0), idx(i, 1), -1])
            elif j == NV and 1 <= i <= NU - 1:
                faces.append([idx(i - 1, NV), idx(i, NV - 1), idx(i + 1, NV), -1])
            elif i == 0 and 1 <= j <= NV - 1:
                faces.append([idx(0, j - 1), idx(1, j), idx(0, j + 1), -1])
            elif i == NU and 1 <= j <= NV - 1:
                faces.append([idx(NU, j - 1), idx(NU, j + 1), idx(NU - 1, j), -1])
    faces = np.array(faces)
    quad = faces[:, 3] >= 0
    fv = np.where(faces >= 0, faces, faces[:, :1])
    ctr = np.where(quad[:, None], P[fv].mean(1), P[fv[:, :3]].mean(1))
    cu, cv = roof_uv_of(ctr[:, 0], ctr[:, 1])
    wave = roof_wave(cu, cv)
    arch = np.sin(np.pi * cv).clip(0) ** 0.75
    hole = np.zeros(len(faces), bool)
    for cx, cy, a, b in OCULI:
        for c in range(4):
            vx, vy = P[fv[:, c], 0], P[fv[:, c], 1]
            hole |= ((vx - cx) / a) ** 2 + ((vy - cy) / b) ** 2 < 1.15
    garden = ~hole & (wave < -0.42) & (arch > 0.2) & (arch < 0.86) & (np.abs(ctr[:, 1]) > 22)
    pv = ~hole & ~garden & (wave > 0.3) & (arch > 0.45)
    glass = ~hole & ~garden & ~pv

    def panel_set(mask, offset):
        """Per face-size group: (vertices, local face indices) of panels offset along the normal."""
        sel = faces[mask]
        q = sel[:, 3] >= 0
        out_v, out_f = [], []
        for grp, k in ((sel[q], 4), (sel[~q][:, :3], 3)):
            if len(grp):
                out_v.append((P[grp] + N[grp] * offset).reshape(-1, 3))
                out_f.append(np.arange(len(grp) * k).reshape(-1, k))
        return out_v, out_f

    glaze = MeshBuilder()
    for mask, mat in ((glass, M["Glass"]), (pv, M["SolarGlass"])):
        for V, F in zip(*panel_set(mask, 0.29)):
            glaze.add(V, F, mat=mat)
    soil = MeshBuilder()
    for (Vb, Fb), (Vt, Ft) in zip(zip(*panel_set(garden, 0.27)), zip(*panel_set(garden, 0.66))):
        k = Fb.shape[1]
        frame.add(Vb, Fb[:, ::-1], mat=M["Corten"])
        soil.add(Vt, Ft, mat=M["Soil"])
        nb = len(Vb)
        V = np.concatenate([Vb, Vt])
        sides = []
        for c in range(k):
            c1 = (c + 1) % k
            sides.append(np.stack([Fb[:, c], Fb[:, c1], nb + Fb[:, c1], nb + Fb[:, c]], -1))
        frame.add(V, np.concatenate(sides), mat=M["Corten"])
    # planted collar around each oculus: covers the jagged glazing edge and frames the emergent tree
    for cx, cy, a, b in OCULI:
        th = np.linspace(0, 2 * np.pi, 120, endpoint=False)
        rr = np.linspace(1.0, 1.55, 6)
        X, Y = cx + a * rr[:, None] * np.cos(th), cy + b * rr[:, None] * np.sin(th)
        Zr = roof_z_at(X, Y)
        top = np.stack([X, Y, Zr + 0.66], -1)
        bot = np.stack([X, Y, Zr + 0.25], -1)
        F = grid_faces(len(rr), len(th), wrap_v=True)
        soil.add(top.reshape(-1, 3), F, mat=M["Soil"])
        frame.add(bot.reshape(-1, 3), F[:, ::-1], mat=M["Corten"])
        n = len(th)
        i = np.arange(n)
        for k in (0, len(rr) - 1):
            V = np.concatenate([bot[k], top[k] + [0, 0, 0.1]])
            frame.add(V, np.stack([i, (i + 1) % n, n + (i + 1) % n, n + i], -1), mat=M["Corten"])
    rf = frame.build("RoofFrame", coll)
    rg = glaze.build("RoofGlazing", coll)
    rl = lights.build("RoofNodeLights", coll)
    rs = soil.build("RoofGardenSoil", coll)
    me = rs.data
    me.attributes.new("garden", "FLOAT", "POINT").data.foreach_set("value", np.ones(len(me.vertices), np.float32))
    log(f"roof: {keep.sum()} members, {glass.sum()} glass, {pv.sum()} PV, {garden.sum()} garden, {hole.sum()} open")
    return rf, rg, rl, rs, ctr[garden], N[fv[garden, 0]]


# =============================================================== dendriform columns
def dendriform(mb, base, trunk_h, r1, h1, r2, az0, scale=1.0):
    bx, by, bz = base
    T = np.array([bx, by, bz + trunk_h])
    zs = np.linspace(bz - 0.6, T[2], 9)
    rt = np.interp(zs, [bz - 0.6, bz + 0.6, bz + trunk_h * 0.55, T[2]], [0.7, 0.5, 0.44, 0.56]) * scale
    mb.add(*tube(np.c_[np.full(9, bx), np.full(9, by), zs], rt, 18), mat=M["SteelWhite"], smooth=True)
    V, F = ico(2)
    mb.add(T + V * 0.6 * scale, F, mat=M["SteelWhite"], smooth=True)
    tips = []
    for k in range(4):
        a = az0 + k * np.pi / 2
        P1 = T + np.array([r1 * np.cos(a), r1 * np.sin(a), h1])
        L = np.linalg.norm(P1 - T)
        path = bezier(T, P1, T + np.array([0, 0, L * 0.45]), P1 - norm(P1 - T) * L * 0.3, 12)
        mb.add(*tube(path, np.linspace(0.44, 0.31, 12) * scale, 14), mat=M["SteelWhite"], smooth=True)
        mb.add(P1 + V * 0.34 * scale, F, mat=M["SteelWhite"], smooth=True)
        d1 = norm(P1 - T)
        for sgn in (-1, 1):
            a2 = a + sgn * 0.5
            tx, ty = bx + r2 * np.cos(a2), by + r2 * np.sin(a2)
            u, _v = roof_uv_of(tx, ty)
            half = float(roof_w(u)) / 2 - 1.8
            ty = float(np.clip(ty, -half, half))
            tip = np.array([tx, ty, float(roof_z_at(tx, ty)) - 0.45])
            L2 = np.linalg.norm(tip - P1)
            path = bezier(P1, tip, P1 + d1 * L2 * 0.35, tip - np.array([0, 0, L2 * 0.3]), 12)
            mb.add(*tube(path, np.linspace(0.29, 0.15, 12) * scale, 12), mat=M["SteelWhite"], smooth=True)
            mb.add(*tube([tip, tip + [0, 0, 0.3]], 0.32, 12), mat=M["SteelWhite"], smooth=True)
            tips.append(tip)
    return tips


def build_columns(coll):
    mb = MeshBuilder()
    for k, (x, y) in enumerate(VALLEY_COLS):
        z = float(terrain_h(x, y))
        dendriform(mb, (x, y, z), 13.5 - z, 5.2, 6.0, 11.0, np.pi / 4 + 0.2 * k)
        mb.add(*tube([(x, y, z - 0.3), (x, y, z + 0.25)], 1.3, 24), mat=M["Concrete"], smooth=True)
    for k, (x, y) in enumerate(SIDE_COLS):
        dendriform(mb, (x, y, GRADE), 4.2, 3.2, 3.0, 7.6, np.pi / 4, scale=0.85)
    return mb.build("DendriformColumns", coll)


# =============================================================== decks, bridge, walks
def build_decks(coll):
    mb = MeshBuilder()
    leds = MeshBuilder()
    prof = [(-1.6, 10.6), (1.6, 10.6), (2.7, 11.8), (2.7, 13.6), (2.3, 13.6), (2.3, 13.0),
            (-2.3, 13.0), (-2.3, 13.6), (-2.7, 13.6), (-2.7, 11.8)]
    for s in (1, -1):
        yc = s * GUIDE_Y
        mb.add(*sweep([(-1100, yc, 0), (1100, yc, 0)], prof), mat=M["ConcreteDark"])
        # piers
        for x in np.arange(-1080, 1081, 30.0):
            g = float(terrain_h(x, yc))
            if g > 9.6:
                continue
            zs = np.array([g - 1.0, 10.3])
            mb.add(*tube(np.c_[[x, x], [yc, yc], zs], [1.05, 0.8], 20), mat=M["Concrete"], smooth=True)
            mb.add(*boxes([(x, yc, 10.35)], (2.2, 3.4, 0.5)), mat=M["Concrete"])
        # platform slab on the outer side
        py = [25.85, 26.8, 26.8, 38.0, 38.0, 25.85]
        pz = [13.62, 13.62, 12.4, 12.4, GRADE, GRADE]
        pp = [(s * a, b) for a, b in zip(py, pz)]
        if s < 0:
            pp = pp[::-1]
        mb.add(*sweep([(-102, 0, 0), (102, 0, 0)], pp), mat=M["Concrete"])
        mb.add(*boxes([(0, s * 26.3, GRADE + 0.01)], (204, 0.6, 0.03)), mat=M["Tactile"])
        mb.add(*boxes([(0, s * 32.3, GRADE + 0.006)], (204, 11.4, 0.012)), mat=M["Terrazzo"])
        leds.add(*boxes([(0, s * 25.88, GRADE - 0.12)], (204, 0.05, 0.08)), mat=M["LED"])
        for x in np.arange(-96, 97, 16.0):
            if abs(x) < 20:
                continue
            mb.add(*boxes([(x, s * 34.2, GRADE + 0.45)], (3.2, 0.55, 0.08)), mat=M["Timber"])
            mb.add(*boxes([(x - 1.3, s * 34.2, GRADE + 0.2), (x + 1.3, s * 34.2, GRADE + 0.2)], (0.1, 0.5, 0.42)),
                   mat=M["SteelDark"])
        for x in np.arange(-88, 89, 40.0):
            mb.add(*boxes([(x, s * 30.6, GRADE + 1.25)], (0.5, 0.25, 2.5)), mat=M["SteelDark"])
            mb.add(*boxes([(x, s * 30.6 - s * 0.13, GRADE + 1.55)], (0.42, 0.02, 1.3)), mat=M["Display"])
        for x in np.arange(-100, 101, 20.0):
            g = float(terrain_h(x, s * 31.0))
            if g < 12.0:
                mb.add(*tube([(x, s * 31.0, g - 1), (x, s * 31.0, 12.4)], 0.55, 16), mat=M["Concrete"], smooth=True)
        # valley-side planter and LED cove
        mb.add(*boxes([(0, s * (GUIDE_Y - 3.05), 13.9)], (208, 0.7, 0.6)), mat=M["Corten"])
        mb.add(*boxes([(0, s * (GUIDE_Y - 3.05), 14.17)], (207.6, 0.6, 0.08)), mat=M["Soil"])
        leds.add(*boxes([(0, s * (GUIDE_Y - 3.42), 13.58)], (208, 0.06, 0.06)), mat=M["LED"])
    # plaza and south terrace podiums
    mb.add(*boxes([(0, 78, GRADE - 0.6), (0, -48.5, GRADE - 0.6)], [(208, 80, 1.2), (208, 21, 1.2)]), mat=M["Paving"])
    return mb.build("Decks", coll), leds


def build_bridge_and_walks(coll, leds):
    mb = MeshBuilder()
    glass = MeshBuilder()
    # sky bridge over both guideways
    L = 2 * 38.0
    mb.add(*boxes([(0, 0, BRIDGE_Z - 0.45)], (2 * BRIDGE_X, L, 0.9)), mat=M["SteelWhite"])
    mb.add(*boxes([(0, 0, BRIDGE_Z + 0.01)], (2 * BRIDGE_X - 0.3, L, 0.03)), mat=M["Timber"])
    for sx in (-1, 1):
        x = sx * (BRIDGE_X - 0.05)
        for y0, y1 in ((-38, 9.5), (12.5, 38)):
            glass.add(*boxes([(x, (y0 + y1) / 2, BRIDGE_Z + 1.1)], (0.03, y1 - y0, 2.2)), mat=M["Glass"])
            mb.add(*tube([(x, y0, BRIDGE_Z + 2.25), (x, y1, BRIDGE_Z + 2.25)], 0.05, 8), mat=M["SteelWhite"], smooth=True)
        leds.add(*boxes([(x - sx * 0.05, 0, BRIDGE_Z - 0.92)], (0.06, L, 0.05)), mat=M["LED"])
        for y in np.arange(-36, 37, 6.0):
            top = float(roof_z_at(x, y)) - 0.3
            mb.add(*tube([(x, y, BRIDGE_Z + 2.25), (x, y, top)], 0.035, 6), mat=M["SteelWhite"])
    # escalators down to both platforms
    for sy in (1, -1):
        for sx in (1, -1):
            y0, y1 = sy * 30.0, sy * 34.0
            xa, xb = sx * BRIDGE_X, sx * 16.5
            yy = sorted([y0, y1])
            V = np.array([[xa, yy[0], BRIDGE_Z], [xb, yy[0], GRADE], [xb, yy[1], GRADE], [xa, yy[1], BRIDGE_Z],
                          [xa, yy[0], BRIDGE_Z - 0.8], [xb, yy[0], GRADE - 0.4], [xb, yy[1], GRADE - 0.4],
                          [xa, yy[1], BRIDGE_Z - 0.8]])
            mb.add(V, BOX_F[:, ::-1] if sx > 0 else BOX_F, mat=M["SteelDark"])
            for yb in yy:
                gv = np.array([[xa, yb, BRIDGE_Z], [xb, yb, GRADE], [xb, yb, GRADE + 1.0], [xa, yb, BRIDGE_Z + 1.0]])
                glass.add(gv, [[0, 1, 2, 3]], mat=M["Glass"])
                mb.add(*tube([gv[3], gv[2]], 0.045, 8), mat=M["Rubber"], smooth=True)

    # canopy walks, hung from the roof
    deck = [(-1.3, -0.22), (1.3, -0.22), (1.3, 0.0), (-1.3, 0.0)]
    for path in walk_paths():
        mb.add(*sweep(path, deck), mat=M["Timber"])
        for side in (-1, 1):
            stringer = [(side * 1.25, -0.5), (side * 1.45, -0.5), (side * 1.45, 0.08), (side * 1.25, 0.08)]
            if side < 0:
                stringer = stringer[::-1]
            mb.add(*sweep(path, stringer), mat=M["SteelWhite"])
            T = norm(np.gradient(path, axis=0))
            S = norm(np.cross([0, 0, 1], T))
            rail = path + S * side * 1.36 + [0, 0, 1.1]
            mb.add(*tube(rail, 0.04, 8), mat=M["SteelWhite"], smooth=True)
            leds.add(*tube(rail - [0, 0, 0.06], 0.018, 6), mat=M["LED"])
            net = [(side * 1.33, 0.05), (side * 1.37, 0.05), (side * 1.37, 1.05), (side * 1.33, 1.05)]
            if side < 0:
                net = net[::-1]
            glass.add(*sweep(path, net), mat=M["SteelMesh"])
            for k in range(0, len(path), 3):
                mb.add(*boxes([path[k] + S[k] * side * 1.36 + [0, 0, 0.55]], (0.07, 0.07, 1.1)), mat=M["SteelWhite"])
            for k in range(0, len(path), 8):
                p = rail[k]
                top = float(roof_z_at(p[0], p[1])) - 0.3
                mb.add(*tube([p, (p[0], p[1], top)], 0.03, 6), mat=M["SteelWhite"])
    # spiral stair tower at the west end
    tx, ty = TOWER
    g = float(terrain_h(tx, ty))
    top = WALK_WEST[-1][2]
    mb.add(*tube([(tx, ty, g - 0.5), (tx, ty, top + 2.5)], 0.4, 20), mat=M["SteelWhite"], smooth=True)
    mb.add(*boxes([(tx, ty + 1.2, top - 0.15)], (5.5, 6.0, 0.3)), mat=M["Timber"])
    n = int((top - g) / 0.19)
    ang = np.arange(n) * np.radians(17.0)
    rad = 1.4
    C = np.c_[tx + rad * np.cos(ang), ty + rad * np.sin(ang), g + 0.19 * np.arange(1, n + 1) - 0.05]
    mb.add(*boxes(C, (2.0, 0.42, 0.06), rotz=ang), mat=M["Timber"])
    hel = np.c_[tx + 2.45 * np.cos(ang), ty + 2.45 * np.sin(ang), C[:, 2] + 1.0]
    mb.add(*tube(hel, 0.04, 6), mat=M["SteelWhite"], smooth=True)
    for k in range(0, n, 4):
        mb.add(*tube([hel[k] - [0, 0, 1.0], hel[k]], 0.025, 5), mat=M["SteelWhite"])
    # cantilevered lookout at the east mouth
    lx, ly, lz = LOOKOUT
    mb.add(*lathe([(0.0, lz - 0.35), (5.0, lz - 0.35), (5.2, lz), (0.0, lz)], 48, center=(lx, ly, 0), caps=False),
           mat=M["Timber"])
    th = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    ring = np.c_[lx + 5.0 * np.cos(th), ly + 5.0 * np.sin(th), np.full(64, lz + 1.1)]
    mb.add(*sweep(ring, [(-0.04, -0.04), (0.04, -0.04), (0.04, 0.04), (-0.04, 0.04)], closed_path=True), mat=M["SteelWhite"])
    glass.add(*sweep(ring - [0, 0, 1.1], [(0.0, 0.05), (0.0, 1.05), (0.02, 1.05), (0.02, 0.05)], closed_path=True),
              mat=M["Glass"])
    g = float(terrain_h(lx, ly))
    mb.add(*tube([(lx, ly, g - 0.5), (lx, ly, lz - 0.3)], [0.65, 0.45], 18), mat=M["SteelWhite"], smooth=True)
    mb.build("BridgeAndWalks", coll)
    glass.build("Balustrades", coll)


# =============================================================== vines
def build_vines(coll):
    rng = np.random.default_rng(42)
    mb = MeshBuilder()
    anchors = []
    for s in (1, -1):        # guideway planters, hanging over the valley
        xs = np.arange(-103, 103, 0.32)
        xs = xs[np.sin(xs / 5.3 + s) + 0.35 * rng.standard_normal(len(xs)) > -0.55]
        y = s * (GUIDE_Y - 3.45)
        for x in xs:
            g = float(terrain_h(x, y))
            anchors.append((x + rng.normal(0, 0.05), y + s * rng.normal(0, 0.05), 14.05,
                            min(rng.uniform(2.5, 6.2), 14.0 - g - 0.6)))
    for v, s in ((0.0, -1), (1.0, 1)):   # roof eaves above plaza / terrace
        us = np.linspace(0.02, 0.98, 520)
        P = roof_uv_xyz(us, np.full_like(us, v))
        for (x, y, z) in P:
            if abs(x) < 12 or abs(abs(x) - 60) < 6 or rng.random() < 0.18:
                continue
            anchors.append((x, y - s * 0.4, z - 0.5, min(rng.uniform(2.5, 6.5), z - GRADE - 3.2)))
    stems_v, stems_f, leaves = [], [], []
    base = 0
    for (x, y, z, L) in anchors:
        if L < 0.8:
            continue
        n = max(3, int(L / 0.45))
        t = np.linspace(0, 1, n)
        ph = rng.uniform(0, 2 * np.pi, 2)
        P = np.c_[x + 0.12 * np.sin(t * 5 + ph[0]) * t, y + 0.12 * np.sin(t * 4 + ph[1]) * t, z - L * t]
        V, F = tube(P, np.linspace(0.016, 0.006, n), 3, caps=False)
        stems_v.append(V)
        stems_f.append(F[0] + base)
        base += len(V)
        k = int(L / 0.09)
        tt = rng.random(k)
        lp = np.c_[np.interp(tt, t, P[:, 0]), np.interp(tt, t, P[:, 1]), np.interp(tt, t, P[:, 2])]
        leaves.append(lp + rng.normal(0, 0.06, (k, 3)))
    mb.add(np.concatenate(stems_v), np.concatenate(stems_f), mat=M["Bark"])
    lp = np.concatenate(leaves)
    nrm = norm(rng.normal(0, 1, lp.shape) + [0, 0, 0.3])
    V, F, lv = leaf_cards(rng, lp, nrm, 0.13, aspect=0.8, droop=0.1, spread=0.2)
    mb.add(V, F, mat=M["VineLeaf"], lv=lv)
    log(f"vines: {len(anchors)} strands, {len(lp)} leaves")
    return mb.build("VineCurtains", coll)


# =============================================================== plaza
def build_plaza(coll, P, leds):
    mb = MeshBuilder()
    glass = MeshBuilder()
    cx, cy, hl, r = POOL
    ring_o = stadium(cx, cy, hl + 0.7, r + 0.7)
    ring_i = stadium(cx, cy, hl, r)
    n = len(ring_i)
    V = np.concatenate([np.c_[ring_o, np.full(n, GRADE + 0.18)], np.c_[ring_i, np.full(n, GRADE + 0.18)],
                        np.c_[ring_o, np.full(n, GRADE - 0.01)], np.c_[ring_i, np.full(n, GRADE + 0.02)]])
    i = np.arange(n)
    j = (i + 1) % n
    F = np.concatenate([np.stack([i, j, n + j, n + i], -1),
                        np.stack([2 * n + i, 2 * n + j, j, i], -1),
                        np.stack([n + i, n + j, 3 * n + j, 3 * n + i], -1)])
    mb.add(V, F, mat=M["StoneLight"])
    water = np.c_[ring_i, np.full(n, GRADE + 0.12)]
    mb.add(water, [np.arange(n)], mat=M["PoolWater"])
    mb.add(np.c_[ring_i, np.full(n, GRADE + 0.02)], [np.arange(n)[::-1]], mat=M["Granite"])
    # sign monolith
    mb.add(*boxes([(0, 56.0, GRADE + 1.1)], (26, 1.2, 2.2)), mat=M["Granite"])
    # lawns either side of the paved promenade, with trees and grass
    rng = np.random.default_rng(3)
    protos = [P["GroveA"], P["GroveB"], P["GroveC"], P["TreeB"], P["TreeD"]]
    lawn = MeshBuilder()
    k = 0
    for x0, x1, y0, y1 in LAWNS:
        nx, ny = int((x1 - x0) / 2) + 1, int((y1 - y0) / 2) + 1
        gx, gy = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny), indexing="ij")
        lawn.add(np.stack([gx, gy, np.full_like(gx, GRADE + 0.06)], -1).reshape(-1, 3), grid_faces(nx, ny),
                 mat=M["Lawn"], pattrs={"lawn": 1.0})
        mb.add(*boxes([((x0 + x1) / 2, (y0 + y1) / 2, GRADE - 0.05)], (x1 - x0 + 0.5, y1 - y0 + 0.5, 0.2)), mat=M["StoneLight"])
        for tx in np.arange(x0 + 7, x1 - 4, 13.0):
            for ty in np.arange(y0 + 6, y1 - 3, 12.0):
                x, y = tx + rng.uniform(-3, 3), ty + rng.uniform(-3, 3)
                new_object(f"PlazaTree{k:02d}", protos[rng.integers(len(protos))].data, coll, (x, y, GRADE + 0.05),
                           (0, 0, rng.uniform(0, 6.3)), rng.uniform(0.8, 1.05))
                k += 1
    lawn_ob = lawn.build("Lawns", coll)
    for x in (-46, -24, 24, 46):        # planters on the promenade near the entrance
        y = 47.0
        mb.add(*lathe([(0, GRADE - 0.1), (2.3, GRADE - 0.1), (2.3, GRADE + 0.55), (2.0, GRADE + 0.55), (0, GRADE + 0.55)],
                      40, center=(x, y, 0), caps=False), mat=M["Corten"])
        mb.add(*lathe([(0, GRADE + 0.5), (2.02, GRADE + 0.5)], 40, center=(x, y, 0), caps=False), mat=M["Soil"])
        new_object(f"PlazaTree{k:02d}", P["GroveA"].data, coll, (x, y, GRADE + 0.5), (0, 0, rng.uniform(0, 6.3)), 0.8)
        k += 1
    # lamp posts
    lamps = [(x, cy + sy * (r + 3.5)) for x in np.arange(-36, 37, 12.0) for sy in (-1, 1)]
    lamps += [(x, y) for x in (-98, -75, 75, 98) for y in (66, 92, 114)]
    for (x, y) in lamps:
        mb.add(*tube([(x, y, GRADE), (x, y, GRADE + 4.6)], [0.09, 0.06], 10), mat=M["SteelDark"], smooth=True)
        mb.add(*boxes([(x, y, GRADE + 4.7)], (0.7, 0.18, 0.12)), mat=M["SteelDark"])
        leds.add(*boxes([(x, y, GRADE + 4.635)], (0.62, 0.14, 0.01)), mat=M["LampHead"])
    mb.build("Plaza", coll)
    glass.build("PlazaGlass", coll)

    # station name: 翠谷站 / EMERALD VALLEY STATION
    font = None
    for path in (os.path.join(bpy.utils.system_resource("DATAFILES", path="fonts"), "Noto Sans CJK Regular.woff2"),
                 "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"):
        if os.path.exists(path):
            try:
                font = bpy.data.fonts.load(path)
                break
            except RuntimeError:
                pass
    for body, size, z, mat in (("翠谷站" if font else "EMERALD VALLEY", 1.25, GRADE + 0.82, M["SignGlow"]),
                               ("EMERALD VALLEY STATION", 0.36, GRADE + 0.28, M["SignGlow"])):
        cu = bpy.data.curves.new("Sign", "FONT")
        cu.body = body
        if font:
            cu.font = font
        cu.size = size
        cu.extrude = 0.02
        cu.align_x, cu.align_y = "CENTER", "BOTTOM"
        cu.materials.append(mat)
        new_object("SignText", cu, coll, (0, 56.62, z), (math.pi / 2, 0, math.pi))
    return lawn_ob


# =============================================================== trains
NOSE = 15.0


def train_section(t, n=40):
    a = 1.7 * max(0.06, (1 - (1 - t) ** 2.4) ** 0.5)
    zt = 0.95 + (4.05 - 0.95) * (1 - (1 - t) ** 2.0) ** 0.55
    zb = 0.25 + 0.55 * (1 - t) ** 3
    zc, b = (zt + zb) / 2, (zt - zb) / 2
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = np.cos(th), np.sin(th)
    ex = np.where(s > 0, 2 / 2.6, 2 / 5.0)
    return np.c_[a * np.sign(c) * np.abs(c) ** ex, zc + b * np.sign(s) * np.abs(s) ** ex], a, zb, zt


def build_train(name, coll, cars=6, car_len=26.0, gap=0.9):
    mb = MeshBuilder()
    nring = 40
    dense = NOSE * np.linspace(0, 1, 30) ** 1.7
    mats = [M["TrainBody"], M["TrainStripe"], M["TrainSkirt"], M["TrainWindow"], M["Headlight"]]
    for c in range(cars):
        x0 = -c * (car_len + gap)
        lead, tail = c == 0, c == cars - 1
        s = np.arange(0, car_len + 1e-6, 0.5)
        t = np.ones_like(s)
        if lead:
            s = np.union1d(s, dense)
        if tail:
            s = np.union1d(s, car_len - dense)
        t = np.ones_like(s)
        if lead:
            t = np.minimum(t, s / NOSE)
        if tail:
            t = np.minimum(t, (car_len - s) / NOSE)
        t = t.clip(0, 1)
        rings, info = [], []
        for si, ti in zip(s, t):
            yz, a, zb, zt = train_section(ti, nring)
            rings.append(np.c_[np.full(nring, x0 - si), yz[:, 0], yz[:, 1]])
            info.append((si, ti, a, zb, zt))
        V = np.concatenate(rings)
        F = grid_faces(len(rings), nring, wrap_v=True)
        ctr = V[F].mean(1)
        ring_i = F[:, 0] // nring
        si = np.array([info[k][0] for k in ring_i])
        ti = np.array([info[k][1] for k in ring_i])
        a = np.array([info[k][2] for k in ring_i])
        side = np.abs(ctr[:, 1]) > 0.55 * a
        z = ctr[:, 2]
        mat = np.zeros(len(F), int)
        mat[(z > 1.25) & (z < 1.55) & side & (ti > 0.55)] = 1
        mat[z < 0.75] = 2
        bay = ((si - 1.0) % 2.3) < 1.7
        win = side & (z > 2.2) & (z < 3.05) & bay & (ti >= 1.0) & (si > 2) & (si < car_len - 2)
        cab = (ti > 0.3) & (ti < 0.8) & (z > 0.5 * (np.array([info[k][3] + info[k][4] for k in ring_i])) + 0.35)
        mat[win | cab] = 3
        head = (ti > 0.06) & (ti < 0.2) & side & (z < 1.3) & (z > 0.6)
        mat[head] = 4
        mb.add(V, F, mat=mats, mat_index=mat, smooth=True)
        mb.add(rings[0], [np.arange(nring)], mat=M["TrainBody"] if lead else M["TrainSkirt"])
        mb.add(rings[-1], [np.arange(nring)[::-1]], mat=M["TrainBody"] if tail else M["TrainSkirt"])
        if not tail:
            mb.add(*boxes([(x0 - car_len - gap / 2, 0, 2.2)], (gap + 0.4, 2.9, 3.2)), mat=M["Rubber"])
    return mb.build(name, coll)


def fcurves_of(ob):
    act = ob.animation_data.action
    if hasattr(act, "fcurves") and not getattr(act, "is_action_layered", False):
        return list(act.fcurves)
    slot = ob.animation_data.action_slot
    for layer in act.layers:
        for strip in layer.strips:
            cb = strip.channelbag(slot)
            if cb:
                return list(cb.fcurves)
    return []


def build_trains(coll):
    a = build_train("TrainA", coll)
    a.location = (-500.0, GUIDE_Y, TRAIN_Z)
    a.keyframe_insert("location", index=0, frame=1)
    a.location.x = -500.0 + 5.0 * (FRAMES - 1)
    a.keyframe_insert("location", index=0, frame=FRAMES)
    for fc in fcurves_of(a):
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"
    b = new_object("TrainB", a.data, coll)
    b.rotation_euler = (0, 0, math.pi)
    b.location = (260.0, -GUIDE_Y, TRAIN_Z)
    b.keyframe_insert("location", index=0, frame=1)
    b.location.x = -70.0
    b.keyframe_insert("location", index=0, frame=200)
    for fc in fcurves_of(b):
        kp = fc.keyframe_points[0]
        kp.interpolation = "QUAD"
        kp.easing = "EASE_OUT"


# =============================================================== people
SKIN = [(0.87, 0.68, 0.55), (0.6, 0.42, 0.3), (0.95, 0.8, 0.68), (0.38, 0.25, 0.18)]
CLOTH = [(0.85, 0.85, 0.82), (0.1, 0.12, 0.18), (0.55, 0.12, 0.1), (0.15, 0.3, 0.5), (0.75, 0.6, 0.35),
         (0.2, 0.35, 0.22), (0.9, 0.75, 0.2), (0.3, 0.3, 0.32), (0.65, 0.4, 0.55)]


HAIR = [(0.05, 0.035, 0.025), (0.02, 0.02, 0.02), (0.35, 0.22, 0.1), (0.6, 0.5, 0.3), (0.5, 0.5, 0.5)]


def build_people(coll, spots, seed=9):
    rng = np.random.default_rng(seed)
    mb = MeshBuilder()
    Vi, Fi = ico(2)
    for (x, y, z) in spots:
        h = rng.uniform(0.9, 1.08)
        rot = rng.uniform(0, 2 * np.pi)
        shirt, pants = CLOTH[rng.integers(len(CLOTH))], CLOTH[rng.integers(len(CLOTH))]
        skin, hair = SKIN[rng.integers(len(SKIN))], HAIR[rng.integers(len(HAIR))]
        st = rng.uniform(-0.14, 0.14)        # stride: legs and arms swing in opposition
        parts = [  # centre, size, colour, top taper
            ((-0.09, st + 0.03, 0.04), (0.11, 0.26, 0.08), (0.05, 0.05, 0.05), 1.0),
            ((0.09, -st + 0.03, 0.04), (0.11, 0.26, 0.08), (0.05, 0.05, 0.05), 1.0),
            ((-0.09, st * 0.5, 0.47), (0.12, 0.14, 0.8), pants, 1.2),
            ((0.09, -st * 0.5, 0.47), (0.12, 0.14, 0.8), pants, 1.2),
            ((0.0, 0.0, 0.93), (0.33, 0.19, 0.2), pants, 1.0),
            ((0.0, 0.0, 1.25), (0.3, 0.19, 0.5), shirt, 1.3),
            ((-0.235, -st * 0.6, 1.14), (0.08, 0.09, 0.58), shirt, 1.1),
            ((0.235, st * 0.6, 1.14), (0.08, 0.09, 0.58), shirt, 1.1),
            ((-0.24, -st * 0.9, 0.81), (0.065, 0.07, 0.1), skin, 1.0),
            ((0.24, st * 0.9, 0.81), (0.065, 0.07, 0.1), skin, 1.0),
            ((0.0, 0.0, 1.53), (0.09, 0.09, 0.08), skin, 1.0),
        ]
        C = np.array([p[0] for p in parts]) * h
        S = np.array([p[1] for p in parts]) * h
        V, F = boxes(C, S)
        V = V.reshape(len(parts), 8, 3)
        taper = np.array([p[3] for p in parts])[:, None]
        V[:, 4:, :2] = C[:, None, :2] + (V[:, 4:, :2] - C[:, None, :2]) * taper[:, None]
        head = Vi * np.array([0.1, 0.11, 0.12]) * h + np.array([0, 0.01, 1.66]) * h
        hair_v = Vi * np.array([0.108, 0.118, 0.11]) * h + np.array([0, -0.012, 1.69]) * h
        allv = np.concatenate([V.reshape(-1, 3), head, hair_v])
        cr, sr = np.cos(rot), np.sin(rot)
        allv = np.c_[allv[:, 0] * cr - allv[:, 1] * sr + x, allv[:, 0] * sr + allv[:, 1] * cr + y, allv[:, 2] + z]
        nb = len(parts) * 8
        cols = np.repeat(np.array([p[2] for p in parts]), 6, 0)
        mb.add(allv[:nb], F, mat=M["People"], pcol=cols)
        mb.add(allv[nb:nb + len(Vi)], Fi, mat=M["People"], smooth=True, pcol=skin)
        mb.add(allv[nb + len(Vi):], Fi, mat=M["People"], smooth=True, pcol=hair)
    return mb.build("People", coll)


def people_spots():
    rng = np.random.default_rng(5)
    spots = []
    cx, cy, hl, r = POOL
    while len(spots) < 95:
        x, y = rng.uniform(-56, 56) if len(spots) < 75 else rng.uniform(-100, 100), rng.uniform(42, 116)
        if in_stadium(x, y, cx, cy, hl, r, 1.5) or abs(y - 56) < 1.5 and abs(x) < 14:
            continue
        spots.append((x, y, GRADE))
    for s in (1, -1):
        for _ in range(34):
            spots.append((rng.uniform(-98, 98), s * rng.uniform(27.5, 37.0), GRADE))
    for _ in range(12):
        spots.append((rng.uniform(-5, 5), rng.uniform(-36, 36), BRIDGE_Z + 0.03))
    for path in walk_paths():
        for k in rng.choice(len(path), 8, replace=False):
            p = path[k]
            spots.append((p[0] + rng.uniform(-0.7, 0.7), p[1] + rng.uniform(-0.7, 0.7), p[2] + 0.02))
    for x in rng.uniform(-200, 200, 10):
        spots.append((x, float(trail_y(x)), float(terrain_h(x, trail_y(x))) + 0.07))
    for _ in range(10):
        a, rr = rng.uniform(0, 2 * np.pi), rng.uniform(0, 4.2)
        spots.append((LOOKOUT[0] + rr * np.cos(a), LOOKOUT[1] + rr * np.sin(a), LOOKOUT[2] + 0.01))
    cams = np.array([spec[0] for spec in camera_specs().values()])
    return [p for p in spots if np.min(np.hypot(cams[:, 0] - p[0], cams[:, 1] - p[1]) + np.abs(cams[:, 2] - p[2])) > 9.0]


# =============================================================== cameras, world, render
def look_at(ob, target):
    d = Vector(target) - ob.location
    ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def add_camera(name, loc, target, lens, frame, coll, dof=None):
    cam = bpy.data.cameras.new(name)
    cam.lens, cam.clip_start, cam.clip_end = lens, 0.1, 6000.0
    cam.sensor_width = 36.0
    if dof:
        cam.dof.use_dof = True
        cam.dof.focus_distance, cam.dof.aperture_fstop = dof
    ob = new_object(name, cam, coll, loc)
    look_at(ob, target)
    ob["still_frame"] = frame
    return ob


def fly_path():
    def on_stream(x, z):
        return (x, float(stream_y(x)), z)

    pts = [on_stream(300, 60), on_stream(225, 42), on_stream(170, 27), on_stream(138, 12), on_stream(118, 6.0)]
    pts += [on_stream(x, 4.6) for x in (95, 65, 35, 5, -25, -55, -85)]
    pts += [on_stream(-116, 7.0), on_stream(-132, 17), on_stream(-160, 34), on_stream(-200, 50), on_stream(-240, 66)]
    return catmull(pts, 0.5)


def camera_specs():
    """Fixed still cameras: name -> (location, target, lens, still frame, dof)."""
    xv = 93.0
    sv = float(stream_y(xv)) - 3.6
    xt = -84.0
    return {
        "CAM_Aerial": ((240, 222, 150), (-6, 2, 10), 30, 100, None),
        "CAM_Plaza": ((-7, 93.0, GRADE + 1.7), (3, 40, 17.0), 20, 100, None),
        "CAM_Valley": ((xv, sv, float(terrain_h(xv, sv)) + 1.9), (20, float(stream_y(20)) + 2, 12), 17, 120, None),
        "CAM_Platform": ((-84, -28.4, GRADE + 1.62), (-40, -25.2, 16.0), 24, 230, (15.5, 5.6)),
        "CAM_TrailUp": ((xt, float(trail_y(xt)), float(terrain_h(xt, trail_y(xt))) + 1.6), (-103, 11, 13.5), 17, 140, None),
    }


def build_cameras(coll, garden_ctr, garden_n):
    for name, (loc, tgt, lens, frame, dof) in camera_specs().items():
        add_camera(name, loc, tgt, lens, frame, coll, dof)
    k = int(np.argmin(np.hypot(garden_ctr[:, 0] + 36, garden_ctr[:, 1] + 38)))
    p = garden_ctr[k] + np.array([0, 0, 9.0])
    tgt = (45.0, -14.0, float(roof_z_at(45.0, -14.0)) - 2.0)
    add_camera("CAM_Roof", tuple(p), tgt, 20, 100, coll)

    cam = add_camera("CAM_Fly", (0, 0, 0), (1, 0, 0), 22, 1, coll)
    path = fly_path()
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    inside = (np.abs(path[:-1, 0]) < 125) & (path[:-1, 2] < 12)
    w = np.where(inside, 1.0 / 0.55, 1.0)
    s = np.concatenate([[0], np.cumsum(seg * w)])
    s /= s[-1]
    tf = np.linspace(0, 1, FRAMES)
    ease = tf * tf * (3 - 2 * tf) * 0.25 + tf * 0.75
    def yaw_pitch(v):
        v = v / np.linalg.norm(v)
        return math.atan2(v[1], v[0]), math.asin(float(np.clip(v[2], -1, 1)))

    prev, turn = None, None
    for f, e in enumerate(ease, start=1):
        p = np.array([np.interp(e, s, path[:, i]) for i in range(3)])
        q = np.array([np.interp(min(e + 0.035, 1.0), s, path[:, i]) for i in range(3)])
        if e > 0.985:
            q = p + (p - np.array([np.interp(e - 0.03, s, path[:, i]) for i in range(3)]))
        # look ahead along the path, then swing round (heading/pitch blend) to look back at the station
        look = 0.0 if f < FRAMES - 75 else float(smoothstep(FRAMES - 75, FRAMES - 15, f))
        ya, pa = yaw_pitch(q - p)
        yb, pb = yaw_pitch(np.array([0.0, 0.0, 22.0]) - p)
        dyaw = (yb - ya + np.pi) % (2 * np.pi) - np.pi
        if turn is not None and abs(dyaw - turn) > np.pi:
            dyaw += 2 * np.pi * np.sign(turn - dyaw)
        turn = dyaw
        yaw, pitch = ya + dyaw * look, pa + (pb - pa) * look
        d = Vector((math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)))
        cam.location = p
        quat = d.to_track_quat("-Z", "Y")
        eul = quat.to_euler("XYZ", prev) if prev is not None else quat.to_euler("XYZ")
        cam.rotation_euler = eul
        prev = eul
        cam.keyframe_insert("location", frame=f)
        cam.keyframe_insert("rotation_euler", frame=f)
    for fc in fcurves_of(cam):
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"


def build_world_and_render(scene, coll):
    world = bpy.data.worlds.new("Sky")
    scene.world = world
    nt = world.node_tree
    sky = nt.nodes.new("ShaderNodeTexSky")
    sky.name = "Sky"
    sky.sky_type = "MULTIPLE_SCATTERING"
    sky.sun_disc = False
    # procedural cloud deck: noise projected onto a plane overhead, faded towards the horizon
    gen = out(node(nt, "ShaderNodeTexCoord"), "Generated")
    d = out(node(nt, "ShaderNodeVectorMath", {0: gen}, operation="NORMALIZE"))
    dz = out(node(nt, "ShaderNodeSeparateXYZ", {0: d}), "Z")
    proj = out(node(nt, "ShaderNodeVectorMath", {0: d, 3: math_node(nt, "DIVIDE", 1.0, math_node(nt, "ADD", dz, 0.12))},
                    operation="SCALE"))
    stretch = out(node(nt, "ShaderNodeMapping", {"Vector": proj, "Scale": (1.0, 2.2, 1.0), "Location": (3.1, 0.7, 0.0)}))
    cloud = node(nt, "ShaderNodeTexNoise", {"Vector": stretch, "Scale": 1.6, "Detail": 9.0, "Roughness": 0.62,
                                            "Distortion": 0.35}, noise_dimensions="3D")
    cover = node(nt, "ShaderNodeValue", name="CloudCover")
    cover.outputs[0].default_value = 0.56
    mask = out(node(nt, "ShaderNodeMapRange", {"Value": out(cloud, "Fac"), "From Min": out(cover),
                                               "From Max": math_node(nt, "ADD", out(cover), 0.2)},
                    interpolation_type="SMOOTHSTEP"), "Result")
    mask = math_node(nt, "MULTIPLY", mask, smooth_fac(nt, dz, 0.02, 0.18))
    # clouds take their brightness from the sky model itself, tinted per mode (CloudColor)
    ccol = node(nt, "ShaderNodeRGB", name="CloudColor")
    ccol.outputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
    lit = node(nt, "ShaderNodeHueSaturation", {"Saturation": 0.2, "Value": 2.4, "Color": sky.outputs[0]})
    lit = node(nt, "ShaderNodeMix", {"Factor": 1.0, "A": out(lit), "B": out(ccol)}, data_type="RGBA", blend_type="MULTIPLY")
    dark = node(nt, "ShaderNodeHueSaturation", {"Value": 0.6, "Color": out(lit, "Result")})
    shade = node(nt, "ShaderNodeMix", {"Factor": math_node(nt, "MULTIPLY", out(cloud, "Fac"), 0.9),
                                       "A": out(lit, "Result"), "B": out(dark)}, data_type="RGBA")
    sky_mix = node(nt, "ShaderNodeMix", {"Factor": math_node(nt, "MULTIPLY", mask, 0.9), "A": sky.outputs[0],
                                         "B": out(shade, "Result")}, data_type="RGBA")
    nt.links.new(out(sky_mix, "Result"), nt.nodes["Background"].inputs["Color"])
    world.mist_settings.start, world.mist_settings.depth, world.mist_settings.falloff = 60.0, 1400.0, "LINEAR"

    sun = bpy.data.lights.new("Sun", "SUN")
    sun.angle = math.radians(1.2)
    new_object("Sun", sun, coll)
    for k, (x, y) in enumerate(VALLEY_COLS + SIDE_COLS):
        z = float(terrain_h(x, y)) if (x, y) in VALLEY_COLS else GRADE
        lt = bpy.data.lights.new(f"Uplight{k:02d}", "SPOT")
        lt.spot_size, lt.spot_blend, lt.color = math.radians(70), 0.6, (1.0, 0.78, 0.5)
        lt.shadow_soft_size = 0.2
        ob = new_object(f"Uplight{k:02d}", lt, coll, (x + 1.1, y, z + 0.3))
        ob.rotation_euler = Vector((-0.3, 0, 1)).to_track_quat("-Z", "Y").to_euler()
        ob["dusk_energy"] = 2500.0 if (x, y) in VALLEY_COLS else 900.0

    r = scene.render
    r.engine = "CYCLES"
    r.resolution_x, r.resolution_y, r.resolution_percentage = 1920, 1080, 100
    r.fps = 24
    r.use_motion_blur = True
    r.motion_blur_shutter = 0.5
    scene.frame_start, scene.frame_end = 1, FRAMES
    c = scene.cycles
    c.device = "CPU"
    c.samples = 64
    c.use_adaptive_sampling = True
    c.adaptive_threshold = 0.02
    c.use_denoising = True
    c.denoiser = "OPENIMAGEDENOISE"
    c.use_light_tree = True
    c.max_bounces, c.diffuse_bounces, c.glossy_bounces = 8, 3, 3
    c.transmission_bounces, c.transparent_max_bounces, c.volume_bounces = 8, 24, 0
    c.caustics_reflective = c.caustics_refractive = False
    c.sample_clamp_indirect = 6.0
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_layers[0].use_pass_mist = True

    # compositor: distance haze from the mist pass + bloom for dusk lights
    r.use_compositing = True
    r.compositor_device = "CPU"
    cg = bpy.data.node_groups.new("StationComp", "CompositorNodeTree")
    cg.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    rl = cg.nodes.new("CompositorNodeRLayers")
    go = cg.nodes.new("NodeGroupOutput")
    amt = node(cg, "ShaderNodeMath", {0: rl.outputs["Mist"], 1: 0.3}, name="HazeAmount", operation="MULTIPLY")
    haze = node(cg, "ShaderNodeMix", {"Factor": out(amt), "A": rl.outputs["Image"], "B": (0.7, 0.78, 0.88, 1.0)},
                name="Haze", data_type="RGBA")
    bloom = cg.nodes.new("CompositorNodeGlare")
    bloom.name = "Bloom"
    bloom.inputs["Type"].default_value = "Bloom"
    bloom.inputs["Threshold"].default_value = 1.2
    bloom.inputs["Size"].default_value = 0.6
    cg.links.new(out(haze, "Result"), bloom.inputs["Image"])
    cg.links.new(bloom.outputs["Image"], go.inputs[0])
    scene.compositing_node_group = cg


# =============================================================== main
def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.name = "EmeraldValley"
    build_materials()
    log("materials")
    lib, libc = build_library()
    log("plant library")

    site = collection("Site")
    station = collection("Station")
    planting = collection("Planting")
    trains = collection("Trains")
    cams = collection("Cameras")

    terrain = build_terrain(site)
    log("terrain")
    *_, rsoil, gctr, gn = build_roof(station)
    log("roof")
    build_columns(station)
    _, leds = build_decks(station)
    build_bridge_and_walks(station, leds)
    lawns = build_plaza(station, lib, leds)
    leds.build("LightStrips", station)
    log("decks, bridge, walks, plaza")
    build_vines(planting)

    scatter("Forest", terrain, libc["Forest"], 0.035, "forest", 1, (0.75, 1.3), 3.2, parent=planting)
    scatter("Conifers", terrain, libc["Conifer"], 0.03, "conifer", 2, (0.7, 1.35), 3.5, parent=planting)
    scatter("Grove", terrain, libc["Grove"], 0.03, "grove", 3, (0.8, 1.2), 4.0, parent=planting)
    scatter("Understory", terrain, libc["Under"], 0.45, "under", 4, (0.6, 1.4), 0.6, tilt=0.15, parent=planting)
    scatter("Rocks", terrain, libc["Rocks"], 0.12, "rocks", 5, (0.25, 1.1), 1.0, tilt=0.6, parent=planting)
    scatter("RoofGarden", rsoil, libc["Garden"], 3.5, "garden", 6, (0.7, 1.4), 0.22, tilt=0.12, parent=planting)
    scatter("LawnGrass", lawns, libc["Garden"], 3.0, "lawn", 7, (0.6, 1.1), 0.25, tilt=0.1, parent=planting)
    scatter("Meadow", terrain, libc["Garden"], 2.5, "meadow", 8, (0.7, 1.3), 0.25, tilt=0.15, parent=planting)
    for k, (x, y) in enumerate(EMERGENT):
        new_object(f"EmergentTree{k}", lib["Emergent"].data, planting, (x, y, float(terrain_h(x, y))), (0, 0, 1.3 * k))
    log("planting")

    build_trains(trains)
    build_people(station, people_spots())
    build_cameras(cams, gctr, gn)
    build_world_and_render(scene, collection("Lights"))
    scene.camera = bpy.data.objects["CAM_Aerial"]
    scene.frame_set(100)
    station_modes.apply_mode(scene, "day")
    log("trains, people, cameras, world")

    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=OUT, compress=True)
    log(f"saved {OUT}")


if __name__ == "__main__":
    main()
