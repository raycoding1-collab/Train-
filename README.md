# Emerald Valley Station (翠谷站)

Procedural Blender model of a futuristic green rail hub: undulating gridshell roof on
branching "dendriform" columns, a forest valley with a stream between two elevated
maglev decks, a canopy walk, vine curtains, roof gardens, solar glazing and a reflecting pool.

![Aerial, day](renders/CAM_Aerial_day.jpg)

## Status
- `build_station.py` builds `station.blend` from scratch (~40 s, no external assets).
- `render.py` renders any camera in day or dusk light with Cycles; stills and the
  fly-through are in [`renders/`](renders/).
- Targets **Blender 5.2 LTS** (Cycles path tracing, OpenImageDenoise, geometry-nodes
  scattering, multiple-scattering sky, AgX colour).

## What's in the scene
| Element | How it's made |
|---|---|
| Gridshell roof | 216 m x 106 m diagrid of glulam members over an undulating (u, v) surface; clear glazing, semi-transparent PV glazing (opaque cells, clear gaps), corten roof-garden trays, two oculi with planted collars |
| Dendriform columns | 4 valley columns + 12 plaza/terrace columns; trunk -> 4 branches -> 8 tips that land on the roof |
| Valley | 3 km x 2.2 km terrain with a meandering stream, gravel trail, rocks, and geometry-nodes forest / conifer / grove / understory scatters (all instanced) |
| Maglev decks | Two U-channel guideways on piers, platforms with terrazzo floors, benches, displays, LED coves; a sky bridge with escalators |
| Canopy walk | Two timber walkways hung from the roof, a spiral stair tower to the trail and a cantilevered lookout |
| Vine curtains | ~1,500 hanging strands (~70k leaves) from the guideway planters and the roof eaves |
| Plaza | Stadium-shaped reflecting pool, lawns with trees, lamp posts, 翠谷站 sign |
| Trains | Two 6-car maglev sets: one passes through at speed, one brakes into the south platform |
| People | ~200 simple figures on the plaza, platforms, bridge, walks and trail |

## Setup
Either the Blender 5.2 LTS app from <https://www.blender.org/download/>, or Blender as a
Python module (no GUI needed; Python 3.13):
```
python3.13 -m venv blender52
blender52/bin/pip install bpy==5.2.3
```

## Build
```
blender -b --factory-startup --python build_station.py
# or
blender52/bin/python build_station.py
```

## Render
```
blender -b station.blend --python render.py -- --cam CAM_Aerial --mode day --res 960 540 --samples 16 --out renders/{cam}_{mode}.png
# or
blender52/bin/python render.py --cam all --mode both --res 1920 1080 --samples 48
```
Cameras: CAM_Aerial, CAM_Plaza, CAM_Valley, CAM_Platform, CAM_Roof, CAM_TrailUp, CAM_Fly (or `all`).
Modes: day, dusk (or `both`). `--anim` renders frames 1-240 to MP4 (trains move, CAM_Fly flies
the valley); frames are written to `renders/<name>_frames/` first, so an interrupted run resumes.
Other options: `--frame N`, `--frames A B`, `--sun-az/--sun-el` (compass degrees),
`--no-motion-blur`, `--threads N`.

Rendering uses Cycles on the CPU with OpenImageDenoise, so it works headless without a GPU.
`--engine EEVEE` is available but needs a GPU/display.

## Gallery
1920x1080, Cycles 40 samples + OpenImageDenoise, ~3-7 min per frame on 4 CPU cores.

| Day | Dusk |
|---|---|
| ![](renders/CAM_Aerial_day.jpg) | ![](renders/CAM_Aerial_dusk.jpg) |
| ![](renders/CAM_Plaza_day.jpg) | ![](renders/CAM_Plaza_dusk.jpg) |
| ![](renders/CAM_Valley_day.jpg) | ![](renders/CAM_Valley_dusk.jpg) |
| ![](renders/CAM_Platform_day.jpg) | ![](renders/CAM_Platform_dusk.jpg) |
| ![](renders/CAM_Roof_day.jpg) | ![](renders/CAM_Roof_dusk.jpg) |
| ![](renders/CAM_TrailUp_day.jpg) | ![](renders/CAM_TrailUp_dusk.jpg) |
| ![](renders/CAM_Fly_day.jpg) | ![](renders/CAM_Fly_dusk.jpg) |

## Files
- `build_station.py` - the procedural model (geometry, materials, scatters, trains, cameras, world, compositor)
- `station_modes.py` - day/dusk lighting presets, shared by the build and render scripts
- `render.py` - command-line renderer
- `renders/` - finished stills (`<camera>_<mode>.jpg`) and the fly-through
