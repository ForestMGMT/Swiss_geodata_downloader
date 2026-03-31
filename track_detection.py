"""
Automatische Spurerkennung aus dem SwissAlti3D Reliefschattierungs-Bild.

Beobachtung aus Hillshade-Daten:
  - Waldstrassen  : breite gleichmässige dunkle Bänder (~4-8 m)
  - Rückegassen   : zwei parallele dunkle Linien (Radspuren, ~1 m breit, ~2 m Abstand)

Pipeline:
  1. SwissAlti3D Hillshade via WMS bei nativer Auflösung (~0.5 m/px)
  2. Waldperimeter via WFS (Fallback: gesamtes Gebiet)
  3. Graustufen + Invertieren → dunkle Linien werden hell
  4. Frangi-Filter (multi-Skala) → verstärkt lineare Strukturen
  5. Schwellenwert → Binärmaske
  6. Morphologische Bereinigung
  7. Distanz-Transform → Breitenabschätzung
  8. Klassifikation:
       Waldstrasse  → breite kontinuierliche Fläche  (Halbbreite ≥ ROAD_HALF_W)
       Rückegasse   → schmale Merkmale, Doppellinien-Prüfung
  9. Skelettierung + Vektorisierung → GeoPackage
"""

import os
import tempfile

import numpy as np
import requests
from io import BytesIO
from PIL import Image
from scipy import ndimage
from shapely.geometry import LineString, mapping
from shapely.geometry import shape as shapely_shape
from skimage.filters import frangi
from skimage.morphology import skeletonize, remove_small_objects
import geopandas as gpd
import rasterio
from rasterio.crs import CRS
from rasterio.features import rasterize as rio_rasterize
from rasterio.transform import from_bounds as transform_from_bounds

CRS_2056   = CRS.from_epsg(2056)
WMS_URL    = "https://wms.geo.admin.ch/"
WMS_LAYER  = "ch.swisstopo.swissalti3d-reliefschattierung"
NATIVE_RES   = 0.5    # SwissAlti3D native resolution [m/px]
PROCESS_RES  = 1.0    # Auflösung für Frangi-Verarbeitung [m/px] — gröber = viel schneller
MAX_PX       = 4000   # max WMS image dimension

# ── Erkennungsparameter (kalibriert auf PROCESS_RES) ─────────────────────────
FRANGI_SIGMA_MIN  = 0.5   # kleinste Linienbreite [m] → kleinste Skala
FRANGI_SIGMA_MAX  = 8.0   # grösste zu erkennende Breite [m]
FRANGI_STEP       = 1.0   # grober Schritt → weniger Skalen → viel schneller
FRANGI_THRESHOLD  = 0.01  # Schwellenwert auf dem Frangi-Ergebnis [0..1]
MIN_AREA_PX       = 20    # Mindestfläche [px²]
MIN_LENGTH_M      = 8.0   # Mindestlänge einer Vektorlinie [m]
ROAD_HALF_W_M     = 1.8   # Halbbreite ≥ Wert → Waldstrasse [m]
DUAL_BRIDGE_M     = 3.5   # Brückenbreite für Doppellinien-Detektion [m]
SIMPLIFY_M        = 1.0   # Vereinfachungstoleranz [m]


# ── 1. Hillshade via WMS laden ────────────────────────────────────────────────

def _download_hillshade(bbox_2056, status):
    """Hillshade bei nativer Auflösung laden; gibt (img_gray float32, transform, res_m) zurück."""
    x_min, y_min, x_max, y_max = bbox_2056
    w_m = x_max - x_min
    h_m = y_max - y_min

    # Zielauflösung: nativ (0.5 m/px), Deckelung bei MAX_PX
    target_w = int(w_m / NATIVE_RES)
    target_h = int(h_m / NATIVE_RES)
    scale    = min(1.0, MAX_PX / max(target_w, target_h))
    width_px = max(1, int(target_w * scale))
    height_px= max(1, int(target_h * scale))
    res_m    = w_m / width_px

    status(f"Hillshade WMS: {width_px} × {height_px} px @ {res_m:.2f} m/px …")

    r = requests.get(WMS_URL, params={
        "SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetMap",
        "LAYERS":  WMS_LAYER, "CRS": "EPSG:2056",
        "BBOX":    f"{x_min},{y_min},{x_max},{y_max}",
        "WIDTH":   width_px, "HEIGHT": height_px,
        "FORMAT":  "image/png", "STYLES": "",
    }, timeout=120)
    r.raise_for_status()
    ct = r.headers.get("Content-Type", "")
    if "xml" in ct.lower() or "text" in ct.lower():
        raise ValueError(f"WMS-Fehler: {r.text[:200]}")

    img  = Image.open(BytesIO(r.content)).convert("L")   # Graustufen 0–255
    arr  = np.array(img, dtype=np.float32) / 255.0       # normalisiert 0..1
    tf   = transform_from_bounds(x_min, y_min, x_max, y_max, width_px, height_px)
    return arr, tf, res_m, width_px, height_px


# ── 2. Waldmaske laden ────────────────────────────────────────────────────────

def _forest_mask(bbox_2056, arr_shape, transform, status):
    x_min, y_min, x_max, y_max = bbox_2056
    status("Waldperimeter laden (WFS) …")
    try:
        r = requests.get("https://wfs.geo.admin.ch/", params={
            "SERVICE": "WFS", "VERSION": "2.0.0", "REQUEST": "GetFeature",
            "typeName": "ch.swisstopo.swisstlm3d-wald",
            "outputFormat": "application/json",
            "bbox": f"{x_min},{y_min},{x_max},{y_max},urn:ogc:def:crs:EPSG::2056",
            "srsName": "EPSG:2056", "count": "1000",
        }, timeout=30)
        r.raise_for_status()
        feats = r.json().get("features", [])
        if feats:
            geoms = [shapely_shape(f["geometry"]) for f in feats if f.get("geometry")]
            mask = rio_rasterize(
                [(mapping(g), 1) for g in geoms],
                out_shape=arr_shape, transform=transform,
                fill=0, dtype=np.uint8,
            )
            status(f"Waldmaske: {len(geoms)} Polygone geladen.")
            return mask.astype(bool)
    except Exception as e:
        status(f"WFS fehlgeschlagen ({e}) — kein Waldfilter angewendet.")
    # Fallback: gesamtes Gebiet
    return np.ones(arr_shape, dtype=bool)


# ── 3–6. Merkmalsdetektion ────────────────────────────────────────────────────

def _detect_features(gray, forest_mask, res_m, status):
    """
    Gibt zwei Binärmasken zurück: (road_mask, track_mask)
    road_mask  = breite Merkmale  (Waldstrassen)
    track_mask = schmale Merkmale (Rückegassen / Fahrspuren)
    """
    # Invertieren: dunkle Linien (Eintiefungen) werden hell
    inv = 1.0 - gray

    # Frangi-Sigmas in Pixeln aus physischen Grössen berechnen
    sig_min = max(0.5, FRANGI_SIGMA_MIN / res_m)
    sig_max = max(sig_min + 0.5, FRANGI_SIGMA_MAX / res_m)
    sigmas  = np.arange(sig_min, sig_max, FRANGI_STEP / res_m)
    sigmas  = sigmas[sigmas > 0]

    status(f"Frangi-Filter ({len(sigmas)} Skalen, σ {sig_min:.1f}–{sig_max:.1f} px) …")
    resp = frangi(inv, sigmas=sigmas, black_ridges=False)

    # Normalisieren
    if resp.max() > 0:
        resp = resp / resp.max()

    # Schwellenwert
    binary = (resp > FRANGI_THRESHOLD) & forest_mask
    binary = remove_small_objects(binary, min_size=MIN_AREA_PX)

    # Distanz-Transform für Breitenabschätzung
    dist = ndimage.distance_transform_edt(binary)   # [px]

    # Breitenklassifikation (in Metern)
    road_half_px  = ROAD_HALF_W_M / res_m
    road_mask  = binary & (dist >= road_half_px)
    track_mask = binary & (dist <  road_half_px)

    # Rückegassen: Doppellinien zusammenführen
    # Brücke über den Mittelstreifen zwischen den beiden Radspuren
    bridge_px = max(3, int(DUAL_BRIDGE_M / res_m))
    struct_h  = np.ones((1, bridge_px), bool)
    struct_v  = np.ones((bridge_px, 1), bool)
    bridged   = (ndimage.binary_closing(track_mask, structure=struct_h, iterations=2) |
                 ndimage.binary_closing(track_mask, structure=struct_v, iterations=2))
    bridged   = remove_small_objects(bridged, min_size=MIN_AREA_PX * 2)

    # Nach dem Brücken: Doppellinien erscheinen nun als ~3-4 m breite Fläche
    # → nicht breiter als Waldstrasse; unterscheide durch Ursprungssignal
    dist2 = ndimage.distance_transform_edt(bridged)
    # Bridged-Flächen die doch sehr breit sind → eher Waldstrasse
    road_mask  = road_mask | (bridged & (dist2 >= road_half_px * 1.2))
    track_final= bridged & (dist2 <  road_half_px * 1.2)

    return road_mask, track_final, dist, dist2


# ── 7. Skelettierung + Vektorisierung ─────────────────────────────────────────

def _vectorize(binary_mask, transform, dist_arr, res_m, min_length_m):
    """Binärmaske → Liste von (LineString, avg_half_width_m)."""
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("networkx fehlt — bitte 'pip install networkx' ausführen.")

    skel = skeletonize(binary_mask)
    rows, cols = np.where(skel)
    if len(rows) == 0:
        return []

    min_px  = max(3, int(min_length_m / res_m))
    pset    = set(zip(rows.tolist(), cols.tolist()))
    G       = nx.Graph()

    for r, c in pset:
        G.add_node((r, c), hw=float(dist_arr[r, c]) * res_m)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0: continue
                nb = (r + dr, c + dc)
                if nb in pset:
                    G.add_edge((r, c), nb)

    result = []
    for comp in nx.connected_components(G):
        if len(comp) < min_px:
            continue
        subG   = G.subgraph(comp).copy()
        branch = {n for n in subG if subG.degree(n) >= 3}

        if branch:
            tmp  = subG.copy(); tmp.remove_nodes_from(branch)
            segs = []
            for seg in nx.connected_components(tmp):
                adj = {nb for n in seg for nb in subG.neighbors(n) if nb in branch}
                segs.append(seg | adj)
        else:
            segs = [set(comp)]

        for seg in segs:
            if len(seg) < max(2, min_px // 2):
                continue
            subH = subG.subgraph(seg)
            ends = [n for n in subH if subH.degree(n) <= 1]
            if len(ends) >= 2:
                try:    path = nx.shortest_path(subH, ends[0], ends[-1])
                except: path = list(nx.dfs_preorder_nodes(subH, ends[0]))
            else:
                path = list(nx.dfs_preorder_nodes(subH, next(iter(seg))))
            if len(path) < 2:
                continue

            xs, ys   = rasterio.transform.xy(transform,
                                              [p[0] for p in path],
                                              [p[1] for p in path])
            avg_hw   = float(np.mean([subG.nodes[p]["hw"] for p in path]))
            line     = LineString(zip(xs, ys)).simplify(SIMPLIFY_M)
            if line.length >= min_length_m:
                result.append((line, avg_hw))

    return result


# ── Haupt-Funktion ─────────────────────────────────────────────────────────────

def detect_forest_tracks(bbox_wgs84, bbox_2056, status_callback=None):
    """
    Erkennt Waldstrassen und Rückegassen/Fahrspuren aus dem SwissAlti3D Hillshade.

    Gibt Pfad zu 'fahrspuren.gpkg' zurück mit Layern:
      'Waldstrassen'  — breite Merkmale ≥ ~3.5 m
      'Fahrspuren'    — schmale Merkmale / Rückegassen (Radspuren)
    """
    def status(msg):
        if status_callback: status_callback(msg)

    # 1. Hillshade laden
    gray, tf, res_m, w_px, h_px = _download_hillshade(bbox_2056, status)
    status(f"Hillshade geladen: {w_px}×{h_px} px @ {res_m:.2f} m/px")

    # Auf PROCESS_RES herunterskalieren für schnellere Verarbeitung
    if res_m < PROCESS_RES:
        from skimage.transform import resize as sk_resize
        factor   = res_m / PROCESS_RES
        new_h    = max(1, int(h_px * factor))
        new_w    = max(1, int(w_px * factor))
        status(f"Herunterskalierung auf {new_w}×{new_h} px @ {PROCESS_RES:.1f} m/px für Verarbeitung …")
        gray     = sk_resize(gray, (new_h, new_w), anti_aliasing=True).astype(np.float32)
        tf       = transform_from_bounds(*bbox_2056, new_w, new_h)
        res_m    = PROCESS_RES
        w_px, h_px = new_w, new_h

    # 2. Waldmaske
    fmask = _forest_mask(bbox_2056, (h_px, w_px), tf, status)

    # 3–6. Merkmale erkennen
    status("Linienmerkmale werden erkannt …")
    road_bin, track_bin, dist_road, dist_track = _detect_features(gray, fmask, res_m, status)

    n_road  = int(road_bin.sum())
    n_track = int(track_bin.sum())
    status(f"Erkannte Pixel: {n_road} Waldstrasse, {n_track} Fahrspur/Rückegasse")

    if n_road + n_track < MIN_AREA_PX * 2:
        raise ValueError(
            "Keine Fahrspuren erkannt. Bitte ein Waldgebiet wählen und sicherstellen, "
            "dass SwissAlti3D-Daten für das Gebiet verfügbar sind."
        )

    # 7. Vektorisieren
    status("Waldstrassen vektorisieren …")
    roads  = _vectorize(road_bin,  tf, dist_road,  res_m, MIN_LENGTH_M)
    status("Fahrspuren vektorisieren …")
    tracks = _vectorize(track_bin, tf, dist_track, res_m, MIN_LENGTH_M)
    status(f"Vektoren: {len(roads)} Waldstrassen, {len(tracks)} Fahrspuren")

    # 8. GeoPackage
    def make_gdf(pairs, kat):
        if not pairs:
            return gpd.GeoDataFrame(
                {"kategorie": [], "breite_m": [], "geometry": []},
                geometry="geometry", crs="EPSG:2056",
            )
        return gpd.GeoDataFrame(
            {"kategorie": [kat] * len(pairs),
             "breite_m":  [round(hw * 2, 1) for _, hw in pairs]},
            geometry=[l for l, _ in pairs],
            crs="EPSG:2056",
        )

    out_dir  = tempfile.mkdtemp(prefix="spur_out_")
    out_path = os.path.join(out_dir, "fahrspuren.gpkg")
    make_gdf(roads,  "Waldstrasse"        ).to_file(out_path, driver="GPKG", layer="Waldstrassen", mode="w")
    make_gdf(tracks, "Rückegasse/Fahrspur").to_file(out_path, driver="GPKG", layer="Fahrspuren",   mode="a")
    status("GeoPackage erstellt ✓")
    return out_path
