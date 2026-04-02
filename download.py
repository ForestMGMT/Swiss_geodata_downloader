"""
Download geodata layers as GeoTIFFs or GeoPackages.

Sources:
  stac          — swisstopo/BAFU STAC catalog, COG windowed read → float32 GeoTIFF
  wms           — swisstopo WMS GetMap               → RGB uint8 GeoTIFF
  arcgis_vector — Kanton Bern ArcGIS FeatureServer   → GeoPackage (full attribute table)
"""

import os
import tempfile
from io import BytesIO

import numpy as np
import requests
import rasterio
from PIL import Image
from rasterio.crs import CRS
from rasterio.env import Env
from rasterio.merge import merge
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.windows import from_bounds as window_from_bounds
from shapely.geometry import box

STAC_BASE   = "https://data.geo.admin.ch/api/stac/v0.9"
WMS_URL     = "https://wms.geo.admin.ch/"
ARCGIS_BASE = "https://www.map.apps.be.ch/geoservice4/rest/services/a42top"
CRS_2056    = CRS.from_epsg(2056)
MAX_PX      = 4000   # max image dimension for WMS requests

LAYERS = {
    # ── Nationale Geodaten (swisstopo / BAFU) ─────────────────────────────────
    # LiDAR VHM — 0.5 m resolution, derived from swissSURFACE3D point cloud
    # (vegetation classes 3/4/5 rasterized at 0.5 m grid, leaf-off, updated annually).
    # Higher-res replacement for the stereo-based 1 m VHM.
    # Source: BAFU/LFI via swisstopo STAC + WMS.
    # STAC collection: ch.bafu.landesforstinventar-vegetationshoehenmodell_lidar
    # WMS layer:       ch.bafu.landesforstinventar-vegetationshoehenmodell_lidar
    # VHM stereo — 1 m resolution, derived from stereo aerial imagery (ADS-80/100).
    # Full Switzerland coverage. Use this if the LiDAR version returns no data.
    # STAC collection: ch.bafu.landesforstinventar-vegetationshoehenmodell
    "vhm_stereo": {
        "category":   "national",
        "source":     "stac",
        "collection": "ch.bafu.landesforstinventar-vegetationshoehenmodell",
        "file_stem":  "vegetationshoehenmodell_1m",
        "label":      "Vegetationshöhenmodell Stereo 1 m — volle CH-Abdeckung (BAFU/LFI)",
    },
    # VHM LiDAR — 0.5 m resolution, derived from swissSURFACE3D point cloud
    # (vegetation classes 3/4/5 rasterized at 0.5 m grid, leaf-off, updated annually).
    # Coverage is incomplete (~1/6 of Switzerland added per year) → black result = no coverage yet.
    # Pixel values = tree height in metres. WMS intentionally not used (returns rendered RGB, not values).
    # STAC collection: ch.bafu.landesforstinventar-vegetationshoehenmodell_lidar
    "vhm_lidar": {
        "category":   "national",
        "source":     "stac",
        "collection": "ch.bafu.landesforstinventar-vegetationshoehenmodell_lidar",
        "file_stem":  "vegetationshoehenmodell_lidar_05m",
        "label":      "Vegetationshöhenmodell LiDAR 0.5 m — wo verfügbar (BAFU/LFI)",
    },
    "alti_relief": {
        "category":  "national",
        "source":    "wms",
        "wms_layer": "ch.swisstopo.swissalti3d-reliefschattierung",
        "file_stem": "swissalti3d_relief_multidirektional",
        "label":     "SwissAlti3D Reliefschattierung multidirektional (swisstopo)",
    },
    "swissimage": {
        "category":  "national",
        "source":    "wms",
        "wms_layer": "ch.swisstopo.swissimage",
        "file_stem": "swissimage",
        "label":     "SWISSIMAGE Luftbild aktuell (swisstopo)",
    },
    # ── Kantonale Geodaten — Kanton Zürich ───────────────────────────────────
    "zh_schutzwald": {
        "category":  "zh",
        "source":    "wms",
        "wms_url":   "https://wms.zh.ch/WaldSWZHWMS",
        "wms_layer": "schutzwald",
        "file_stem": "zh_schutzwald",
        "label":     "Schutzwald (Kt. Zürich)",
    },
    "zh_wep_waldfunktionen": {
        "category":  "zh",
        "source":    "wms",
        "wms_url":   "https://wms.zh.ch/OGDWaldWEP3ZH",
        "wms_layer": "vorrang-holznutzung,vorrang-biologische-vielfalt,vorrang-schutz,gebiete-ohne-vorrang",
        "file_stem": "zh_wep_waldfunktionen",
        "label":     "WEP Waldfunktionen (Kt. Zürich)",
    },
    "zh_wep_besondere_ziele": {
        "category":  "zh",
        "source":    "wms",
        "wms_url":   "https://wms.zh.ch/OGDWaldWEPZH",
        "wms_layer": "b1-naturwaldreservate,b2-waldstandorte-von-naturkundlicher-bedeutung-wnb,b3-dauernd-lichte-waelder,b4-eichenfoerderung,b6-waldrandfoerderung,s1-gravitative-naturgefahren,e1-haeufig-begangene-waelder",
        "file_stem": "zh_wep_besondere_ziele",
        "label":     "WEP Besondere Ziele (Kt. Zürich)",
    },
    "zh_waldstandorte_wnb": {
        "category":  "zh",
        "source":    "wms",
        "wms_url":   "https://wms.zh.ch/WaldWNBZHWMS",
        "wms_layer": "waldstandorte-von-naturkundlicher-bedeutung-wnb",
        "file_stem": "zh_waldstandorte_wnb",
        "label":     "Waldstandorte naturkundl. Bedeutung WNB (Kt. Zürich)",
    },
    "zh_waldgesellschaften": {
        "category":  "zh",
        "source":    "wms",
        "wms_url":   "https://wms.zh.ch/WaldVKWMS",
        "wms_layer": "waldgesellschaften",
        "file_stem": "zh_waldgesellschaften",
        "label":     "Waldgesellschaften / Vegetationskartierung (Kt. Zürich)",
    },
    "zh_forstkreise": {
        "category":  "zh",
        "source":    "wms",
        "wms_url":   "https://wms.zh.ch/OGDAdminZH",
        "wms_layer": "forstkreise",
        "file_stem": "zh_forstkreise",
        "label":     "Forstkreise (Kt. Zürich)",
    },
    # ── Kantonale Geodaten — Kanton Aargau ───────────────────────────────────
    # Phytosociological survey polygons with full attribute table (STAO_87, STAO_G10,
    # E_KLASSE, STANDORT, GEOLOGIE, VERD_RISK). Downloaded as GeoPackage via ArcGIS
    # REST MapServer query endpoint (layer 1 = verdichtet, full attribute table).
    # Source: AGIS Kanton Aargau — https://www.ag.ch/geoportal/rest/services/w_pflanzensoziologie/MapServer
    "ag_pflanzensoziologie": {
        "category":        "ag",
        "source":          "arcgis_vector",
        "arcgis_base_url": "https://www.ag.ch/geoportal/rest/services/w_pflanzensoziologie/MapServer",
        "layer_ids":       [1],
        "file_stem":       "ag_pflanzensoziologische_kartierung",
        "label":           "Pflanzensoziologische Kartierung (Kt. Aargau)",
        "style_field":     "STAO_87",   # categorized fill + label per Standorttyp
    },
    "ag_wni": {
        "category":  "ag",
        "source":    "wms",
        "wms_url":   "https://wms.geo.ag.ch/public/ch_ag_geo_aw_wni/wms",
        "wms_layer": "ch_ag_geo_aw_wni_01",
        "file_stem": "ag_waldnaturschutzinventar",
        "label":     "Waldnaturschutzinventar WNI (Kt. Aargau)",
    },
    "ag_nkbw": {
        "category":  "ag",
        "source":    "wms",
        "wms_url":   "https://wms.geo.ag.ch/public/ch_ag_geo_are_rp11wni/wms",
        "wms_layer": "ch_ag_geo_are_rp11wni_01",
        "file_stem": "ag_naturschutz_wald",
        "label":     "Naturschutzgebiete im Wald NkBW (Kt. Aargau)",
    },
    "ag_waldreservate": {
        "category":  "ag",
        "source":    "wms",
        "wms_url":   "https://geodienste.ch/db/waldreservate_v2_0_0/deu",
        "wms_layer": "waldreservat",
        "file_stem": "waldreservate",
        "label":     "Waldreservate rechtskräftig (national, inkl. AG)",
    },
    # ── Kantonale Geodaten — Kanton Bern ──────────────────────────────────────
    "be_wsp": {
        "category":  "be",
        "source":    "arcgis_vector",
        "service":   "tf_biota01_n_ms",
        "layer_ids": [13634],   # WSP_WSPSTR_KMGDM1 — Waldstrassen (Linien)
        "file_stem": "be_waldstrassenplan",
        "label":     "Waldstrassenplan (Kt. Bern)",
    },
    "be_waldinfo": {
        "category":  "be",
        "source":    "arcgis_vector",
        "service":   "tf_biota01_n_ms",
        "layer_ids": [4479],    # WALDINFO_TBK_KMGDM1 — Bestandeskarte (Polygone)
        "file_stem": "be_waldinformation",
        "label":     "Waldinformation / Bestandeskarte (Kt. Bern)",
    },
    "be_dipanu": {
        "category":  "be",
        "source":    "arcgis_vector",
        "service":   "tf_planningcadastre01_n_ms",
        "layer_ids": [13541],   # DIPANU_DIPANUF_KMGDM1 — Parzellen (Polygone)
        "file_stem": "be_parzellennummern",
        "label":     "Parzellennummern / Kataster (Kt. Bern)",
    },
}


# ── STAC helpers ───────────────────────────────────────────────────────────────

def _query_stac(collection, bbox_wgs84):
    url    = f"{STAC_BASE}/collections/{collection}/items"
    params = {"bbox": ",".join(f"{v:.6f}" for v in bbox_wgs84), "limit": 50}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("features", [])


def _best_tif_url(feature):
    assets     = feature.get("assets", {})
    candidates = [
        (k, a["href"]) for k, a in assets.items()
        if a.get("href", "").lower().endswith((".tif", ".tiff"))
    ]
    if not candidates:
        return None
    for _, href in candidates:
        if "sentinel" not in href.lower():
            return href
    return candidates[0][1]


# ── STAC/COG tile download ─────────────────────────────────────────────────────

def _read_cog_window(url, bbox_2056, output_path):
    x_min, y_min, x_max, y_max = bbox_2056
    with Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
             CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
             GDAL_HTTP_TIMEOUT="60"):
        with rasterio.open(url) as src:
            if src.crs != CRS_2056:
                from rasterio.warp import transform_bounds
                x_min, y_min, x_max, y_max = transform_bounds(
                    CRS_2056, src.crs, x_min, y_min, x_max, y_max)
            window = window_from_bounds(x_min, y_min, x_max, y_max, src.transform)
            if int(window.width) <= 0 or int(window.height) <= 0:
                raise ValueError("Bbox does not overlap with this tile.")
            data          = src.read(window=window)
            out_transform = src.window_transform(window)
            out_meta      = src.meta.copy()
            out_meta.update(driver="GTiff", height=data.shape[1],
                            width=data.shape[2], transform=out_transform,
                            compress="lzw", crs=src.crs)
            with rasterio.open(output_path, "w", **out_meta) as dst:
                dst.write(data)


def _download_full_and_clip(url, bbox_2056, output_path):
    from rasterio.mask import mask as rio_mask
    tmp = output_path + ".tmp.tif"
    r = requests.get(url, stream=True, timeout=600)
    r.raise_for_status()
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(chunk_size=65536):
            f.write(chunk)
    geom = box(*bbox_2056)
    with rasterio.open(tmp) as src:
        clipped, out_transform = rio_mask(src, [geom], crop=True)
        out_meta = src.meta.copy()
        out_meta.update(driver="GTiff", height=clipped.shape[1],
                        width=clipped.shape[2], transform=out_transform,
                        compress="lzw")
        with rasterio.open(output_path, "w", **out_meta) as dst:
            dst.write(clipped)
    os.remove(tmp)


def _download_stac(layer, bbox_wgs84, bbox_2056, status):
    status("STAC-Katalog wird abgefragt...")
    features = _query_stac(layer["collection"], bbox_wgs84)
    if not features:
        raise ValueError(
            f"Keine Daten für das gewählte Gebiet in Collection "
            f"'{layer['collection']}'. Bitte ein anderes Gebiet wählen."
        )
    urls = []
    seen = set()
    for feat in features:
        url = _best_tif_url(feat)
        if url and url not in seen:
            urls.append(url)
            seen.add(url)
    if not urls:
        raise ValueError("STAC lieferte Einträge, aber keine GeoTIFF-Assets.")

    status(f"{len(urls)} Kachel(n) gefunden. Download läuft...")
    out_dir    = tempfile.mkdtemp(prefix="geodl_")
    tile_paths = []
    for i, url in enumerate(urls):
        status(f"Kachel {i + 1} / {len(urls)} wird gelesen...")
        tile_path = os.path.join(out_dir, f"tile_{i}.tif")
        try:
            _read_cog_window(url, bbox_2056, tile_path)
            tile_paths.append(tile_path)
        except Exception as e:
            status(f"COG-Lesen fehlgeschlagen ({e}), lade vollständige Kachel...")
            try:
                _download_full_and_clip(url, bbox_2056, tile_path)
                tile_paths.append(tile_path)
            except Exception as e2:
                status(f"Kachel {i + 1} übersprungen: {e2}")
    if not tile_paths:
        raise ValueError("Alle Kachel-Downloads fehlgeschlagen.")

    merged_tif = os.path.join(out_dir, "merged.tif")
    if len(tile_paths) == 1:
        import shutil
        shutil.copy(tile_paths[0], merged_tif)
    else:
        status("Kacheln werden zusammengesetzt...")
        from rasterio.mask import mask as rio_mask
        datasets = [rasterio.open(p) for p in tile_paths]
        mosaic, out_transform = merge(datasets)
        out_meta = datasets[0].meta.copy()
        out_meta.update(driver="GTiff", height=mosaic.shape[1],
                        width=mosaic.shape[2], transform=out_transform,
                        compress="lzw")
        raw_path = os.path.join(out_dir, "mosaic_raw.tif")
        with rasterio.open(raw_path, "w", **out_meta) as dst:
            dst.write(mosaic)
        for ds in datasets:
            ds.close()
        status("Auf Auswahl zuschneiden...")
        geom = box(*bbox_2056)
        with rasterio.open(raw_path) as src:
            clipped, clip_transform = rio_mask(src, [geom], crop=True)
            clip_meta = src.meta.copy()
            clip_meta.update(height=clipped.shape[1], width=clipped.shape[2],
                             transform=clip_transform)
        with rasterio.open(merged_tif, "w", **clip_meta) as dst:
            dst.write(clipped)

    output_path = os.path.join(out_dir, f"{layer['file_stem']}.tif")
    import shutil
    shutil.move(merged_tif, output_path)
    return output_path


# ── WMS download ───────────────────────────────────────────────────────────────

def _calc_px(bbox_2056):
    x_min, y_min, x_max, y_max = bbox_2056
    w_m, h_m = x_max - x_min, y_max - y_min
    scale = min(1.0, MAX_PX / max(w_m, h_m))
    return max(1, int(w_m * scale)), max(1, int(h_m * scale)), w_m / max(1, int(w_m * scale))


def _download_wms(layer, bbox_2056, status):
    x_min, y_min, x_max, y_max = bbox_2056
    width_px, height_px, res_m = _calc_px(bbox_2056)
    status(f"WMS-Anfrage: {width_px} × {height_px} px (~{res_m:.2f} m/px)...")
    url = layer.get("wms_url", WMS_URL)
    params = {
        "SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetMap",
        "LAYERS": layer["wms_layer"], "CRS": "EPSG:2056",
        "BBOX": f"{x_min},{y_min},{x_max},{y_max}",
        "WIDTH": width_px, "HEIGHT": height_px,
        "FORMAT": "image/jpeg", "STYLES": "",
    }
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    ct = r.headers.get("Content-Type", "")
    if "xml" in ct.lower() or "text" in ct.lower():
        raise ValueError(f"WMS-Fehler: {r.text[:300]}")

    status("Bild wird georeferenziert und gespeichert...")
    img = Image.open(BytesIO(r.content)).convert("RGB")
    arr = np.moveaxis(np.array(img), -1, 0)

    out_dir     = tempfile.mkdtemp(prefix="geodl_")
    output_path = os.path.join(out_dir, f"{layer['file_stem']}.tif")
    transform   = transform_from_bounds(x_min, y_min, x_max, y_max, width_px, height_px)

    with rasterio.open(output_path, "w", driver="GTiff",
                       height=height_px, width=width_px, count=3, dtype="uint8",
                       crs=CRS_2056, transform=transform,
                       compress="jpeg", photometric="RGB") as dst:
        dst.write(arr)
    return output_path


# ── QGIS style helpers ────────────────────────────────────────────────────────

def _generate_qml(gdf, style_field: str) -> str:
    """Generate a QGIS QML style: categorized fill colours + centroid labels."""
    import colorsys, html

    values = sorted(gdf[style_field].dropna().astype(str).unique())
    n      = max(len(values), 1)

    cats, syms = "", ""
    for i, val in enumerate(values):
        h       = i / n
        r, g, b = (int(x * 255) for x in colorsys.hsv_to_rgb(h, 0.65, 0.85))
        safe    = html.escape(val)
        cats += f'      <category value="{safe}" label="{safe}" symbol="{i}" render="true"/>\n'
        syms += (
            f'      <symbol type="fill" name="{i}" clip_to_extent="1" alpha="1" force_rhr="0">\n'
            f'        <layer class="SimpleFill" enabled="1" pass="0" locked="0">\n'
            f'          <Option type="Map">\n'
            f'            <Option value="{r},{g},{b},180" name="color" type="QString"/>\n'
            f'            <Option value="60,60,60,255" name="outline_color" type="QString"/>\n'
            f'            <Option value="solid" name="outline_style" type="QString"/>\n'
            f'            <Option value="0.26" name="outline_width" type="QString"/>\n'
            f'            <Option value="MM" name="outline_width_unit" type="QString"/>\n'
            f'            <Option value="solid" name="style" type="QString"/>\n'
            f'          </Option>\n'
            f'        </layer>\n'
            f'      </symbol>\n'
        )

    return f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28" styleCategories="AllStyleCategories">
  <renderer-v2 type="categorizedSymbol" attr="{style_field}" enableorderby="0" symbollevels="0" forceraster="0">
    <categories>
{cats}    </categories>
    <symbols>
{syms}    </symbols>
  </renderer-v2>
  <labeling type="simple">
    <settings calloutType="simple">
      <text-style fieldName="{style_field}" fontSize="8" fontWeight="50"
        textColor="0,0,0,255" fontFamily="Arial" textOpacity="1"
        fontItalic="0" fontUnderline="0" fontStrikeout="0"
        fontSizeUnit="Point" isExpression="0" namedStyle="Regular">
        <text-buffer bufferEnabled="1" bufferSize="1" bufferSizeUnits="MM"
          bufferColor="255,255,255,255" bufferOpacity="1" bufferJoinStyle="128" bufferNoFill="1"/>
      </text-style>
      <text-format wrapChar="" multilineAlign="3"/>
      <placement placement="1" centroidInside="1" centroidWhole="0"
        fitInPolygonOnly="0" dist="0" distUnits="MM" offsetType="0"
        xOffset="0" yOffset="0" offsetUnits="MM" rotationAngle="0"/>
      <rendering drawLabels="1" obstacle="1" obstacleFactor="1"
        minFeatureSize="0" maxNumLabels="2000" displayAll="0"
        fontLimitPixelSize="0" fontMinPixelSize="3" fontMaxPixelSize="10000"/>
    </settings>
  </labeling>
  <blendMode>0</blendMode>
  <featureBlendMode>0</featureBlendMode>
  <layerOpacity>1</layerOpacity>
</qgis>"""


def _embed_qgis_style(gpkg_path: str, layer_name: str, qml: str) -> None:
    """Insert a QML style into the GeoPackage layer_styles table (auto-loaded by QGIS)."""
    import sqlite3, datetime
    con = sqlite3.connect(gpkg_path)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS layer_styles (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            f_table_catalog   TEXT DEFAULT '',
            f_table_schema    TEXT DEFAULT '',
            f_table_name      TEXT NOT NULL,
            f_geometry_column TEXT DEFAULT 'geom',
            styleName         TEXT NOT NULL,
            styleQML          TEXT,
            styleSLD          TEXT DEFAULT '',
            useAsDefault      BOOLEAN DEFAULT 1,
            description       TEXT DEFAULT '',
            owner             TEXT DEFAULT '',
            ui                TEXT DEFAULT '',
            update_time       DATETIME
        )
    """)
    cur.execute(
        "INSERT INTO layer_styles "
        "(f_table_name, f_geometry_column, styleName, styleQML, useAsDefault, update_time) "
        "VALUES (?, 'geom', 'default', ?, 1, ?)",
        (layer_name, qml, datetime.datetime.utcnow().isoformat()),
    )
    con.commit()
    con.close()


# ── ArcGIS vector download ─────────────────────────────────────────────────────

def _download_arcgis_vector(layer, bbox_2056, status):
    try:
        import geopandas as gpd
    except ImportError:
        raise ImportError(
            "geopandas ist nicht installiert. Bitte 'pip install geopandas' ausführen."
        )

    x_min, y_min, x_max, y_max = bbox_2056
    out_dir     = tempfile.mkdtemp(prefix="geodl_")
    output_path = os.path.join(out_dir, f"{layer['file_stem']}.gpkg")
    written     = False
    last_gdf    = None

    for layer_id in layer["layer_ids"]:
        status(f"Vektor-Abfrage Layer {layer_id}...")
        base   = layer.get("arcgis_base_url") or f"{ARCGIS_BASE}/{layer['service']}/MapServer"
        url    = f"{base}/{layer_id}/query"
        offset = 0
        page   = 1000
        all_features = []

        while True:
            params = {
                "where":             "1=1",
                "outFields":         "*",
                "geometry":          f"{x_min},{y_min},{x_max},{y_max}",
                "geometryType":      "esriGeometryEnvelope",
                "spatialRel":        "esriSpatialRelIntersects",
                "inSR":              "2056",
                "outSR":             "2056",
                "resultOffset":      offset,
                "resultRecordCount": page,
                "f":                 "geojson",
            }
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            data = r.json()

            if "error" in data:
                status(f"Layer {layer_id} übersprungen: {data['error'].get('message', '?')}")
                break

            features = data.get("features", [])
            all_features.extend(features)
            status(f"Layer {layer_id}: {len(all_features)} Features geladen...")

            if len(features) < page:
                break
            offset += page

        if all_features:
            gdf      = gpd.GeoDataFrame.from_features(all_features, crs="EPSG:2056")
            mode     = "w" if not written else "a"
            gdf.to_file(output_path, driver="GPKG",
                        layer=layer["file_stem"], mode=mode)
            written  = True
            last_gdf = gdf

    if not written:
        raise ValueError(
            "Keine Features im gewählten Gebiet gefunden. "
            "Bitte Gebiet vergrössern oder anderen Layer wählen."
        )

    if layer.get("style_field") and last_gdf is not None:
        status("QGIS-Stil wird eingebettet …")
        qml = _generate_qml(last_gdf, layer["style_field"])
        _embed_qgis_style(output_path, layer["file_stem"], qml)

    return output_path


# ── Clipping ───────────────────────────────────────────────────────────────────

def _clip_raster(path, geom, status):
    """Clip a GeoTIFF in-place to the given shapely geometry (EPSG:2056)."""
    from rasterio.mask import mask as rio_mask
    status("Raster wird auf Auswahl zugeschnitten...")
    with rasterio.open(path) as src:
        clipped, clip_transform = rio_mask(src, [geom], crop=True, nodata=src.nodata)
        meta = src.meta.copy()
        meta.update(height=clipped.shape[1], width=clipped.shape[2],
                    transform=clip_transform)
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(clipped)


def _clip_vector(path, geom, status):
    """Clip all layers in a GeoPackage in-place to the given shapely geometry."""
    import shutil
    import geopandas as gpd
    import pyogrio

    status("Vektordaten werden auf Auswahl zugeschnitten...")
    clip_gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:2056")
    layers   = pyogrio.list_layers(path)[:, 0].tolist()
    tmp_path = path + ".clip_tmp.gpkg"

    first = True
    for layer_name in layers:
        gdf     = gpd.read_file(path, layer=layer_name)
        clipped = gpd.clip(gdf, clip_gdf) if not gdf.empty else gdf
        clipped.to_file(tmp_path, driver="GPKG", layer=layer_name,
                        mode="w" if first else "a")
        first = False

    shutil.move(tmp_path, path)


# ── Public API ─────────────────────────────────────────────────────────────────

def download_layer(layer_key, bbox_wgs84, bbox_2056,
                   clip_geom=None, status_callback=None):
    def status(msg):
        if status_callback:
            status_callback(msg)

    layer = LAYERS[layer_key]
    src   = layer["source"]

    if src == "stac":
        path = _download_stac(layer, bbox_wgs84, bbox_2056, status)
    elif src == "wms":
        path = _download_wms(layer, bbox_2056, status)
    elif src == "arcgis_vector":
        path = _download_arcgis_vector(layer, bbox_2056, status)
    else:
        raise ValueError(f"Unbekannte Quelle: {src}")

    if clip_geom is not None:
        if path.endswith(".tif"):
            _clip_raster(path, clip_geom, status)
        elif path.endswith(".gpkg"):
            _clip_vector(path, clip_geom, status)

    return path
