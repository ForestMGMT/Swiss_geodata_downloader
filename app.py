import os
import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
from pyproj import Transformer
from shapely.geometry import shape as shapely_shape
from shapely.ops import transform as shapely_transform
from download import download_layer, LAYERS

st.set_page_config(page_title="Geodaten Downloader", layout="wide")
st.title("Geodaten Downloader")

# ── Karte ──────────────────────────────────────────────────────────────────────
m = folium.Map(location=[46.8, 8.2], zoom_start=8, tiles=None)
folium.TileLayer(
    tiles="https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.swissimage/default/current/3857/{z}/{x}/{y}.jpeg",
    attr="© swisstopo",
    name="SWISSIMAGE",
    max_zoom=19,
).add_to(m)
folium.TileLayer(
    tiles="https://wmts.geo.admin.ch/1.0.0/ch.bafu.landesforstinventar-vegetationshoehenmodell/default/current/3857/{z}/{x}/{y}.png",
    attr="© swisstopo / BAFU",
    name="Vegetationshöhenmodell",
    max_zoom=19,
    overlay=True,
    show=False,
    opacity=0.8,
).add_to(m)
folium.TileLayer(
    tiles="https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.swissalti3d-reliefschattierung/default/current/3857/{z}/{x}/{y}.png",
    attr="© swisstopo",
    name="SwissAlti3D Relief multidirektional",
    max_zoom=19,
    overlay=True,
    show=False,
    opacity=0.8,
).add_to(m)
folium.LayerControl(position="topright").add_to(m)
Draw(
    draw_options={
        "rectangle":    True,
        "polygon":      True,
        "polyline":     False,
        "circle":       False,
        "marker":       False,
        "circlemarker": False,
    },
    edit_options={"edit": False},
).add_to(m)

map_result = st_folium(m, width="100%", height=520, returned_objects=["all_drawings"])

# ── Geometrie + Bounding Box aus Zeichnung ────────────────────────────────────
bbox_wgs84  = None
bbox_2056   = None
clip_geom   = None   # shapely geometry in EPSG:2056 for optional clipping
area_km2    = 0.0

if map_result and map_result.get("all_drawings"):
    drawings = map_result["all_drawings"]
    if drawings:
        last     = drawings[-1]
        coords   = last["geometry"]["coordinates"][0]
        lons     = [c[0] for c in coords]
        lats     = [c[1] for c in coords]
        bbox_wgs84 = (min(lons), min(lats), max(lons), max(lats))

        tf_proj  = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
        x_min, y_min = tf_proj.transform(bbox_wgs84[0], bbox_wgs84[1])
        x_max, y_max = tf_proj.transform(bbox_wgs84[2], bbox_wgs84[3])
        bbox_2056 = (x_min, y_min, x_max, y_max)

        # Full drawn geometry transformed to EPSG:2056
        geom_wgs84 = shapely_shape(last["geometry"])
        clip_geom  = shapely_transform(tf_proj.transform, geom_wgs84)

        w_m      = x_max - x_min
        h_m      = y_max - y_min
        area_km2 = (w_m * h_m) / 1e6
        est_mb   = int(w_m) * int(h_m) * 4 * 0.4 / 1e6

        col1, col2, col3 = st.columns(3)
        col1.metric("Fläche", f"{area_km2:.2f} km²")
        col2.metric("Auflösung", f"{int(w_m):,} × {int(h_m):,} px  @ 1 m/px")
        col3.metric("Geschätzte Dateigrösse", f"{est_mb:.1f} MB")

        if area_km2 > 100:
            st.warning(f"Grosse Fläche ({area_km2:.0f} km²) — der Download kann lange dauern.")

# ── Layer-Auswahl ──────────────────────────────────────────────────────────────
layer_labels     = {v["label"]: k for k, v in LAYERS.items()}
national_options = [v["label"] for v in LAYERS.values() if v["category"] == "national"]
canton_options   = [v["label"] for v in LAYERS.values() if v["category"] == "be"]

st.subheader("Nationale Geodaten (swisstopo / BAFU)")
selected_national = st.multiselect(
    "Layer auswählen (Mehrfachauswahl möglich)",
    options=national_options,
    key="ms_national",
)

st.subheader("Kantonale Geodaten")

CANTONS = [
    ("AG",    "Aargau"),
    ("AI",    "Appenzell Innerrhoden"),
    ("AR",    "Appenzell Ausserrhoden"),
    ("BE",    "Bern"),
    ("BL/BS", "Basel-Landschaft / Basel-Stadt"),
    ("FR",    "Freiburg"),
    ("GE",    "Genf"),
    ("GL",    "Glarus"),
    ("GR",    "Graubünden"),
    ("JU",    "Jura"),
    ("LU",    "Luzern"),
    ("NE",    "Neuenburg"),
    ("NW",    "Nidwalden"),
    ("OW",    "Obwalden"),
    ("SG",    "St. Gallen"),
    ("SH",    "Schaffhausen"),
    ("SO",    "Solothurn"),
    ("SZ",    "Schwyz"),
    ("TG",    "Thurgau"),
    ("TI",    "Tessin"),
    ("UR",    "Uri"),
    ("VD",    "Waadt"),
    ("VS",    "Wallis"),
    ("ZG",    "Zug"),
    ("ZH",    "Zürich"),
]

# Map canton abbreviation → LAYERS category key
CANTON_TO_CATEGORY = {
    "BE": "be",
    "ZH": "zh",
    "AG": "ag",
    # weitere Kantone können hier ergänzt werden
}

CANTON_NOTES = {
    "AG": "Mehr kantonale Geodaten sind im Geoportal hinter einem Login verborgen.",
}

canton_display  = ["— Kanton auswählen —"] + [f"{abbr} — {name}" for abbr, name in CANTONS]
canton_choice   = st.selectbox("Kanton", canton_display, key="canton_select")

if canton_choice == "— Kanton auswählen —":
    selected_canton = []
else:
    abbr      = canton_choice.split(" — ")[0]
    cat_key   = CANTON_TO_CATEGORY.get(abbr)
    if cat_key:
        avail_options = [v["label"] for v in LAYERS.values() if v["category"] == cat_key]
        selected_canton = st.multiselect(
            "Layer auswählen (Mehrfachauswahl möglich)",
            options=avail_options,
            key="ms_canton",
        )
        if abbr in CANTON_NOTES:
            st.caption(f"ℹ️ {CANTON_NOTES[abbr]}")
    else:
        st.info(f"Für Kanton {abbr} sind noch keine Geodaten verfügbar.")
        selected_canton = []

selected_keys = [layer_labels[lbl] for lbl in selected_national + selected_canton]

# ── Download-Optionen ──────────────────────────────────────────────────────────
clip_to_shape = st.checkbox(
    "Auf Auswahl zuschneiden",
    value=True,
    help=(
        "Schneidet alle heruntergeladenen Dateien auf die gezeichnete Form zu. "
        "Ohne diese Option wird der umgebende Rahmen (Bounding Box) verwendet."
    ),
)

# ── Download ───────────────────────────────────────────────────────────────────
if "dl_results" not in st.session_state:
    st.session_state.dl_results = {}

if not bbox_2056:
    st.info("Zeichnen Sie ein Rechteck oder Polygon auf der Karte, um den Download-Bereich festzulegen.")
elif not selected_keys:
    st.info("Wählen Sie mindestens einen Layer aus.")
else:
    if st.button("Herunterladen", type="primary", disabled=(area_km2 > 200)):
        st.session_state.dl_results = {}
        geom_for_clip = clip_geom if clip_to_shape else None
        for key in selected_keys:
            label      = LAYERS[key]["label"]
            status_box = st.empty()
            try:
                path = download_layer(
                    key, bbox_wgs84, bbox_2056,
                    clip_geom=geom_for_clip,
                    status_callback=lambda msg, b=status_box: b.info(msg),
                )
                actual_mb = os.path.getsize(path) / 1e6
                status_box.success(f"{label}: fertig ({actual_mb:.1f} MB)")
                st.session_state.dl_results[key] = ("ok", path, actual_mb)
            except Exception as e:
                status_box.error(f"{label}: Fehler — {e}")
                st.session_state.dl_results[key] = ("err", str(e), 0)

# ── Ergebnisse ─────────────────────────────────────────────────────────────────
if st.session_state.dl_results:
    import zipfile
    import io

    st.subheader("Dateien herunterladen")

    ok_results = [
        (key, val, size_mb)
        for key, (state, val, size_mb) in st.session_state.dl_results.items()
        if state == "ok" and os.path.exists(val)
    ]

    for key, val, size_mb in ok_results:
        with open(val, "rb") as f:
            data = f.read()
        mime = ("application/geopackage+sqlite3"
                if val.endswith(".gpkg") else "image/tiff")
        st.download_button(
            label=f"{LAYERS[key]['label']} speichern ({size_mb:.1f} MB)",
            data=data,
            file_name=os.path.basename(val),
            mime=mime,
            key=f"dl_{key}",
        )

    if len(ok_results) > 1:
        st.divider()
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_STORED) as zf:
            for key, val, _ in ok_results:
                zf.write(val, arcname=os.path.basename(val))
        zip_buf.seek(0)
        total_mb = sum(size_mb for _, _, size_mb in ok_results)
        st.download_button(
            label=f"Alle Dateien als ZIP herunterladen ({total_mb:.1f} MB)",
            data=zip_buf,
            file_name="geodaten.zip",
            mime="application/zip",
            key="dl_all_zip",
        )
