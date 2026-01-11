import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity as ssim
import io
import base64

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Scribble Detection Dashboard",
    page_icon="✍️",
    layout="wide"
)

# =========================
# DETECTION CONFIG
# =========================
CANVAS_SIZE = (900, 1100)
STROKE_WIDTH = 3

CONFIG = {
    'repetitive_area_threshold': 4,
    'dense_crossing_threshold': 3,
    'min_scribble_area': 1200,
    'min_stroke_length': 80,
    'sharp_turn_angle': 100,
    'min_sharp_turns': 4,
    'density_in_bbox': 0.08,
    'use_pattern_matching': True,
    'pattern_threshold': 0.45,
    'pattern_weight': 0.50,
    'use_heatmap': True,
    'heatmap_intensity_threshold': 30,
    'scribble_score_threshold': 0.55,
}

# =========================
# UTILITY FUNCTIONS
# =========================
def parse_stroke(stroke_str):
    """Parse stroke dari format string"""
    try:
        header, *pts = stroke_str.strip().split(",")
        points = []
        times = []
        
        for p in pts:
            parts = p.split(":")
            if len(parts) == 3 and all(part.strip() for part in parts):
                t, x, y = parts
                points.append((float(x), float(y)))
                times.append(float(t))
        
        return points, times
    except:
        return [], []


def load_scribble_refs(folder, size=(150, 150)):
    """Load reference scribbles"""
    refs = []
    if not os.path.exists(folder):
        return refs
        
    for fn in os.listdir(folder):
        if fn.lower().endswith((".png", ".jpg", ".jpeg")):
            img = Image.open(os.path.join(folder, fn)).convert("L")
            img_arr = np.array(img)
            _, img_arr = cv2.threshold(img_arr, 127, 255, cv2.THRESH_BINARY)
            img_resized = cv2.resize(img_arr, size)
            refs.append(img_resized)
    return refs


def create_heatmap(canvas, cell_size=20):
    """Buat heatmap untuk deteksi overlap"""
    h, w = canvas.shape
    heatmap = np.zeros((h // cell_size + 1, w // cell_size + 1), dtype=np.int32)
    
    for i in range(0, h, cell_size):
        for j in range(0, w, cell_size):
            cell = canvas[i:min(i+cell_size, h), j:min(j+cell_size, w)]
            black_pixels = np.sum(cell < 200)
            heatmap[i // cell_size, j // cell_size] = black_pixels
    
    return heatmap


def check_heatmap_intensity(heatmap, bbox, cell_size=20):
    """Cek intensitas heatmap di area bbox"""
    x1, y1, x2, y2 = bbox
    
    i1 = int(y1) // cell_size
    j1 = int(x1) // cell_size
    i2 = int(y2) // cell_size
    j2 = int(x2) // cell_size
    
    i1 = max(0, i1)
    j1 = max(0, j1)
    i2 = min(heatmap.shape[0] - 1, i2)
    j2 = min(heatmap.shape[1] - 1, j2)
    
    if i2 <= i1 or j2 <= j1:
        return 0
    
    region = heatmap[i1:i2+1, j1:j2+1]
    return np.max(region)


def analyze_stroke_simple(points):
    """Analisis geometri stroke"""
    if len(points) < 3:
        return None
    
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, y1 = min(xs), min(ys)
    x2, y2 = max(xs), max(ys)
    bbox_w = x2 - x1 + 1
    bbox_h = y2 - y1 + 1
    bbox_area = bbox_w * bbox_h
    
    length = 0
    for i in range(len(points) - 1):
        dx = points[i+1][0] - points[i][0]
        dy = points[i+1][1] - points[i][1]
        length += math.sqrt(dx*dx + dy*dy)
    
    sharp_turns = 0
    for i in range(1, len(points) - 1):
        v1 = np.array([points[i][0] - points[i-1][0], points[i][1] - points[i-1][1]])
        v2 = np.array([points[i+1][0] - points[i][0], points[i+1][1] - points[i][1]])
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 > 1e-6 and norm2 > 1e-6:
            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.degrees(np.arccos(cos_angle))
            
            if angle > CONFIG['sharp_turn_angle']:
                sharp_turns += 1
    
    density = length / (bbox_area + 1e-6)
    
    crossings = 0
    if len(points) > 10:
        step = max(1, len(points) // 20)
        for i in range(0, len(points) - step, step):
            p1 = points[i]
            for j in range(i + step * 2, len(points) - step, step):
                p2 = points[j]
                dist = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                if dist < 15:
                    crossings += 1
    
    return {
        'bbox': (x1, y1, x2, y2),
        'bbox_area': bbox_area,
        'length': length,
        'density': density,
        'sharp_turns': sharp_turns,
        'crossings': crossings,
        'points': points
    }


def match_with_references(canvas, bbox, refs):
    """Match dengan reference scribbles"""
    if len(refs) == 0:
        return 0.0
    
    x1, y1, x2, y2 = bbox
    margin = 60
    
    x1 = max(0, int(x1) - margin)
    y1 = max(0, int(y1) - margin)
    x2 = min(canvas.shape[1], int(x2) + margin)
    y2 = min(canvas.shape[0], int(y2) + margin)
    
    if x2 - x1 < 20 or y2 - y1 < 20:
        return 0.0
    
    region = canvas[y1:y2, x1:x2]
    
    target_size = refs[0].shape
    try:
        region_resized = cv2.resize(region, (target_size[1], target_size[0]))
    except:
        return 0.0
    
    _, region_bin = cv2.threshold(region_resized, 127, 255, cv2.THRESH_BINARY)
    
    max_score = 0.0
    for ref in refs:
        try:
            score1 = ssim(region_bin, ref, data_range=255)
            result = cv2.matchTemplate(region_bin, ref, cv2.TM_CCOEFF_NORMED)
            _, score2, _, _ = cv2.minMaxLoc(result)
            combined = max(score1, score2)
            max_score = max(max_score, combined)
        except:
            continue
    
    return max_score


def calculate_scribble_score(geom, heatmap_intensity, pattern_score):
    """Calculate scribble score"""
    if geom['bbox_area'] < CONFIG['min_scribble_area']:
        return 0.0
    if geom['length'] < CONFIG['min_stroke_length']:
        return 0.0
    
    score = 0.0
    
    if CONFIG['use_pattern_matching'] and pattern_score > CONFIG['pattern_threshold']:
        score += CONFIG['pattern_weight']
        if pattern_score > 0.7:
            score += 0.20
    
    if geom['sharp_turns'] >= CONFIG['min_sharp_turns']:
        score += 0.20
    
    if geom['density'] > CONFIG['density_in_bbox']:
        score += 0.15
    
    if geom['crossings'] >= CONFIG['dense_crossing_threshold']:
        score += 0.25
    
    if CONFIG['use_heatmap'] and heatmap_intensity > CONFIG['heatmap_intensity_threshold']:
        score += 0.15
    
    return min(score, 1.0)


def detect_scribbles_for_actor(strokes_data, refs):
    """Detect scribbles untuk satu actor"""
    canvas = np.ones(CANVAS_SIZE[::-1], dtype=np.uint8) * 255
    heatmap = np.zeros((CANVAS_SIZE[1] // 20 + 1, CANVAS_SIZE[0] // 20 + 1), dtype=np.int32)
    
    results = []
    
    for idx, row in strokes_data.iterrows():
        pts, _ = parse_stroke(row['description'])
        if len(pts) < 2:
            continue
        
        pts_int = [(int(x), int(y)) for x, y in pts]
        cv2.polylines(canvas, [np.array(pts_int)], False, 0, STROKE_WIDTH)
        
        if CONFIG['use_heatmap']:
            heatmap = create_heatmap(canvas, cell_size=20)
        
        geom = analyze_stroke_simple(pts)
        if geom is None:
            continue
        
        pattern_score = 0.0
        if CONFIG['use_pattern_matching'] and len(refs) > 0:
            pattern_score = match_with_references(canvas, geom['bbox'], refs)
        
        heatmap_intensity = 0
        if CONFIG['use_heatmap']:
            heatmap_intensity = check_heatmap_intensity(heatmap, geom['bbox'], cell_size=20)
        
        scribble_score = calculate_scribble_score(geom, heatmap_intensity, pattern_score)
        is_scribble = scribble_score > CONFIG['scribble_score_threshold']
        
        results.append({
            'uniqId': row['uniqId'],
            'timestamp': row['timestamp'],
            'is_scribble': is_scribble,
            'score': scribble_score,
            'points': pts
        })
    
    return results, canvas


def render_images(strokes_data, scribble_results):
    """Render clean dan annotated images"""
    # Clean version
    img_clean = Image.new("RGB", CANVAS_SIZE, (255, 255, 255))
    draw_clean = ImageDraw.Draw(img_clean)
    
    # Annotated version
    img_annotated = Image.new("RGB", CANVAS_SIZE, (255, 255, 255))
    draw_annotated = ImageDraw.Draw(img_annotated)
    
    for idx, (result, row) in enumerate(zip(scribble_results, strokes_data.itertuples())):
        pts = result['points']
        if len(pts) < 2:
            continue
        
        # Color
        if result['is_scribble']:
            color = (220, 0, 0)  # Red for scribble
            text_color = (0, 200, 0)  # Green text
        else:
            color = (0, 0, 0)  # Black for writing
            text_color = (255, 200, 0)  # Yellow text
        
        # Draw on both
        draw_clean.line(pts, fill=color, width=STROKE_WIDTH)
        draw_annotated.line(pts, fill=color, width=STROKE_WIDTH)
        
        # Add annotation
        if len(pts) > 1:
            mid_idx = len(pts) // 2
            x, y = pts[mid_idx]
            stroke_num = str(idx + 1)
            text_offset_x = len(stroke_num) * 3
            text_offset_y = 5
            
            draw_annotated.text(
                (x - text_offset_x, y - text_offset_y),
                stroke_num,
                fill=text_color,
                stroke_width=1,
                stroke_fill=(0, 0, 0)
            )
    
    return img_clean, img_annotated


# =========================
# MAIN APP
# =========================
def main():
    st.title("✍️ Scribble Detection Dashboard")
    st.markdown("---")

    # ======================================================
    # INPUT SECTION (DYNAMIC, NO SIDEBAR)
    # ======================================================
    st.subheader("📁 Input Data")

    csv_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    st.subheader("🎯 Processing Options")

    limit_actors = st.checkbox("Batasi jumlah actor")

    max_actors = None
    if limit_actors:
        max_actors = st.number_input(
            "Jumlah actor yang diproses",
            min_value=1,
            step=1,
            value=5
        )

    submitted = st.button("🚀 Submit & Process")

    if not submitted:
        st.info("⬆️ Atur parameter lalu klik **Submit & Process**")
        return

    if csv_file is None:
        st.error("❌ CSV belum di-upload")
        return

    # ======================================================
    # LOAD CSV (FIX TIMESTAMP SEKALI DI SINI)
    # ======================================================
    try:
        df = pd.read_csv(csv_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        st.success(f"✅ Loaded {len(df)} rows")
    except Exception as e:
        st.error(f"❌ Gagal load CSV: {e}")
        return

    # ======================================================
    # FILTER DATA
    # ======================================================
    df_filtered = df[df['operation_name'] == 'ADD_HW_MEMO'].copy()

    if df_filtered.empty:
        st.warning("⚠️ Tidak ada ADD_HW_MEMO")
        return

    # ======================================================
    # ACTOR SELECTION
    # ======================================================
    actors = df_filtered['actor_name_id'].unique()

    if limit_actors:
        actors = actors[:max_actors]

    total_actors = len(actors)
    st.markdown(f"### 👥 Actor diproses: **{total_actors}**")

    # ======================================================
    # LOAD REFERENCE SCRIBBLES
    # ======================================================
    refs = load_scribble_refs("scribble_refs")

    if refs:
        st.success(f"✅ Loaded {len(refs)} reference scribble images")
    else:
        st.warning("⚠️ Reference scribble tidak ditemukan")

    # ======================================================
    # PROCESSING
    # ======================================================
    actor_data = {}

    progress_bar = st.progress(0.0)
    status_text = st.empty()

    import time
    start_time = time.time()

    for idx, actor in enumerate(actors):
        progress_bar.progress((idx + 1) / total_actors)
        status_text.markdown(
            f"🔍 Processing **{actor}** ({idx+1}/{total_actors})"
        )

        actor_df = (
            df_filtered[df_filtered['actor_name_id'] == actor]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        results, _ = detect_scribbles_for_actor(actor_df, refs)
        img_clean, img_annotated = render_images(actor_df, results)

        actor_data[actor] = {
            "df": actor_df,
            "results": results,
            "img_clean": img_clean,
            "img_annotated": img_annotated
        }

    progress_bar.empty()
    status_text.empty()
    st.success("✅ Processing selesai")

    # ======================================================
    # CUSTOM GANTT CHART (SEJAJAR & RAPI)
    # ======================================================
    st.markdown("---")
    st.header("📊 Stroke Timeline (Custom Gantt)")

    fig = go.Figure()

    y_positions = {actor: i * 2 for i, actor in enumerate(actors)}
    offset = 0.35

    for actor in actors:
        base_y = y_positions[actor]

        actor_df = actor_data[actor]['df']
        results = actor_data[actor]['results']

        actor_all_df = (
            df[df['actor_name_id'] == actor]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        for i, (res, row) in enumerate(zip(results, actor_df.itertuples())):
            start = row.timestamp

            idx_all = actor_all_df[
                actor_all_df['uniqId'] == row.uniqId
            ].index[0]

            if idx_all + 1 < len(actor_all_df):
                finish = actor_all_df.iloc[idx_all + 1]['timestamp']
            else:
                finish = start + timedelta(seconds=1)

            duration = (finish - start).total_seconds()

            y = base_y + (offset if res['is_scribble'] else -offset)

            fig.add_trace(go.Bar(
                x=[duration],
                y=[y],
                base=start,
                orientation='h',
                marker=dict(
                    color='red' if res['is_scribble'] else 'black'
                ),
                showlegend=False,
                hovertext=(
                    f"Actor: {actor}<br>"
                    f"Stroke {i+1}<br>"
                    f"{'Scribble' if res['is_scribble'] else 'Writing'}<br>"
                    f"Score: {res['score']:.2f}"
                )
            ))

    fig.update_layout(
        height=max(600, total_actors * 160),
        bargap=0.25,
        xaxis_title="Time",
        yaxis=dict(
            tickvals=list(y_positions.values()),
            ticktext=list(y_positions.keys()),
            title="Actor"
        ),
        title="Stroke Timeline per Actor (Writing vs Scribble)",
        template="simple_white"
    )

    st.plotly_chart(fig, use_container_width=True)


    # ======================================================
    # STATISTICS
    # ======================================================
    st.markdown("---")
    st.header("📈 Statistics")

    total_strokes = len(gantt_df)
    total_scribbles = len(gantt_df[gantt_df['Type'] == 'Scribble'])
    total_writing = len(gantt_df[gantt_df['Type'] == 'Writing'])

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Strokes", total_strokes)
    c2.metric("Scribbles", total_scribbles)
    c3.metric("Writing", total_writing)

    # ======================================================
    # IMAGES PER ACTOR
    # ======================================================
    st.markdown("---")
    st.header("🖼️ Generated Images per Actor")

    for actor in actors:
        with st.expander(f"👤 {actor}", expanded=False):
            data = actor_data[actor]

            tab1, tab2, tab3 = st.tabs([
                "📊 Summary",
                "🖼️ Clean Image",
                "🏷️ Annotated Image"
            ])

            with tab1:
                actor_strokes = gantt_df[gantt_df['Actor'] == actor]
                scribbles = len(actor_strokes[actor_strokes['Type'] == 'Scribble'])
                writing = len(actor_strokes[actor_strokes['Type'] == 'Writing'])

                c1, c2, c3 = st.columns(3)
                c1.metric("Total Strokes", len(actor_strokes))
                c2.metric("Scribbles", scribbles)
                c3.metric("Writing", writing)

                st.dataframe(
                    actor_strokes[
                        ['Stroke', 'Type', 'UniqId', 'Score', 'Start', 'Finish']
                    ],
                    use_container_width=True
                )

            with tab2:
                st.image(
                    data['img_clean'],
                    caption=f"Clean Image - {actor}",
                    use_container_width=True
                )

            with tab3:
                st.image(
                    data['img_annotated'],
                    caption=f"Annotated Image - {actor}",
                    use_container_width=True
                )

if __name__ == "__main__":
    main()
