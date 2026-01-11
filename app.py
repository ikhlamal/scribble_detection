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
    # Pattern matching thresholds
    'pattern_threshold': 0.50,           # threshold untuk pattern similarity
    'min_scribble_area': 2000,           # minimum area stroke (filter noise)
    'min_stroke_length': 100,            # minimum length stroke
    
    # Complexity requirements (PENTING untuk filter garis sederhana)
    'min_direction_changes': 8,          # minimal perubahan arah (scribble = banyak belok)
    'min_density_ratio': 0.05,           # length/area ratio (scribble = padat)
    'max_linearity': 0.7,                # max linearitas (< 0.7 = tidak lurus)
    
    # Multi-reference matching
    'min_matching_refs': 2,              # minimal harus match dengan 2+ references
    'high_confidence_threshold': 0.65,   # threshold untuk high confidence match
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
            # Preprocessing: threshold untuk binary
            img_arr = np.array(img)
            _, img_arr = cv2.threshold(img_arr, 127, 255, cv2.THRESH_BINARY)
            img_resized = cv2.resize(img_arr, size)
            refs.append(img_resized)
    return refs


def analyze_stroke_complexity(points):
    """Analisis kompleksitas stroke untuk filter garis sederhana"""
    if len(points) < 3:
        return {
            'direction_changes': 0,
            'density_ratio': 0,
            'linearity': 1.0,
            'is_complex': False
        }
    
    # 1. Direction changes (berapa kali arah berubah signifikan)
    direction_changes = 0
    prev_angle = None
    
    for i in range(1, len(points) - 1):
        dx = points[i+1][0] - points[i][0]
        dy = points[i+1][1] - points[i][1]
        
        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
            angle = math.atan2(dy, dx)
            
            if prev_angle is not None:
                angle_diff = abs(angle - prev_angle)
                # Normalize to [0, pi]
                if angle_diff > math.pi:
                    angle_diff = 2 * math.pi - angle_diff
                
                # Count significant direction changes (> 30 degrees)
                if angle_diff > math.pi / 6:
                    direction_changes += 1
            
            prev_angle = angle
    
    # 2. Density ratio (panjang stroke dibanding area bbox)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    bbox_w = max(xs) - min(xs) + 1
    bbox_h = max(ys) - min(ys) + 1
    bbox_area = bbox_w * bbox_h
    
    length = 0
    for i in range(len(points) - 1):
        dx = points[i+1][0] - points[i][0]
        dy = points[i+1][1] - points[i][1]
        length += math.sqrt(dx*dx + dy*dy)
    
    density_ratio = length / (bbox_area + 1e-6)
    
    # 3. Linearity check (seberapa lurus garis ini)
    # Hitung jarak euclidean start-end vs total path length
    if len(points) >= 2:
        start = points[0]
        end = points[-1]
        euclidean_dist = math.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)
        linearity = euclidean_dist / (length + 1e-6)
    else:
        linearity = 1.0
    
    # Determine if complex enough to be scribble
    is_complex = (
        direction_changes >= CONFIG['min_direction_changes'] and
        density_ratio >= CONFIG['min_density_ratio'] and
        linearity < CONFIG['max_linearity']
    )
    
    return {
        'direction_changes': direction_changes,
        'density_ratio': density_ratio,
        'linearity': linearity,
        'is_complex': is_complex,
        'bbox_area': bbox_area,
        'length': length
    }


def get_stroke_bbox(points):
    """Get bounding box"""
    if len(points) < 2:
        return None
    
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, y1 = min(xs), min(ys)
    x2, y2 = max(xs), max(ys)
    
    return (x1, y1, x2, y2)


def match_with_references_advanced(canvas, bbox, refs):
    """Advanced matching dengan multiple references dan multiple methods"""
    if len(refs) == 0:
        return 0.0, 0
    
    x1, y1, x2, y2 = bbox
    margin = 60
    
    x1 = max(0, int(x1) - margin)
    y1 = max(0, int(y1) - margin)
    x2 = min(canvas.shape[1], int(x2) + margin)
    y2 = min(canvas.shape[0], int(y2) + margin)
    
    if x2 - x1 < 20 or y2 - y1 < 20:
        return 0.0, 0
    
    region = canvas[y1:y2, x1:x2]
    
    # Resize ke ukuran reference
    target_size = refs[0].shape
    try:
        region_resized = cv2.resize(region, (target_size[1], target_size[0]))
    except:
        return 0.0, 0
    
    # Threshold untuk binary
    _, region_bin = cv2.threshold(region_resized, 127, 255, cv2.THRESH_BINARY)
    
    # Apply edge detection untuk structural matching
    region_edges = cv2.Canny(region_bin, 50, 150)
    
    scores = []
    high_confidence_matches = 0
    
    for ref in refs:
        ref_edges = cv2.Canny(ref, 50, 150)
        
        # Method 1: SSIM on original
        try:
            score1 = ssim(region_bin, ref, data_range=255)
        except:
            score1 = 0
        
        # Method 2: SSIM on edges (better for structure)
        try:
            score2 = ssim(region_edges, ref_edges, data_range=255)
        except:
            score2 = 0
        
        # Method 3: Template matching
        try:
            result = cv2.matchTemplate(region_bin, ref, cv2.TM_CCOEFF_NORMED)
            _, score3, _, _ = cv2.minMaxLoc(result)
        except:
            score3 = 0
        
        # Method 4: Correlation
        try:
            result = cv2.matchTemplate(region_bin, ref, cv2.TM_CCORR_NORMED)
            _, score4, _, _ = cv2.minMaxLoc(result)
        except:
            score4 = 0
        
        # Method 5: Histogram comparison
        try:
            hist1 = cv2.calcHist([region_bin], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([ref], [0], None, [256], [0, 256])
            score5 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        except:
            score5 = 0
        
        # Combined score - weighted average with emphasis on structure
        combined = (
            score1 * 0.15 +  # SSIM original
            score2 * 0.35 +  # SSIM edges (most important for structure)
            score3 * 0.25 +  # Template matching
            score4 * 0.15 +  # Correlation
            score5 * 0.10    # Histogram
        )
        
        scores.append(combined)
        
        # Count high confidence matches
        if combined > CONFIG['high_confidence_threshold']:
            high_confidence_matches += 1
    
    # Return max score and number of matching references
    max_score = max(scores) if scores else 0.0
    matching_refs = sum(1 for s in scores if s > CONFIG['pattern_threshold'])
    
    return max_score, matching_refs


def detect_scribbles_for_actor(strokes_data, refs):
    """Detect scribbles dengan validation berlapis"""
    # Init canvas
    canvas = np.ones(CANVAS_SIZE[::-1], dtype=np.uint8) * 255
    
    results = []
    
    # Process each stroke
    for idx, row in strokes_data.iterrows():
        pts, _ = parse_stroke(row['description'])
        if len(pts) < 2:
            continue
        
        # Draw stroke
        pts_int = [(int(x), int(y)) for x, y in pts]
        cv2.polylines(canvas, [np.array(pts_int)], False, 0, STROKE_WIDTH)
        
        # Step 1: Analyze complexity (filter garis sederhana)
        complexity = analyze_stroke_complexity(pts)
        
        # Step 2: Size filters
        if complexity['bbox_area'] < CONFIG['min_scribble_area']:
            is_scribble = False
            pattern_score = 0.0
            matching_refs = 0
            reason = "Too small"
        elif complexity['length'] < CONFIG['min_stroke_length']:
            is_scribble = False
            pattern_score = 0.0
            matching_refs = 0
            reason = "Too short"
        elif not complexity['is_complex']:
            # Garis sederhana = bukan scribble
            is_scribble = False
            pattern_score = 0.0
            matching_refs = 0
            reason = f"Not complex (dir_changes={complexity['direction_changes']}, linearity={complexity['linearity']:.2f})"
        else:
            # Step 3: Pattern matching (hanya untuk stroke yang kompleks)
            bbox = get_stroke_bbox(pts)
            if bbox and len(refs) > 0:
                pattern_score, matching_refs = match_with_references_advanced(canvas, bbox, refs)
                
                # Decision: harus match dengan minimal N references DAN score tinggi
                is_scribble = (
                    pattern_score > CONFIG['pattern_threshold'] and
                    matching_refs >= CONFIG['min_matching_refs']
                )
                
                if is_scribble:
                    reason = f"Scribble detected (score={pattern_score:.2f}, matches={matching_refs})"
                else:
                    reason = f"Pattern not matching (score={pattern_score:.2f}, matches={matching_refs})"
            else:
                pattern_score = 0.0
                matching_refs = 0
                is_scribble = False
                reason = "No references"
        
        results.append({
            'uniqId': row['uniqId'],
            'timestamp': row['timestamp'],
            'is_scribble': is_scribble,
            'pattern_score': pattern_score,
            'matching_refs': matching_refs,
            'direction_changes': complexity['direction_changes'],
            'linearity': complexity['linearity'],
            'density_ratio': complexity['density_ratio'],
            'bbox_area': complexity['bbox_area'],
            'length': complexity['length'],
            'reason': reason,
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
            # Scribble = red with intensity based on pattern score
            confidence = result['pattern_score']
            red = int(200 + 55 * min(confidence, 1.0))
            color = (red, 0, 0)
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
    st.markdown("**Detection Method:** Advanced Pattern Matching + Complexity Analysis")
    st.markdown("---")

    # ======================================================
    # INPUT SECTION (OUTSIDE FORM FOR DYNAMIC BEHAVIOR)
    # ======================================================
    st.subheader("📁 Input Data")
    csv_file = st.file_uploader("Upload CSV File", type=["csv"])

    st.subheader("🎯 Processing Options")
    
    # Checkbox untuk limit actors (langsung muncul)
    limit_actors = st.checkbox("Batasi jumlah actor")

    # Number input muncul langsung setelah checkbox dicentang
    max_actors = None
    if limit_actors:
        max_actors = st.number_input(
            "Jumlah actor yang diproses",
            min_value=1,
            step=1,
            value=5
        )

    # Submit button (di luar form)
    submitted = st.button("🚀 Submit & Process", type="primary")

    # ======================================================
    # STOP JIKA BELUM SUBMIT
    # ======================================================
    if not submitted:
        st.info("⬆️ Upload CSV dan klik **Submit & Process** untuk mulai")
        return

    if csv_file is None:
        st.error("❌ CSV belum di-upload")
        return

    # ======================================================
    # LOAD CSV
    # ======================================================
    try:
        df = pd.read_csv(csv_file)
        st.success(f"✅ Loaded {len(df)} rows from CSV")
    except Exception as e:
        st.error(f"❌ Error loading CSV: {e}")
        return

    # ======================================================
    # FILTER ADD_HW_MEMO
    # ======================================================
    df_filtered = df[df['operation_name'] == 'ADD_HW_MEMO'].copy()

    if df_filtered.empty:
        st.warning("⚠️ Tidak ditemukan ADD_HW_MEMO pada CSV")
        return

    df_filtered['timestamp'] = pd.to_datetime(df_filtered['timestamp'])

    # ======================================================
    # ACTOR SELECTION
    # ======================================================
    actors = df_filtered['actor_name_id'].unique()

    if limit_actors and max_actors:
        actors = actors[:max_actors]

    total_actors = len(actors)
    st.markdown(f"### 👥 Actor diproses: **{total_actors}**")

    # ======================================================
    # LOAD REFERENCE SCRIBBLES
    # ======================================================
    ref_folder = "scribble_refs"
    refs = load_scribble_refs(ref_folder)

    if refs:
        st.success(f"✅ Loaded {len(refs)} reference scribble images")
    else:
        st.error("❌ Tidak ada reference scribble image - Detection tidak bisa berjalan!")
        st.info("💡 Tambahkan reference scribble images di folder 'scribble_refs'")
        return

    # ======================================================
    # PROCESS EACH ACTOR
    # ======================================================
    actor_data = {}

    progress_bar = st.progress(0.0)
    status_text = st.empty()
    time_text = st.empty()

    import time
    start_time = time.time()

    for idx, actor in enumerate(actors):
        progress_bar.progress((idx + 1) / total_actors)

        elapsed = time.time() - start_time
        status_text.markdown(
            f"🔍 Processing **{actor}** ({idx+1}/{total_actors})"
        )
        time_text.markdown(f"⏱️ Elapsed: {int(elapsed)}s")

        actor_df = (
            df_filtered[df_filtered['actor_name_id'] == actor]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        results, canvas = detect_scribbles_for_actor(actor_df, refs)
        img_clean, img_annotated = render_images(actor_df, results)

        actor_data[actor] = {
            "df": actor_df,
            "results": results,
            "img_clean": img_clean,
            "img_annotated": img_annotated
        }

    progress_bar.empty()
    status_text.empty()
    time_text.empty()

    st.success("✅ Processing selesai")

    # ======================================================
    # GANTT CHART (DENGAN LAYOUT SEJAJAR)
    # ======================================================
    st.markdown("---")
    st.header("📊 Gantt Chart - Stroke Timeline")

    gantt_data = []

    for actor in actors:
        data = actor_data[actor]
        actor_df = data['df']
        results = data['results']

        actor_all_df = (
            df[df['actor_name_id'] == actor]
            .sort_values('timestamp')
            .reset_index(drop=True)
        )

        for i, (result, row) in enumerate(zip(results, actor_df.itertuples())):
            start_time = row.timestamp

            idx_all = actor_all_df[
                actor_all_df['uniqId'] == row.uniqId
            ].index[0]

            if idx_all + 1 < len(actor_all_df):
                finish_time = actor_all_df.iloc[idx_all + 1]['timestamp']
            else:
                finish_time = start_time + timedelta(seconds=1)

            gantt_data.append({
                'Actor': actor,
                'Stroke': f"Stroke {i+1}",
                'Start': start_time,
                'Finish': finish_time,
                'Type': 'Scribble' if result['is_scribble'] else 'Writing',
                'UniqId': row.uniqId,
                'PatternScore': result['pattern_score'],
                'MatchingRefs': result['matching_refs'],
                'DirectionChanges': result['direction_changes'],
                'Linearity': result['linearity'],
                'Reason': result['reason']
            })

    gantt_df = pd.DataFrame(gantt_data)

    # Buat custom Gantt chart dengan layout sejajar menggunakan Plotly timeline
    fig = px.timeline(
        gantt_df,
        x_start='Start',
        x_end='Finish',
        y='Actor',
        color='Type',
        color_discrete_map={
            'Writing': 'black',
            'Scribble': 'red'
        },
        hover_data=['Stroke', 'UniqId', 'PatternScore', 'MatchingRefs', 'DirectionChanges', 'Linearity', 'Reason'],
        title='Stroke Activity Timeline by Actor'
    )
    
    # Update layout untuk memperbesar bar dan mengatur spacing
    fig.update_yaxes(categoryorder='category ascending')
    fig.update_layout(
        height=max(400, total_actors * 80),
        xaxis_title="Time",
        yaxis_title="Actor",
        hovermode='closest',
        bargap=0.3,
        bargroupgap=0.1
    )
    
    # Perbesar ukuran bar
    fig.update_traces(
        marker=dict(
            line=dict(color='white', width=1)
        )
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
    
    avg_pattern_score_scribbles = gantt_df[gantt_df['Type'] == 'Scribble']['PatternScore'].mean() if total_scribbles > 0 else 0
    avg_matching_refs = gantt_df[gantt_df['Type'] == 'Scribble']['MatchingRefs'].mean() if total_scribbles > 0 else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Strokes", total_strokes)
    c2.metric("Scribbles", total_scribbles)
    c3.metric("Writing", total_writing)
    c4.metric("Avg Pattern Score", f"{avg_pattern_score_scribbles:.3f}")
    c5.metric("Avg Matching Refs", f"{avg_matching_refs:.1f}")

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
                        ['Stroke', 'Type', 'UniqId', 'PatternScore', 'MatchingRefs', 'DirectionChanges', 'Linearity', 'Reason']
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
