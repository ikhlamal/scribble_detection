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
    page_title="Sequential Scribble Detection Dashboard",
    page_icon="✍️",
    layout="wide"
)

# =========================
# DETECTION CONFIG
# =========================
CANVAS_SIZE = (900, 1100)
STROKE_WIDTH = 3

CONFIG = {
    # Pattern matching threshold
    'pattern_threshold': 0.55,           # threshold untuk classify sebagai scribble
    'min_scribble_area': 2000,           # minimum area untuk dianggap scribble
    'scan_grid_size': 150,               # ukuran grid untuk scanning canvas
    'scan_overlap': 50,                  # overlap antar grid untuk deteksi yang lebih baik
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


def match_region_with_references(region, refs):
    """Match region dengan references menggunakan multiple metrics"""
    if len(refs) == 0 or region.size == 0:
        return 0.0
    
    # Resize region ke ukuran reference
    target_size = refs[0].shape
    try:
        region_resized = cv2.resize(region, (target_size[1], target_size[0]))
    except:
        return 0.0
    
    # Threshold untuk binary
    _, region_bin = cv2.threshold(region_resized, 127, 255, cv2.THRESH_BINARY)
    
    max_score = 0.0
    for ref in refs:
        # SSIM (Structural Similarity)
        try:
            score1 = ssim(region_bin, ref, data_range=255)
        except:
            score1 = 0
        
        # Template matching - CCOEFF
        try:
            result = cv2.matchTemplate(region_bin, ref, cv2.TM_CCOEFF_NORMED)
            _, score2, _, _ = cv2.minMaxLoc(result)
        except:
            score2 = 0
        
        # Template matching - CCORR
        try:
            result = cv2.matchTemplate(region_bin, ref, cv2.TM_CCORR_NORMED)
            _, score3, _, _ = cv2.minMaxLoc(result)
        except:
            score3 = 0
        
        # Combined - ambil yang tertinggi
        combined = max(score1, score2, score3)
        max_score = max(max_score, combined)
    
    return max_score


def scan_canvas_for_scribbles(canvas, refs, grid_size=150, overlap=50):
    """
    Scan SELURUH canvas menggunakan sliding window untuk detect scribble patterns.
    Returns: list of detected scribble regions dengan scores dan locations.
    """
    if len(refs) == 0:
        return []
    
    height, width = canvas.shape
    step = grid_size - overlap
    
    detected_regions = []
    
    # Sliding window scan
    for y in range(0, height - grid_size + 1, step):
        for x in range(0, width - grid_size + 1, step):
            # Extract region
            region = canvas[y:y+grid_size, x:x+grid_size]
            
            # Skip mostly white regions (no ink)
            non_white_pixels = np.sum(region < 250)
            if non_white_pixels < 100:  # threshold minimum ink
                continue
            
            # Match dengan references
            score = match_region_with_references(region, refs)
            
            if score > CONFIG['pattern_threshold']:
                detected_regions.append({
                    'bbox': (x, y, x+grid_size, y+grid_size),
                    'score': score,
                    'center': (x + grid_size//2, y + grid_size//2)
                })
    
    return detected_regions


def regions_overlap(region1, region2):
    """Check if two regions overlap"""
    x1_1, y1_1, x2_1, y2_1 = region1['bbox']
    x1_2, y1_2, x2_2, y2_2 = region2['bbox']
    
    # Check if rectangles overlap
    if x1_1 >= x2_2 or x1_2 >= x2_1:
        return False
    if y1_1 >= y2_2 or y1_2 >= y2_1:
        return False
    return True


def find_new_scribble_regions(regions_before, regions_after):
    """
    Compare regions before and after to find NEW scribbles.
    Returns list of new regions that appeared.
    """
    new_regions = []
    
    for region_after in regions_after:
        # Check if this region existed before
        is_new = True
        for region_before in regions_before:
            if regions_overlap(region_after, region_before):
                # Region already existed (or overlaps significantly)
                is_new = False
                break
        
        if is_new:
            new_regions.append(region_after)
    
    return new_regions


def point_in_region(point, region):
    """Check if a point is inside a region"""
    x, y = point
    x1, y1, x2, y2 = region['bbox']
    return x1 <= x <= x2 and y1 <= y <= y2


def stroke_intersects_region(points, region):
    """Check if stroke intersects with region"""
    # Check if any point of stroke is in region
    for point in points:
        if point_in_region(point, region):
            return True
    return False


def detect_scribbles_sequential(strokes_data, refs):
    """
    SEQUENTIAL DETECTION: Detect scribbles dengan menambahkan stroke satu per satu
    dan scanning canvas setelah setiap penambahan.
    """
    # Init canvas kosong
    canvas = np.ones(CANVAS_SIZE[::-1], dtype=np.uint8) * 255
    
    # Track scribble regions yang sudah terdeteksi
    detected_regions = []
    
    results = []
    
    # Process each stroke SEQUENTIALLY
    for idx, row in strokes_data.iterrows():
        pts, _ = parse_stroke(row['description'])
        if len(pts) < 2:
            results.append({
                'uniqId': row['uniqId'],
                'timestamp': row['timestamp'],
                'is_scribble': False,
                'pattern_score': 0.0,
                'triggered_new_scribble': False,
                'points': pts
            })
            continue
        
        # === STEP 1: Scan canvas BEFORE adding stroke ===
        regions_before = scan_canvas_for_scribbles(
            canvas, 
            refs, 
            CONFIG['scan_grid_size'], 
            CONFIG['scan_overlap']
        )
        
        # === STEP 2: Add stroke to canvas ===
        pts_int = [(int(x), int(y)) for x, y in pts]
        cv2.polylines(canvas, [np.array(pts_int)], False, 0, STROKE_WIDTH)
        
        # === STEP 3: Scan canvas AFTER adding stroke ===
        regions_after = scan_canvas_for_scribbles(
            canvas, 
            refs, 
            CONFIG['scan_grid_size'], 
            CONFIG['scan_overlap']
        )
        
        # === STEP 4: Find NEW scribble regions ===
        new_regions = find_new_scribble_regions(regions_before, regions_after)
        
        # === STEP 5: Check if THIS stroke triggered new scribble ===
        is_scribble = False
        max_score = 0.0
        triggered_regions = []
        
        if len(new_regions) > 0:
            # Check if stroke intersects with any new scribble region
            for new_region in new_regions:
                if stroke_intersects_region(pts, new_region):
                    is_scribble = True
                    max_score = max(max_score, new_region['score'])
                    triggered_regions.append(new_region)
        
        # Update detected regions
        detected_regions.extend(triggered_regions)
        
        results.append({
            'uniqId': row['uniqId'],
            'timestamp': row['timestamp'],
            'is_scribble': is_scribble,
            'pattern_score': max_score,
            'triggered_new_scribble': is_scribble,
            'new_regions_count': len(triggered_regions),
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
    st.title("✍️ Sequential Scribble Detection Dashboard")
    st.markdown("**Detection Method:** Incremental Canvas Scanning dengan Pattern Matching")
    st.markdown("✨ **New:** Deteksi stroke yang menyebabkan munculnya pola scribble baru secara real-time")
    st.markdown("---")

    # ======================================================
    # INPUT SECTION
    # ======================================================
    st.subheader("📁 Input Data")
    csv_file = st.file_uploader("Upload CSV File", type=["csv"])

    st.subheader("🎯 Processing Options")
    
    # Checkbox untuk limit actors
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

    # Configuration options
    with st.expander("⚙️ Advanced Configuration"):
        CONFIG['pattern_threshold'] = st.slider(
            "Pattern Matching Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.55,
            step=0.05,
            help="Threshold untuk menganggap pattern sebagai scribble"
        )
        CONFIG['scan_grid_size'] = st.slider(
            "Scan Grid Size",
            min_value=50,
            max_value=300,
            value=150,
            step=10,
            help="Ukuran window untuk scanning canvas"
        )
        CONFIG['scan_overlap'] = st.slider(
            "Scan Overlap",
            min_value=0,
            max_value=100,
            value=50,
            step=10,
            help="Overlap antar grid untuk deteksi yang lebih baik"
        )

    # Submit button
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

        # SEQUENTIAL DETECTION
        results, canvas = detect_scribbles_sequential(actor_df, refs)
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
    # GANTT CHART
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
                'TriggeredNewScribble': result['triggered_new_scribble']
            })

    gantt_df = pd.DataFrame(gantt_data)

    # Create Gantt chart
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
        hover_data=['Stroke', 'UniqId', 'PatternScore', 'TriggeredNewScribble'],
        title='Stroke Activity Timeline by Actor (Sequential Detection)'
    )
    
    fig.update_yaxes(categoryorder='category ascending')
    fig.update_layout(
        height=max(400, total_actors * 80),
        xaxis_title="Time",
        yaxis_title="Actor",
        hovermode='closest',
        bargap=0.3,
        bargroupgap=0.1
    )
    
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

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Strokes", total_strokes)
    c2.metric("Scribbles Detected", total_scribbles)
    c3.metric("Writing", total_writing)
    c4.metric("Avg Pattern Score (Scribbles)", f"{avg_pattern_score_scribbles:.3f}")

    if total_strokes > 0:
        scribble_rate = (total_scribbles / total_strokes) * 100
        st.markdown(f"**Scribble Rate:** {scribble_rate:.2f}%")

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
                        ['Stroke', 'Type', 'UniqId', 'PatternScore', 'TriggeredNewScribble', 'Start', 'Finish']
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
