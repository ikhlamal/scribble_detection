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
from scipy import ndimage
import io
import base64

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Hybrid Scribble Detection Dashboard",
    page_icon="✍️",
    layout="wide"
)

# =========================
# DETECTION CONFIG
# =========================
CANVAS_SIZE = (900, 1100)
STROKE_WIDTH = 3

CONFIG = {
    # Hybrid detection parameters
    'density_threshold': 0.15,           # minimum ink density untuk area scribble
    'pattern_threshold': 0.45,           # threshold pattern matching (lebih rendah)
    'chaos_threshold': 3.5,              # threshold untuk chaos/irregularity
    'min_strokes_for_scribble': 3,       # minimum strokes yang overlap untuk dianggap scribble
    
    # Scanning parameters
    'scan_grid_size': 120,               # ukuran grid untuk scanning
    'scan_overlap': 40,                  # overlap antar grid
    'min_region_ink': 500,               # minimum pixels hitam di region
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


def calculate_ink_density(region):
    """Calculate ink density in a region (ratio of black pixels)"""
    if region.size == 0:
        return 0.0
    black_pixels = np.sum(region < 128)
    total_pixels = region.size
    return black_pixels / total_pixels


def calculate_chaos_metric(region):
    """
    Calculate chaos/irregularity metric for a region.
    Higher value = more chaotic/scribble-like
    Uses edge detection and direction variance.
    """
    if region.size == 0:
        return 0.0
    
    # Edge detection
    edges = cv2.Canny(region, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Sobel for direction analysis
    sobelx = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate gradient directions
    directions = np.arctan2(sobely, sobelx)
    
    # Variance of directions (higher = more chaotic)
    direction_variance = np.var(directions[edges > 0]) if np.sum(edges > 0) > 0 else 0
    
    # Combined chaos metric
    chaos = edge_density * 10 + direction_variance
    
    return chaos


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
        scores = []
        
        # SSIM
        try:
            score = ssim(region_bin, ref, data_range=255)
            scores.append(score)
        except:
            pass
        
        # Template matching - CCOEFF
        try:
            result = cv2.matchTemplate(region_bin, ref, cv2.TM_CCOEFF_NORMED)
            _, score, _, _ = cv2.minMaxLoc(result)
            scores.append(score)
        except:
            pass
        
        # Template matching - CCORR
        try:
            result = cv2.matchTemplate(region_bin, ref, cv2.TM_CCORR_NORMED)
            _, score, _, _ = cv2.minMaxLoc(result)
            scores.append(score)
        except:
            pass
        
        if scores:
            max_score = max(max_score, max(scores))
    
    return max_score


def detect_scribble_regions(canvas, refs):
    """
    Detect scribble regions using HYBRID approach:
    1. High ink density
    2. Pattern matching with references
    3. Chaos/irregularity metric
    """
    height, width = canvas.shape
    grid_size = CONFIG['scan_grid_size']
    overlap = CONFIG['scan_overlap']
    step = grid_size - overlap
    
    detected_regions = []
    
    # Sliding window scan
    for y in range(0, height - grid_size + 1, step):
        for x in range(0, width - grid_size + 1, step):
            # Extract region
            region = canvas[y:y+grid_size, x:x+grid_size]
            
            # Calculate metrics
            ink_density = calculate_ink_density(region)
            
            # Skip mostly white regions
            if ink_density < 0.05:  # very low ink
                continue
            
            # Calculate pattern matching score
            pattern_score = match_region_with_references(region, refs)
            
            # Calculate chaos metric
            chaos_score = calculate_chaos_metric(region)
            
            # HYBRID DECISION:
            # Region is scribble if:
            # - High density AND (high pattern match OR high chaos)
            is_scribble = False
            confidence = 0.0
            
            if ink_density > CONFIG['density_threshold']:
                if pattern_score > CONFIG['pattern_threshold']:
                    is_scribble = True
                    confidence = pattern_score * 0.7 + ink_density * 0.3
                elif chaos_score > CONFIG['chaos_threshold']:
                    is_scribble = True
                    confidence = (chaos_score / 10.0) * 0.6 + ink_density * 0.4
            
            if is_scribble:
                detected_regions.append({
                    'bbox': (x, y, x+grid_size, y+grid_size),
                    'center': (x + grid_size//2, y + grid_size//2),
                    'confidence': confidence,
                    'pattern_score': pattern_score,
                    'chaos_score': chaos_score,
                    'ink_density': ink_density
                })
    
    return detected_regions


def regions_overlap(region1, region2, threshold=0.3):
    """Check if two regions overlap significantly"""
    x1_1, y1_1, x2_1, y2_1 = region1['bbox']
    x1_2, y1_2, x2_2, y2_2 = region2['bbox']
    
    # Calculate intersection
    x_overlap = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
    y_overlap = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
    intersection = x_overlap * y_overlap
    
    # Calculate areas
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Overlap ratio
    overlap_ratio = intersection / min(area1, area2) if min(area1, area2) > 0 else 0
    
    return overlap_ratio > threshold


def find_new_scribble_regions(regions_before, regions_after):
    """Find NEW scribble regions that appeared"""
    new_regions = []
    
    for region_after in regions_after:
        is_new = True
        for region_before in regions_before:
            if regions_overlap(region_after, region_before):
                is_new = False
                break
        
        if is_new:
            new_regions.append(region_after)
    
    return new_regions


def stroke_intersects_region(points, region, margin=20):
    """Check if stroke intersects with region (with margin)"""
    x1, y1, x2, y2 = region['bbox']
    x1 -= margin
    y1 -= margin
    x2 += margin
    y2 += margin
    
    for x, y in points:
        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
    return False


def create_stroke_mask(canvas_shape, points, width=STROKE_WIDTH):
    """Create a mask for a single stroke"""
    mask = np.zeros(canvas_shape, dtype=np.uint8)
    pts_int = [(int(x), int(y)) for x, y in points]
    cv2.polylines(mask, [np.array(pts_int)], False, 255, width)
    return mask


def count_overlapping_strokes(canvas, new_stroke_mask, region):
    """Count how many strokes are overlapping in the region"""
    x1, y1, x2, y2 = region['bbox']
    region_canvas = canvas[y1:y2, x1:x2]
    region_stroke = new_stroke_mask[y1:y2, x1:x2]
    
    # Area where both have ink
    overlap = np.logical_and(region_canvas < 250, region_stroke > 0)
    overlap_pixels = np.sum(overlap)
    
    return overlap_pixels


def detect_scribbles_sequential(strokes_data, refs):
    """
    SEQUENTIAL DETECTION dengan HYBRID approach:
    - Density-based detection
    - Pattern matching dengan references
    - Chaos/irregularity analysis
    """
    # Init canvas kosong
    canvas = np.ones(CANVAS_SIZE[::-1], dtype=np.uint8) * 255
    
    results = []
    
    # Process each stroke SEQUENTIALLY
    for idx, row in strokes_data.iterrows():
        pts, _ = parse_stroke(row['description'])
        if len(pts) < 2:
            results.append({
                'uniqId': row['uniqId'],
                'timestamp': row['timestamp'],
                'is_scribble': False,
                'confidence': 0.0,
                'pattern_score': 0.0,
                'chaos_score': 0.0,
                'ink_density': 0.0,
                'points': pts
            })
            continue
        
        # === STEP 1: Scan canvas BEFORE adding stroke ===
        regions_before = detect_scribble_regions(canvas, refs)
        
        # === STEP 2: Create mask for new stroke ===
        stroke_mask = create_stroke_mask(canvas.shape, pts)
        
        # === STEP 3: Add stroke to canvas ===
        pts_int = [(int(x), int(y)) for x, y in pts]
        cv2.polylines(canvas, [np.array(pts_int)], False, 0, STROKE_WIDTH)
        
        # === STEP 4: Scan canvas AFTER adding stroke ===
        regions_after = detect_scribble_regions(canvas, refs)
        
        # === STEP 5: Find NEW scribble regions ===
        new_regions = find_new_scribble_regions(regions_before, regions_after)
        
        # === STEP 6: Check if THIS stroke contributed to new scribble ===
        is_scribble = False
        max_confidence = 0.0
        max_pattern = 0.0
        max_chaos = 0.0
        max_density = 0.0
        
        if len(new_regions) > 0:
            for new_region in new_regions:
                # Check if stroke intersects this new scribble region
                if stroke_intersects_region(pts, new_region):
                    # Check if stroke overlaps with existing ink (forming scribble)
                    overlap_pixels = count_overlapping_strokes(canvas, stroke_mask, new_region)
                    
                    if overlap_pixels > 50:  # meaningful overlap
                        is_scribble = True
                        max_confidence = max(max_confidence, new_region['confidence'])
                        max_pattern = max(max_pattern, new_region['pattern_score'])
                        max_chaos = max(max_chaos, new_region['chaos_score'])
                        max_density = max(max_density, new_region['ink_density'])
        
        results.append({
            'uniqId': row['uniqId'],
            'timestamp': row['timestamp'],
            'is_scribble': is_scribble,
            'confidence': max_confidence,
            'pattern_score': max_pattern,
            'chaos_score': max_chaos,
            'ink_density': max_density,
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
        
        # Color based on detection
        if result['is_scribble']:
            # Scribble = red with intensity based on confidence
            confidence = result['confidence']
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
    st.title("✍️ Hybrid Scribble Detection Dashboard")
    st.markdown("**Detection Method:** Density + Pattern Matching + Chaos Analysis")
    st.markdown("🎯 **Hybrid Approach:** Deteksi scribble dari gabungan multiple strokes")
    st.markdown("---")

    # ======================================================
    # INPUT SECTION
    # ======================================================
    st.subheader("📁 Input Data")
    csv_file = st.file_uploader("Upload CSV File", type=["csv"])

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

    # Advanced configuration
    with st.expander("⚙️ Advanced Configuration"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Detection Thresholds**")
            CONFIG['density_threshold'] = st.slider(
                "Ink Density Threshold",
                min_value=0.0,
                max_value=0.5,
                value=0.15,
                step=0.05,
                help="Minimum ink density untuk area scribble"
            )
            CONFIG['pattern_threshold'] = st.slider(
                "Pattern Match Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.45,
                step=0.05,
                help="Threshold pattern matching dengan reference"
            )
            CONFIG['chaos_threshold'] = st.slider(
                "Chaos Threshold",
                min_value=0.0,
                max_value=10.0,
                value=3.5,
                step=0.5,
                help="Threshold untuk irregularity/chaos"
            )
        
        with col2:
            st.markdown("**Scanning Parameters**")
            CONFIG['scan_grid_size'] = st.slider(
                "Scan Grid Size",
                min_value=50,
                max_value=250,
                value=120,
                step=10,
                help="Ukuran window untuk scanning canvas"
            )
            CONFIG['scan_overlap'] = st.slider(
                "Scan Overlap",
                min_value=0,
                max_value=100,
                value=40,
                step=10,
                help="Overlap antar grid"
            )

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
        
        # Show reference images
        with st.expander("👀 View Reference Images"):
            cols = st.columns(min(len(refs), 4))
            for i, ref in enumerate(refs):
                with cols[i % len(cols)]:
                    st.image(ref, caption=f"Reference {i+1}", use_container_width=True)
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

        # HYBRID SEQUENTIAL DETECTION
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
                'Confidence': result['confidence'],
                'PatternScore': result['pattern_score'],
                'ChaosScore': result['chaos_score'],
                'InkDensity': result['ink_density']
            })

    gantt_df = pd.DataFrame(gantt_data)

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
        hover_data=['Stroke', 'UniqId', 'Confidence', 'PatternScore', 'ChaosScore', 'InkDensity'],
        title='Stroke Activity Timeline by Actor (Hybrid Detection)'
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
    
    avg_confidence = gantt_df[gantt_df['Type'] == 'Scribble']['Confidence'].mean() if total_scribbles > 0 else 0
    avg_pattern = gantt_df[gantt_df['Type'] == 'Scribble']['PatternScore'].mean() if total_scribbles > 0 else 0
    avg_chaos = gantt_df[gantt_df['Type'] == 'Scribble']['ChaosScore'].mean() if total_scribbles > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Strokes", total_strokes)
    c2.metric("Scribbles", total_scribbles)
    c3.metric("Writing", total_writing)
    c4.metric("Scribble Rate", f"{(total_scribbles/total_strokes*100):.1f}%" if total_strokes > 0 else "0%")

    st.markdown("### Detection Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Confidence", f"{avg_confidence:.3f}")
    c2.metric("Avg Pattern Score", f"{avg_pattern:.3f}")
    c3.metric("Avg Chaos Score", f"{avg_chaos:.3f}")

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
                        ['Stroke', 'Type', 'UniqId', 'Confidence', 'PatternScore', 'ChaosScore', 'InkDensity', 'Start', 'Finish']
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
