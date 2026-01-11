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
from scipy.spatial. distance import cosine
import io
import base64
from collections import defaultdict

# =========================
# PAGE CONFIG
# =========================
st. set_page_config(
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
    # Similarity thresholds (LEBIH KETAT)
    'similarity_threshold': 0.55,        # threshold untuk combined similarity
    'min_matching_refs': 2,              # minimal match dengan N references
    
    # Area-based detection (SEQUENTIAL)
    'min_scribble_region_area': 3000,    # minimum area untuk region scribble
    'density_threshold': 0.15,           # minimal density untuk scribble (15% area terisi)
    'overlap_threshold': 0.3,            # minimal overlap ratio untuk scribble
    
    # Incremental detection
    'region_margin': 80,                 # margin untuk region analysis
    'min_strokes_in_region': 3,          # minimal stroke overlap untuk scribble
    
    # Texture features
    'entropy_threshold': 4.5,            # minimal entropy (randomness)
    'edge_density_threshold': 0.12,      # minimal edge density
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
    """Load reference scribbles dengan preprocessing"""
    refs = []
    refs_features = []
    
    if not os.path.exists(folder):
        return refs, refs_features
        
    for fn in os.listdir(folder):
        if fn.lower().endswith((".png", ".jpg", ".jpeg")):
            img = Image.open(os.path.join(folder, fn)).convert("L")
            img_arr = np.array(img)
            
            # Preprocessing
            _, img_arr = cv2.threshold(img_arr, 127, 255, cv2.THRESH_BINARY)
            img_resized = cv2.resize(img_arr, size)
            
            # Extract features untuk reference
            features = extract_texture_features(img_resized)
            
            refs.append(img_resized)
            refs_features.append(features)
    
    return refs, refs_features


def extract_texture_features(img):
    """Extract texture features untuk similarity comparison"""
    features = {}
    
    # 1. Entropy (randomness/chaos)
    hist, _ = np.histogram(img. ravel(), bins=256, range=(0, 256))
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    features['entropy'] = -np.sum(hist * np. log2(hist))
    
    # 2. Edge density
    edges = cv2.Canny(img, 50, 150)
    features['edge_density'] = np. sum(edges > 0) / (img.shape[0] * img.shape[1])
    
    # 3. Ink density
    features['ink_density'] = np. sum(img < 128) / (img.shape[0] * img.shape[1])
    
    # 4. Orientation histogram (HOG-like)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    angles = np.arctan2(sobely, sobelx)
    
    # Histogram of orientations (8 bins)
    hist_orient, _ = np.histogram(angles[edges > 0], bins=8, range=(-np.pi, np.pi))
    hist_orient = hist_orient / (hist_orient.sum() + 1e-6)
    features['orientation_hist'] = hist_orient
    
    # 5. Spatial frequency
    f_transform = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)
    features['spatial_frequency'] = np.mean(magnitude_spectrum)
    
    return features


def compute_similarity_score(region, refs, refs_features):
    """
    Compute comprehensive similarity score dengan multiple methods
    LEBIH OPTIMAL: kombinasi texture, structure, dan pattern
    """
    if len(refs) == 0:
        return 0.0, 0
    
    # Extract features dari region
    region_features = extract_texture_features(region)
    
    scores = []
    
    for ref, ref_features in zip(refs, refs_features):
        # Resize region ke ukuran reference
        try:
            region_resized = cv2.resize(region, (ref.shape[1], ref. shape[0]))
        except:
            scores.append(0.0)
            continue
        
        # Method 1: SSIM (structural similarity)
        try:
            score_ssim = ssim(region_resized, ref, data_range=255)
        except:
            score_ssim = 0
        
        # Method 2: Edge-based SSIM
        region_edges = cv2.Canny(region_resized, 50, 150)
        ref_edges = cv2.Canny(ref, 50, 150)
        try:
            score_edge_ssim = ssim(region_edges, ref_edges, data_range=255)
        except:
            score_edge_ssim = 0
        
        # Method 3: Feature similarity (PENTING untuk texture!)
        # Entropy similarity
        entropy_diff = abs(region_features['entropy'] - ref_features['entropy'])
        score_entropy = 1.0 / (1.0 + entropy_diff)
        
        # Edge density similarity
        edge_density_diff = abs(region_features['edge_density'] - ref_features['edge_density'])
        score_edge_density = 1.0 / (1.0 + edge_density_diff * 10)
        
        # Ink density similarity
        ink_density_diff = abs(region_features['ink_density'] - ref_features['ink_density'])
        score_ink_density = 1.0 / (1.0 + ink_density_diff * 5)
        
        # Orientation similarity (cosine similarity)
        try:
            score_orientation = 1.0 - cosine(
                region_features['orientation_hist'],
                ref_features['orientation_hist']
            )
            if np.isnan(score_orientation):
                score_orientation = 0
        except:
            score_orientation = 0
        
        # Method 4: Template matching
        try:
            result = cv2.matchTemplate(region_resized, ref, cv2.TM_CCOEFF_NORMED)
            _, score_template, _, _ = cv2.minMaxLoc(result)
        except:
            score_template = 0
        
        # COMBINED SCORE (weighted)
        combined = (
            score_ssim * 0.15 +              # Structure
            score_edge_ssim * 0.25 +         # Edge structure
            score_entropy * 0.20 +           # Chaos/randomness (PENTING!)
            score_edge_density * 0.15 +      # Edge density
            score_ink_density * 0.10 +       # Ink coverage
            score_orientation * 0.10 +       # Orientation pattern
            score_template * 0.05            # Template match
        )
        
        scores.append(combined)
    
    # Return max score dan jumlah yang match
    max_score = max(scores) if scores else 0.0
    matching_refs = sum(1 for s in scores if s > CONFIG['similarity_threshold'])
    
    return max_score, matching_refs


def get_stroke_bbox(points, margin=0):
    """Get bounding box with margin"""
    if len(points) < 2:
        return None
    
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, y1 = min(xs), min(ys)
    x2, y2 = max(xs), max(ys)
    
    x1 = max(0, int(x1) - margin)
    y1 = max(0, int(y1) - margin)
    x2 = int(x2) + margin
    y2 = int(y2) + margin
    
    return (x1, y1, x2, y2)


def bbox_overlap(bbox1, bbox2):
    """Calculate overlap area between two bboxes"""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    if x2 < x1 or y2 < y1:
        return 0
    
    return (x2 - x1) * (y2 - y1)


def detect_scribbles_sequential(strokes_data, refs, refs_features):
    """
    SEQUENTIAL DETECTION:  Setiap stroke ditambah → deteksi region → tandai scribble
    Ini yang Anda minta!
    """
    # Init canvas kosong
    canvas = np.ones(CANVAS_SIZE[: :-1], dtype=np.uint8) * 255
    
    # Track stroke bboxes dan overlap
    stroke_bboxes = []
    stroke_points = []
    
    # Results
    results = []
    scribble_regions = []  # Track region yang sudah terdeteksi scribble
    
    # Process SEQUENTIAL - satu per satu
    for idx, row in strokes_data.iterrows():
        pts, _ = parse_stroke(row['description'])
        if len(pts) < 2:
            results.append({
                'uniqId': row['uniqId'],
                'timestamp': row['timestamp'],
                'is_scribble': False,
                'reason': 'Invalid stroke',
                'similarity_score': 0.0,
                'matching_refs': 0,
                'region_density': 0.0,
                'overlapping_strokes': 0,
                'entropy': 0.0,
                'points': pts
            })
            continue
        
        # 1. TAMBAHKAN stroke ke canvas
        pts_int = [(int(x), int(y)) for x, y in pts]
        cv2.polylines(canvas, [np.array(pts_int)], False, 0, STROKE_WIDTH)
        
        # 2. Get bbox untuk stroke ini
        current_bbox = get_stroke_bbox(pts, margin=CONFIG['region_margin'])
        if not current_bbox:
            results. append({
                'uniqId': row['uniqId'],
                'timestamp': row['timestamp'],
                'is_scribble': False,
                'reason':  'No bbox',
                'similarity_score': 0.0,
                'matching_refs': 0,
                'region_density': 0.0,
                'overlapping_strokes': 0,
                'entropy': 0.0,
                'points': pts
            })
            continue
        
        # 3. CEK overlap dengan stroke sebelumnya
        overlapping_strokes = 0
        for prev_bbox in stroke_bboxes: 
            overlap_area = bbox_overlap(current_bbox, prev_bbox)
            bbox_area = (current_bbox[2] - current_bbox[0]) * (current_bbox[3] - current_bbox[1])
            if bbox_area > 0 and overlap_area / bbox_area > CONFIG['overlap_threshold']:
                overlapping_strokes += 1
        
        # 4. Extract REGION dari canvas (cumulative state!)
        x1, y1, x2, y2 = current_bbox
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(canvas.shape[1], x2)
        y2 = min(canvas.shape[0], y2)
        
        region = canvas[y1:y2, x1:x2]
        
        if region.shape[0] < 20 or region.shape[1] < 20:
            results. append({
                'uniqId': row['uniqId'],
                'timestamp': row['timestamp'],
                'is_scribble': False,
                'reason':  'Region too small',
                'similarity_score': 0.0,
                'matching_refs': 0,
                'region_density': 0.0,
                'overlapping_strokes': overlapping_strokes,
                'entropy': 0.0,
                'points': pts
            })
            stroke_bboxes.append(current_bbox)
            stroke_points.append(pts)
            continue
        
        # 5.  ANALISIS REGION (bukan stroke individual!)
        region_area = region.shape[0] * region. shape[1]
        ink_pixels = np.sum(region < 128)
        region_density = ink_pixels / region_area
        
        # Extract texture features
        region_features = extract_texture_features(region)
        
        # 6. CEK apakah region ini SCRIBBLE
        is_scribble = False
        reason = ""
        similarity_score = 0.0
        matching_refs = 0
        
        # Condition 1: Cukup overlap dengan stroke sebelumnya? 
        if overlapping_strokes < CONFIG['min_strokes_in_region']:
            reason = f"Not enough overlap ({overlapping_strokes} strokes)"
        # Condition 2: Density cukup tinggi?
        elif region_density < CONFIG['density_threshold']:
            reason = f"Low density ({region_density:.3f})"
        # Condition 3: Entropy cukup tinggi (chaotic)?
        elif region_features['entropy'] < CONFIG['entropy_threshold']:
            reason = f"Low entropy ({region_features['entropy']:.2f})"
        # Condition 4: Edge density cukup tinggi? 
        elif region_features['edge_density'] < CONFIG['edge_density_threshold']:
            reason = f"Low edge density ({region_features['edge_density']:.3f})"
        else:
            # 7. SIMILARITY CHECK dengan references
            similarity_score, matching_refs = compute_similarity_score(
                region, refs, refs_features
            )
            
            if similarity_score > CONFIG['similarity_threshold'] and matching_refs >= CONFIG['min_matching_refs']:
                is_scribble = True
                reason = f"SCRIBBLE DETECTED!  (sim={similarity_score:.3f}, refs={matching_refs}, overlaps={overlapping_strokes})"
                scribble_regions.append(current_bbox)
            else:
                reason = f"Pattern not matching (sim={similarity_score:.3f}, refs={matching_refs})"
        
        # 8. SAVE result
        results.append({
            'uniqId': row['uniqId'],
            'timestamp':  row['timestamp'],
            'is_scribble': is_scribble,
            'reason':  reason,
            'similarity_score': similarity_score,
            'matching_refs': matching_refs,
            'region_density': region_density,
            'overlapping_strokes': overlapping_strokes,
            'entropy': region_features['entropy'],
            'edge_density': region_features['edge_density'],
            'points': pts,
            'bbox': current_bbox
        })
        
        # 9. UPDATE tracking
        stroke_bboxes. append(current_bbox)
        stroke_points.append(pts)
    
    return results, canvas


def render_images(strokes_data, scribble_results):
    """Render clean dan annotated images"""
    # Clean version
    img_clean = Image.new("RGB", CANVAS_SIZE, (255, 255, 255))
    draw_clean = ImageDraw.Draw(img_clean)
    
    # Annotated version
    img_annotated = Image.new("RGB", CANVAS_SIZE, (255, 255, 255))
    draw_annotated = ImageDraw.Draw(img_annotated)
    
    for idx, (result, row) in enumerate(zip(scribble_results, strokes_data. itertuples())):
        pts = result['points']
        if len(pts) < 2:
            continue
        
        # Color
        if result['is_scribble']:
            # Scribble = red with intensity based on similarity
            confidence = result['similarity_score']
            red = int(200 + 55 * min(confidence, 1.0))
            color = (red, 0, 0)
            text_color = (0, 200, 0)  # Green text
        else:
            color = (0, 0, 0)  # Black for writing
            text_color = (100, 100, 100)  # Gray text
        
        # Draw on both
        draw_clean.line(pts, fill=color, width=STROKE_WIDTH)
        draw_annotated.line(pts, fill=color, width=STROKE_WIDTH)
        
        # Add annotation dengan info tambahan
        if len(pts) > 1:
            mid_idx = len(pts) // 2
            x, y = pts[mid_idx]
            
            if result['is_scribble']: 
                stroke_label = f"{idx+1}⚠"
            else:
                stroke_label = f"{idx+1}"
            
            text_offset_x = len(stroke_label) * 3
            text_offset_y = 5
            
            draw_annotated.text(
                (x - text_offset_x, y - text_offset_y),
                stroke_label,
                fill=text_color,
                stroke_width=1,
                stroke_fill=(255, 255, 255)
            )
    
    return img_clean, img_annotated


# =========================
# MAIN APP
# =========================
def main():
    st.title("✍️ Scribble Detection Dashboard (Sequential)")
    st.markdown("**Detection Method:** Sequential Region Analysis + Advanced Texture Similarity")
    st.markdown("🔍 **Setiap stroke ditambah → Region dianalisis → Deteksi scribble pada region**")
    st.markdown("---")

    # INPUT SECTION
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

    # Configuration expander
    with st.expander("⚙️ Advanced Configuration"):
        CONFIG['similarity_threshold'] = st.slider(
            "Similarity Threshold",
            0.0, 1.0, CONFIG['similarity_threshold'], 0.05
        )
        CONFIG['min_matching_refs'] = st.slider(
            "Min Matching References",
            1, 5, CONFIG['min_matching_refs'], 1
        )
        CONFIG['density_threshold'] = st.slider(
            "Density Threshold",
            0.0, 0.5, CONFIG['density_threshold'], 0.01
        )
        CONFIG['min_strokes_in_region'] = st.slider(
            "Min Overlapping Strokes",
            1, 10, CONFIG['min_strokes_in_region'], 1
        )
        CONFIG['entropy_threshold'] = st.slider(
            "Entropy Threshold (Chaos)",
            0.0, 8.0, CONFIG['entropy_threshold'], 0.5
        )

    submitted = st.button("🚀 Submit & Process", type="primary")

    if not submitted:
        st.info("⬆️ Upload CSV dan klik **Submit & Process** untuk mulai")
        return

    if csv_file is None:
        st.error("❌ CSV belum di-upload")
        return

    # LOAD CSV
    try:
        df = pd.read_csv(csv_file)
        st.success(f"✅ Loaded {len(df)} rows from CSV")
    except Exception as e:
        st.error(f"❌ Error loading CSV: {e}")
        return

    # FILTER
    df_filtered = df[df['operation_name'] == 'ADD_HW_MEMO']. copy()

    if df_filtered.empty:
        st.warning("⚠️ Tidak ditemukan ADD_HW_MEMO pada CSV")
        return

    df_filtered['timestamp'] = pd.to_datetime(df_filtered['timestamp'])

    # ACTOR SELECTION
    actors = df_filtered['actor_name_id'].unique()
    if limit_actors and max_actors:
        actors = actors[:max_actors]

    total_actors = len(actors)
    st.markdown(f"### 👥 Actor diproses: **{total_actors}**")

    # LOAD REFERENCES
    ref_folder = "scribble_refs"
    refs, refs_features = load_scribble_refs(ref_folder)

    if refs: 
        st.success(f"✅ Loaded {len(refs)} reference scribble images dengan texture features")
        
        # Show reference samples
        with st.expander("📸 Reference Scribble Samples"):
            cols = st.columns(min(len(refs), 5))
            for i, (ref, col) in enumerate(zip(refs[: 5], cols)):
                with col:
                    st.image(ref, caption=f"Ref {i+1}", use_container_width=True)
                    st.caption(f"Entropy: {refs_features[i]['entropy']:.2f}")
    else:
        st.error("❌ Tidak ada reference scribble image - Detection tidak bisa berjalan!")
        st.info("💡 Tambahkan reference scribble images di folder 'scribble_refs'")
        return

    # PROCESS EACH ACTOR
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

        # SEQUENTIAL DETECTION! 
        results, canvas = detect_scribbles_sequential(actor_df, refs, refs_features)
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

    # GANTT CHART
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
                'Similarity': result['similarity_score'],
                'MatchingRefs': result['matching_refs'],
                'Density': result['region_density'],
                'Overlaps': result['overlapping_strokes'],
                'Entropy': result['entropy'],
                'Reason': result['reason']
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
        hover_data=['Stroke', 'Similarity', 'MatchingRefs', 'Density', 'Overlaps', 'Entropy', 'Reason'],
        title='Sequential Stroke Detection Timeline'
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

    # STATISTICS
    st.markdown("---")
    st.header("📈 Statistics")

    total_strokes = len(gantt_df)
    total_scribbles = len(gantt_df[gantt_df['Type'] == 'Scribble'])
    total_writing = len(gantt_df[gantt_df['Type'] == 'Writing'])
    
    avg_similarity = gantt_df[gantt_df['Type'] == 'Scribble']['Similarity'].mean() if total_scribbles > 0 else 0
    avg_overlaps = gantt_df[gantt_df['Type'] == 'Scribble']['Overlaps'].mean() if total_scribbles > 0 else 0
    avg_entropy = gantt_df[gantt_df['Type'] == 'Scribble']['Entropy']. mean() if total_scribbles > 0 else 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Strokes", total_strokes)
    c2.metric("Scribbles", total_scribbles)
    c3.metric("Writing", total_writing)
    c4.metric("Avg Similarity", f"{avg_similarity:.3f}")
    c5.metric("Avg Overlaps", f"{avg_overlaps:.1f}")
    c6.metric("Avg Entropy", f"{avg_entropy:.2f}")

    # IMAGES PER ACTOR
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
                        ['Stroke', 'Type', 'Similarity', 'MatchingRefs', 'Density', 'Overlaps', 'Entropy', 'Reason']
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
