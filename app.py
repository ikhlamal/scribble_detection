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
    page_title="Incremental Scribble Detection Dashboard",
    page_icon="✍️",
    layout="wide"
)

# =========================
# DETECTION CONFIG
# =========================
CANVAS_SIZE = (900, 1100)
STROKE_WIDTH = 3

CONFIG = {
    # Pattern matching - sama seperti kode original
    'pattern_threshold': 0.55,           # threshold untuk classify sebagai scribble
    'min_scribble_area': 2000,           # minimum area stroke (filter noise)
    'min_stroke_length': 50,             # minimum length stroke (filter noise)
    'min_consecutive': 4,                # minimum strictly consecutive scribbles
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


def get_stroke_bbox_and_metrics(points):
    """Get bounding box and basic metrics"""
    if len(points) < 2:
        return None
    
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, y1 = min(xs), min(ys)
    x2, y2 = max(xs), max(ys)
    bbox_w = x2 - x1 + 1
    bbox_h = y2 - y1 + 1
    bbox_area = bbox_w * bbox_h
    
    # Calculate length
    length = 0
    for i in range(len(points) - 1):
        dx = points[i+1][0] - points[i][0]
        dy = points[i+1][1] - points[i][1]
        length += math.sqrt(dx*dx + dy*dy)
    
    return {
        'bbox': (x1, y1, x2, y2),
        'bbox_area': bbox_area,
        'length': length,
        'points': points
    }


def match_with_references(canvas, bbox, refs):
    """Match region dengan references - SAMA SEPERTI KODE ORIGINAL"""
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
    
    # Resize ke ukuran reference
    target_size = refs[0].shape
    try:
        region_resized = cv2.resize(region, (target_size[1], target_size[0]))
    except:
        return 0.0
    
    # Threshold untuk binary
    _, region_bin = cv2.threshold(region_resized, 127, 255, cv2.THRESH_BINARY)
    
    max_score = 0.0
    for ref in refs:
        # SSIM
        try:
            score1 = ssim(region_bin, ref, data_range=255)
        except:
            score1 = 0
        
        # Template matching
        try:
            result = cv2.matchTemplate(region_bin, ref, cv2.TM_CCOEFF_NORMED)
            _, score2, _, _ = cv2.minMaxLoc(result)
        except:
            score2 = 0
        
        # Correlation
        try:
            result = cv2.matchTemplate(region_bin, ref, cv2.TM_CCORR_NORMED)
            _, score3, _, _ = cv2.minMaxLoc(result)
        except:
            score3 = 0
        
        # Combined - ambil yang tertinggi
        combined = max(score1, score2, score3)
        max_score = max(max_score, combined)
    
    return max_score


def is_stroke_complex(points, bbox_area, length):
    """
    Tentukan apakah stroke kompleks (layak jadi scribble meski isolated).
    Stroke kompleks = banyak titik, panjang, atau area besar
    """
    num_points = len(points)
    
    # Kriteria kompleksitas
    is_long = length > 300000 # stroke panjang
    is_large_area = bbox_area > 800000  # area besar
    is_many_points = num_points > 200  # banyak titik
    
    # Kompleks jika memenuhi salah satu kriteria dengan margin
    return is_long or is_large_area or is_many_points


def post_process_isolated_scribbles(results, min_consecutive=4):
    """
    POST-PROCESSING: Filter scribbles yang tidak strictly consecutive.
    
    Logic:
    - Scribble harus BENAR-BENAR BERUNTUN minimal min_consecutive stroke
    - TIDAK BOLEH TERPUTUS sama sekali (strictly consecutive)
    - Jika scribble beruntun < min_consecutive = BUKAN scribble (false positive)
    - KECUALI stroke tersebut kompleks (panjang/besar)
    
    Args:
        results: list of detection results
        min_consecutive: minimum jumlah scribble strictly consecutive
    """
    if len(results) == 0:
        return results
    
    # Create copy untuk modifikasi
    processed = results.copy()
    
    # Find all scribble indices
    scribble_indices = [i for i, r in enumerate(results) if r['is_scribble']]
    
    if len(scribble_indices) == 0:
        return processed
    
    # Group scribbles into STRICTLY consecutive sequences (no gaps)
    def find_strictly_consecutive_groups(indices):
        """Group indices yang BENAR-BENAR berurutan (gap = 0)"""
        if not indices:
            return []
        
        groups = []
        current_group = [indices[0]]
        
        for i in range(1, len(indices)):
            # STRICTLY consecutive: harus tepat +1 dari sebelumnya
            if indices[i] == indices[i-1] + 1:
                current_group.append(indices[i])
            else:
                # Terputus, mulai group baru
                groups.append(current_group)
                current_group = [indices[i]]
        
        groups.append(current_group)
        return groups
    
    # Find strictly consecutive groups
    groups = find_strictly_consecutive_groups(scribble_indices)
    
    # Filter groups that are too small
    for group in groups:
        if len(group) < min_consecutive:
            # Group terlalu kecil, cek apakah ada yang kompleks
            for idx in group:
                result = results[idx]
                
                # Check if complex stroke (exception)
                if is_stroke_complex(result['points'], result['bbox_area'], result['length']):
                    # Stroke kompleks, tetap scribble
                    continue
                else:
                    # Not complex and group too small -> filter
                    processed[idx]['is_scribble'] = False
                    processed[idx]['was_filtered'] = True
                    processed[idx]['filter_reason'] = f'group_too_small_{len(group)}_strictly_consecutive'
    
    return processed


def detect_scribbles_incremental(strokes_data, refs):
    """
    INCREMENTAL DETECTION:
    - Canvas dimulai kosong
    - Tiap stroke ditambah satu per satu
    - Setelah stroke ditambah, cek apakah area stroke tersebut sekarang scribble
    - Stroke yang membuat deteksi scribble = dilabeli scribble
    - POST-PROCESSING: Filter isolated scribbles
    """
    # Init canvas kosong
    canvas = np.ones(CANVAS_SIZE[::-1], dtype=np.uint8) * 255
    
    results = []
    
    # Process each stroke INCREMENTALLY
    for idx, row in strokes_data.iterrows():
        pts, _ = parse_stroke(row['description'])
        if len(pts) < 2:
            results.append({
                'uniqId': row['uniqId'],
                'timestamp': row['timestamp'],
                'is_scribble': False,
                'pattern_score': 0.0,
                'bbox_area': 0,
                'length': 0,
                'points': pts,
                'was_filtered': False
            })
            continue
        
        # Get basic metrics stroke ini
        metrics = get_stroke_bbox_and_metrics(pts)
        if metrics is None:
            results.append({
                'uniqId': row['uniqId'],
                'timestamp': row['timestamp'],
                'is_scribble': False,
                'pattern_score': 0.0,
                'bbox_area': 0,
                'length': 0,
                'points': pts,
                'was_filtered': False
            })
            continue
        
        # === STEP 1: Tambahkan stroke ke canvas ===
        pts_int = [(int(x), int(y)) for x, y in pts]
        cv2.polylines(canvas, [np.array(pts_int)], False, 0, STROKE_WIDTH)
        
        # === STEP 2: CEK apakah SEKARANG ada scribble di area stroke ini ===
        is_scribble = False
        pattern_score = 0.0
        
        # Filter noise berdasarkan size (sama seperti original)
        if metrics['bbox_area'] >= CONFIG['min_scribble_area'] and metrics['length'] >= CONFIG['min_stroke_length']:
            # Pattern matching di area stroke ini (dengan canvas yang SUDAH berisi stroke ini)
            if len(refs) > 0:
                pattern_score = match_with_references(canvas, metrics['bbox'], refs)
                is_scribble = pattern_score > CONFIG['pattern_threshold']
        
        results.append({
            'uniqId': row['uniqId'],
            'timestamp': row['timestamp'],
            'is_scribble': is_scribble,
            'pattern_score': pattern_score,
            'bbox_area': metrics['bbox_area'],
            'length': metrics['length'],
            'points': pts,
            'was_filtered': False
        })
    
    # === STEP 3: POST-PROCESSING - Filter non-strictly-consecutive scribbles ===
    results = post_process_isolated_scribbles(
        results, 
        min_consecutive=CONFIG.get('min_consecutive', 4)
    )
    
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
    st.title("✍️ Incremental Scribble Detection Dashboard")
    st.markdown("**Detection Method:** Pattern Matching (Original) + Incremental Canvas Update")
    st.markdown("🔄 **Incremental:** Deteksi stroke yang memicu munculnya scribble")
    st.markdown("🧹 **Post-Processing:** Filter scribbles yang tidak STRICTLY consecutive (≥4 beruntun tanpa terputus)")
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

    # Configuration
    with st.expander("⚙️ Detection Configuration"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Pattern Matching**")
            CONFIG['pattern_threshold'] = st.slider(
                "Pattern Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.55,
                step=0.05,
                help="Threshold untuk classify sebagai scribble"
            )
            CONFIG['min_scribble_area'] = st.number_input(
                "Min Scribble Area",
                min_value=100,
                max_value=10000,
                value=2000,
                step=100,
                help="Minimum area stroke untuk filter noise"
            )
        
        with col2:
            st.markdown("**Post-Processing**")
            CONFIG['min_stroke_length'] = st.number_input(
                "Min Stroke Length",
                min_value=10,
                max_value=500,
                value=50,
                step=10,
                help="Minimum length stroke untuk filter noise"
            )
            CONFIG['min_consecutive'] = st.slider(
                "Min Strictly Consecutive",
                min_value=2,
                max_value=10,
                value=4,
                step=1,
                help="Minimum jumlah scribble STRICTLY BERUNTUN (tidak boleh terputus)"
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
        
        # Show references
        with st.expander("👀 View Reference Images"):
            cols = st.columns(min(len(refs), 5))
            for i, ref in enumerate(refs):
                with cols[i % len(cols)]:
                    st.image(ref, caption=f"Ref {i+1}", use_container_width=True)
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

        # INCREMENTAL DETECTION
        results, canvas = detect_scribbles_incremental(actor_df, refs)
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
                'Area': result['bbox_area'],
                'Length': result['length']
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
        hover_data=['Stroke', 'UniqId', 'PatternScore', 'Area', 'Length'],
        title='Stroke Activity Timeline by Actor (Incremental Detection)'
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
    
    # Count filtered scribbles
    total_filtered = sum(1 for actor in actors for r in actor_data[actor]['results'] if r.get('was_filtered', False))
    
    avg_pattern_score_scribbles = gantt_df[gantt_df['Type'] == 'Scribble']['PatternScore'].mean() if total_scribbles > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Strokes", total_strokes)
    c2.metric("Scribbles", total_scribbles)
    c3.metric("Writing", total_writing)
    c4.metric("Filtered (False +)", total_filtered)

    if total_strokes > 0:
        scribble_rate = (total_scribbles / total_strokes) * 100
        st.markdown(f"**Scribble Rate:** {scribble_rate:.2f}% | **Avg Pattern Score:** {avg_pattern_score_scribbles:.3f}")
    
    if total_filtered > 0:
        st.info(f"ℹ️ **Post-processing:** {total_filtered} isolated scribble(s) filtered sebagai false positive")

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
                        ['Stroke', 'Type', 'UniqId', 'PatternScore', 'Area', 'Length', 'Start', 'Finish']
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
