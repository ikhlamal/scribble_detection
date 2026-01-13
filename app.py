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

st.set_page_config(
    page_title="Scribble Detection",
    page_icon="✍️",
    layout="wide"
)

CANVAS_SIZE = (900, 1100)
STROKE_WIDTH = 3

CONFIG = {
    # Pattern matching
    'pattern_threshold': 0.55,           
    'min_scribble_area': 5000,           
    'min_stroke_length': 50,            
    'min_consecutive': 4,              
}

def parse_stroke(stroke_str):
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
        # Normalized Cross-Correlation
        try:
            result = cv2.matchTemplate(region_bin, ref, cv2.TM_CCORR_NORMED)
            _, score, _, _ = cv2.minMaxLoc(result)
            max_score = max(max_score, score)
        except:
            continue
    
    return max_score


def is_stroke_complex(points, bbox_area, length):
    num_points = len(points)
    
    is_long = length > 600
    is_large_area = bbox_area > 800000
    is_many_points = num_points > 200
    
    return is_long or is_large_area or is_many_points


def post_process_isolated_scribbles(results, min_consecutive=4):
    """
    POST-PROCESSING: Filter scribbles yang tidak strictly consecutive.
    
    Logic:
    - Scribble harus BENAR-BENAR BERUNTUN minimal min_consecutive stroke
    - TIDAK BOLEH TERPUTUS sama sekali (strictly consecutive)
    - Jika scribble beruntun < min_consecutive = BUKAN scribble (false positive)
    - KECUALI stroke tersebut kompleks (panjang/besar)
    """
    if len(results) == 0:
        return results
    
    processed = results.copy()
    scribble_indices = [i for i, r in enumerate(results) if r['is_scribble']]
    
    if len(scribble_indices) == 0:
        return processed

    def find_strictly_consecutive_groups(indices):
        """Group indices yang BENAR-BENAR berurutan (gap = 0)"""
        if not indices:
            return []
        
        groups = []
        current_group = [indices[0]]
        
        for i in range(1, len(indices)):
            if indices[i] == indices[i-1] + 1:
                current_group.append(indices[i])
            else:
                groups.append(current_group)
                current_group = [indices[i]]
        
        groups.append(current_group)
        return groups
    
    groups = find_strictly_consecutive_groups(scribble_indices)
    for group in groups:
        if len(group) < min_consecutive:
            for idx in group:
                result = results[idx]
                if is_stroke_complex(result['points'], result['bbox_area'], result['length']):
                    continue
                else:
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
    canvas = np.ones(CANVAS_SIZE[::-1], dtype=np.uint8) * 255
    
    results = []

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
    
        pts_int = [(int(x), int(y)) for x, y in pts]
        cv2.polylines(canvas, [np.array(pts_int)], False, 0, STROKE_WIDTH)
        is_scribble = False
        pattern_score = 0.0
        if metrics['bbox_area'] >= CONFIG['min_scribble_area'] and metrics['length'] >= CONFIG['min_stroke_length']:
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
    
    results = post_process_isolated_scribbles(
        results, 
        min_consecutive=CONFIG.get('min_consecutive', 2)
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
        if result['is_scribble']:
            confidence = result['pattern_score']
            red = int(200 + 55 * min(confidence, 1.0))
            color = (red, 0, 0)
            text_color = (0, 200, 0)  # Green text
        else:
            color = (0, 0, 0)  # Black for writing
            text_color = (255, 200, 0)  # Yellow text
        
        draw_clean.line(pts, fill=color, width=STROKE_WIDTH)
        draw_annotated.line(pts, fill=color, width=STROKE_WIDTH)
        
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


def shorten_actor_name(actor_name, max_length=8):
    if len(actor_name) <= max_length:
        return actor_name
    return actor_name[:max_length-4] + "..."

def main():
    st.title("✍️ Scribble Detection")
    st.markdown("---")
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
    submitted = st.button("🚀 Submit & Process", type="primary")
    if not submitted:
        st.info("⬆️ Upload CSV dan klik **Submit & Process** untuk mulai")
        return

    if csv_file is None:
        st.error("❌ CSV belum di-upload")
        return
    try:
        df = pd.read_csv(csv_file)
        st.success(f"✅ Loaded {len(df)} rows from CSV")
    except Exception as e:
        st.error(f"❌ Error loading CSV: {e}")
        return

    df_filtered = df[df['operation_name'] == 'ADD_HW_MEMO'].copy()

    if df_filtered.empty:
        st.warning("⚠️ Tidak ditemukan ADD_HW_MEMO pada CSV")
        return

    df_filtered['timestamp'] = pd.to_datetime(df_filtered['timestamp'])
    actors = df_filtered['actor_name_id'].unique()

    if limit_actors and max_actors:
        actors = actors[:max_actors]

    total_actors = len(actors)
    st.markdown(f"### 👥 Actor diproses: **{total_actors}**")
    ref_folder = "scribble_refs"
    refs = load_scribble_refs(ref_folder)

    if refs:
        st.success(f"✅ Loaded {len(refs)} reference scribble images")
        with st.expander("👀 View Reference Images"):
            cols = st.columns(min(len(refs), 5))
            for i, ref in enumerate(refs):
                with cols[i % len(cols)]:
                    st.image(ref, caption=f"Ref {i+1}", use_container_width=True)
    else:
        st.error("❌ Tidak ada reference scribble image - Detection tidak bisa berjalan!")
        st.info("💡 Tambahkan reference scribble images di folder 'scribble_refs'")
        return
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

    st.markdown("---")
    st.header("📊 Gantt Chart - Stroke Timeline")

    gantt_data = []

    for actor in actors:
        data = actor_data[actor]
        actor_df = data['df']
        results = data['results']

        for i, (result, row) in enumerate(zip(results, actor_df.itertuples())):
            start_time = row.timestamp
            if i + 1 < len(actor_df):
                finish_time = actor_df.iloc[i + 1]['timestamp']
            else:
                finish_time = start_time + timedelta(seconds=1)

            gantt_data.append({
                'Actor': actor,
                'ActorDisplay': shorten_actor_name(actor),
                'Stroke': f"Stroke {i+1}",
                'Start': start_time,
                'Finish': finish_time,
                'Type': 'Scribble' if result['is_scribble'] else 'Writing',
                'UniqId': row.uniqId,
                'PatternScore': result['pattern_score'],
                'Area': result['bbox_area'],
                'Length': result['length'],
                'Date': start_time.date()
            })

    gantt_df = pd.DataFrame(gantt_data)
    unique_dates = gantt_df['Date'].unique()
    
    st.info(f"📅 Total unique dates found: **{len(unique_dates)}**")
    for date_idx, date in enumerate(unique_dates):
        date_df = gantt_df[gantt_df['Date'] == date].copy()
        
        st.markdown(f"### 📅 Date: {date.strftime('%Y-%m-%d')} ({date.strftime('%A')})")
        date_strokes = len(date_df)
        date_scribbles = len(date_df[date_df['Type'] == 'Scribble'])
        date_writing = len(date_df[date_df['Type'] == 'Writing'])
        date_actors = date_df['Actor'].nunique()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Strokes", date_strokes)
        col2.metric("Scribbles", date_scribbles)
        col3.metric("Writing", date_writing)
        col4.metric("Actors", date_actors)
        
        fig = px.timeline(
            date_df,
            x_start='Start',
            x_end='Finish',
            y='ActorDisplay', 
            color='Type',
            color_discrete_map={
                'Writing': 'black',
                'Scribble': 'red'
            },
            hover_data=['Actor', 'Stroke', 'UniqId', 'PatternScore', 'Area', 'Length'], 
            title=f'Stroke Activity Timeline - {date.strftime("%Y-%m-%d")}'
        )
        
        fig.update_yaxes(
            categoryorder='category ascending',
            title='Actor',
            showgrid=True,  
            gridcolor='rgba(128, 128, 128, 0.3)',  # Warna grid abu-abu
            gridwidth=1
        )
        fig.update_xaxes(
            showgrid=True,  # Tampilkan grid vertical
            gridcolor='rgba(128, 128, 128, 0.3)',  # Warna grid abu-abu
            gridwidth=1
        )
        fig.update_layout(
            height=max(400, date_actors * 80),
            xaxis_title="Time",
            yaxis_title="Actor",
            hovermode='closest',
            bargap=0.2,
            bargroupgap=0.05, 
            margin=dict(l=150, r=50, t=80, b=80), 
            plot_bgcolor='rgba(240, 240, 240, 0.3)' 
        )

        st.plotly_chart(fig, use_container_width=True)
        if date_idx < len(unique_dates) - 1:
            st.markdown("---")

    st.markdown("---")
    st.header("📈 Statistics")

    total_strokes = len(gantt_df)
    total_scribbles = len(gantt_df[gantt_df['Type'] == 'Scribble'])
    total_writing = len(gantt_df[gantt_df['Type'] == 'Writing'])
    
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
