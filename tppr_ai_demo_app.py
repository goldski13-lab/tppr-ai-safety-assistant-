
import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt

# Load dataset
DF_PATH = "/mnt/data/tppr_simulated.csv"
df = pd.read_csv(DF_PATH, parse_dates=["timestamp"])

st.set_page_config(page_title="TPPR AI Demo", layout="wide")

st.title("TPPR AI Safety Assistant — Live Simulation")
st.markdown("Simulated TPPR data playback with onboard AI anomaly detection and alert ranking.")

# Sidebar controls
st.sidebar.header("Playback Controls")
speed = st.sidebar.selectbox("Playback speed", options=["0.5x", "1x", "2x", "5x"], index=1)
speed_map = {"0.5x": 0.5, "1x": 1.0, "2x": 2.0, "5x": 5.0}
speed_factor = speed_map[speed]
play = st.sidebar.checkbox("Play", value=False)

st.sidebar.header("AI Settings")
window = st.sidebar.slider("Rolling window (minutes) for baseline", min_value=3, max_value=30, value=12)
z_threshold = st.sidebar.slider("Anomaly z-score threshold", min_value=1.0, max_value=5.0, value=3.0, step=0.5)
ranking_sensitivity = st.sidebar.slider("Ranking sensitivity (slope multiplier)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

# Channels
channels = df['channel'].unique().tolist()
channel_map = {1: "CH4 (methane)", 2: "H2S", 3: "CO"}
thresholds = {1: 100, 2: 50, 3: 200}

# Playback state
if "pos" not in st.session_state:
    st.session_state.pos = 0

rows = df.shape[0]
step_seconds = 1.0 / speed_factor  # how fast to iterate (1 second per simulated minute / speed)

col1, col2 = st.columns([2,1])

with col2:
    st.subheader("Active Alerts")
    alert_table = st.empty()
    st.subheader("TPPR vs AI")
    status_box = st.empty()

with col1:
    st.subheader("Channel Readings")
    chart_area = st.empty()
    st.subheader("Timeline (current minute)")
    timeline_area = st.empty()

# Helper: compute rolling z-score anomaly per channel
def detect_anomaly(series, window, z_thresh):
    if len(series) < window+1:
        return False, 0.0, 0.0
    recent = series[-window:]
    mean = np.mean(recent)
    std = np.std(recent, ddof=0)
    if std == 0:
        return False, 0.0, 0.0
    last = series.iloc[-1]
    z = (last - mean) / std
    return (abs(z) >= z_thresh), float(z), float(mean)

# Simple slope-based time-to-threshold estimate
def time_to_threshold(series, threshold):
    # use last N points to compute slope (minutes)
    if len(series) < 3:
        return None, 0.0
    x = np.arange(len(series))
    y = series.values
    # linear fit
    A = np.vstack([x, np.ones_like(x)]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    if m <= 0:
        return None, m  # not increasing
    # estimate minutes until threshold
    last = y[-1]
    minutes = (threshold - last) / m
    return max(minutes, 0.0), m

# Playback loop - driven by Play control
def render_at_index(idx):
    # prepare snapshot up to idx
    snapshot = df.iloc[:idx+1].copy()
    latest = snapshot.tail(len(channels))
    # Build per-channel series for plotting
    fig, axs = plt.subplots(len(channels), 1, figsize=(10, 3*len(channels)), sharex=True)
    ai_alerts = []
    ai_alerts_rows = []
    tppr_alarm_rows = []
    for i, ch in enumerate(channels):
        ch_data = snapshot[snapshot['channel']==ch].copy()
        ch_data = ch_data.set_index('timestamp').resample('1T').mean().ffill()
        series = ch_data['gas_level_ppm']
        axs[i].plot(series.index, series.values)
        axs[i].set_title(f"Channel {ch} — {series.name if series.name else ''} {('('+str(ch)+')') if True else ''} — Threshold {thresholds[ch]} ppm")
        axs[i].set_ylabel("ppm")
        # mark TPPR alarm points
        alarm_points = series[series >= thresholds[ch]]
        if not alarm_points.empty:
            axs[i].scatter(alarm_points.index, alarm_points.values, marker='x')
        # AI anomaly detection using rolling window of last 'window' minutes
        is_anom, zscore, mean = detect_anomaly(series, window, z_threshold)
        t2t, slope = time_to_threshold(series.tail(max(window,5)), thresholds[ch])
        urgency = 0.0
        if is_anom:
            # severity score = zscore * (1 + slope * ranking_sensitivity)
            urgency = abs(zscore) * (1 + max(0.0, slope) * ranking_sensitivity)
            ai_alerts.append({
                "channel": ch,
                "gas": channel_map.get(ch, str(ch)),
                "latest_ppm": float(series.iloc[-1]),
                "zscore": round(zscore,2),
                "slope": round(slope,3),
                "minutes_to_threshold": None if t2t is None else round(t2t,1),
                "urgency": round(urgency,3)
            })
            ai_alerts_rows.append([channel_map.get(ch, str(ch)), series.index[-1].strftime("%Y-%m-%d %H:%M:%S"), round(series.iloc[-1],2), round(zscore,2), None if t2t is None else round(t2t,1), round(urgency,3)])
        # TPPR alarm detection (simulated)
        tppr_alarm = df[(df['channel']==ch) & (df['alarm_state']==1) & (df.index <= idx)]
        if not tppr_alarm.empty:
            last_alarm = tppr_alarm.iloc[-1]
            tppr_alarm_rows.append([channel_map.get(ch, str(ch)), last_alarm['timestamp'], round(last_alarm['gas_level_ppm'],2)])
    # Display chart
    chart_area.pyplot(fig)
    # Update alerts table and TPPR status
    if ai_alerts:
        alerts_df = pd.DataFrame(ai_alerts).sort_values("urgency", ascending=False)
        alert_table.table(alerts_df[['channel','gas','latest_ppm','zscore','minutes_to_threshold','urgency']])
    else:
        alert_table.markdown("**No AI alerts** — system nominal.")

    if tppr_alarm_rows:
        tppr_df = pd.DataFrame(tppr_alarm_rows, columns=["channel","timestamp","ppm"])
        status_box.table(tppr_df)
    else:
        status_box.markdown("**No TPPR alarms have occurred yet.**")

# Main loop control
if play:
    # iterate while Play is checked
    if st.session_state.pos >= rows-1:
        st.session_state.pos = 0
    for i in range(st.session_state.pos, rows):
        st.session_state.pos = i
        render_at_index(i)
        time.sleep(step_seconds)
        # allow manual break
        if not st.checkbox("Keep Playing (uncheck to stop)", value=True, key=f"playchk_{i}"):
            break
else:
    # show current snapshot at st.session_state.pos
    render_at_index(st.session_state.pos)
    # allow stepping
    col_prev, col_next = st.columns(2)
    if col_prev.button("<< Back"):
        st.session_state.pos = max(0, st.session_state.pos - len(channels))
    if col_next.button("Forward >>"):
        st.session_state.pos = min(rows-1, st.session_state.pos + len(channels))

st.markdown("---")
st.write("Simulation time:", df.iloc[st.session_state.pos]['timestamp'])
