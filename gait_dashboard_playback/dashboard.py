"""Offline gait-state playback synchronized to IMU signals and CNN predictions.

Run: python -m streamlit run dashboard.py
The trained model sees a causal 148-sample window and outputs TWO binary states,
not a continuous 0–100% gait-cycle phase estimate.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gait_dashboard_core import add_binary_predictions, calculate_metrics, infer_recording, transition_times

st.set_page_config(page_title="Gait CNN | Recorded playback", page_icon="🚶", layout="wide")
st.title("Gait CNN · Recorded Playback")
st.caption("Offline experiment · IMU signal, 148-sample CNN window, and bilateral stance/swing prediction")

@st.cache_data(show_spinner="Running the CNN across the complete recording…")
def run_inference(csv_bytes: bytes, model_bytes: bytes, sampling_rate: float):
    return infer_recording(csv_bytes, model_bytes, sampling_rate_hz=sampling_rate)

@st.cache_data(show_spinner=False)
def get_signals(csv_bytes: bytes, features: tuple[str, ...], sampling_rate: float):
    df = pd.read_csv(io.BytesIO(csv_bytes), low_memory=False)
    columns = list(features)
    arr = df[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    if "time_s" in df:
        tt = pd.to_numeric(df["time_s"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(tt).all() or np.any(np.diff(tt) <= 0):
            tt = np.arange(len(df), dtype=float) / sampling_rate
    else:
        tt = np.arange(len(df), dtype=float) / sampling_rate
    return arr, tt


def metric_str(x, fmt=".2f", unit=""):
    return "—" if x is None or not np.isfinite(x) else f"{x:{fmt}}{unit}"


def signal_plot(signals, sample_times, features, end_row, window, chosen, seconds):
    wstart = max(0, end_row - window + 1)
    t = sample_times[end_row]
    lo = max(float(sample_times[0]), t - seconds)
    first = int(np.searchsorted(sample_times, lo, side="left"))
    inds = np.arange(first, end_row + 1)
    vals = signals[inds, features.index(chosen)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sample_times[inds], y=vals, mode="lines", name=chosen,
                             line=dict(color="#2563eb", width=2)))
    fig.add_vrect(x0=float(sample_times[wstart]), x1=float(sample_times[end_row]),
                  fillcolor="rgba(22,163,74,0.16)", opacity=0.8, line_width=0,
                  annotation_text="CNN input: 148 samples", annotation_position="top left")
    fig.add_vline(x=float(t), line_color="#f97316", line_dash="dash")
    fig.update_layout(height=330, margin=dict(l=15,r=15,t=32,b=25),
                      xaxis_title="Recording time (s)", yaxis_title="IMU channel (raw units)",
                      hovermode="x unified", showlegend=False)
    return fig


def state_plot(pred, now, history_s, threshold):
    view = pred[(pred.time_s >= now - history_s) & (pred.time_s <= now)]
    if view.empty:
        view = pred.iloc[:1]
    fig = go.Figure()
    for col, name, offset, color in [("L_pred","Left leg",0,"#16a34a"),
                                     ("R_pred","Right leg",2,"#2563eb")]:
        fig.add_trace(go.Scatter(x=view.time_s, y=view[col] + offset, mode="lines",
                                 name=name, line=dict(color=color, width=3, shape="hv"),
                                 hovertemplate="t=%{x:.3f}s<extra>"+name+"</extra>"))
    if {"L_true", "R_true"}.issubset(view.columns):
        for col, offset, color, name in [("L_true",0,"#a7d8ae","FSR left"),
                                         ("R_true",2,"#9dbdf5","FSR right")]:
            truth = view.copy()
            truth[col] = pd.to_numeric(truth[col], errors="coerce")
            truth.loc[~truth[col].isin([0,1]), col] = np.nan
            fig.add_trace(go.Scatter(x=truth.time_s, y=truth[col] + offset,
                                     mode="lines", name=name, line=dict(color=color,width=1.5,
                                     dash="dot",shape="hv")))
    fig.add_vline(x=float(now),line_color="#f97316",line_dash="dash")
    fig.update_layout(height=310, margin=dict(l=15,r=15,t=25,b=25), xaxis_title="Recording time (s)",
                      yaxis=dict(tickmode="array",tickvals=[0,1,2,3],
                                 ticktext=["Left swing","Left stance","Right swing","Right stance"],
                                 range=[-.3,3.3]),
                      legend=dict(orientation="h",y=1.22,x=0),hovermode="x unified")
    return fig


def probability_plot(pred,now,history_s,threshold):
    view=pred[(pred.time_s >= now-history_s)&(pred.time_s<=now)]
    fig=go.Figure()
    for col,name,color in [("L_prob_stance","Left P(stance)","#16a34a"),
                           ("R_prob_stance","Right P(stance)","#2563eb")]:
        fig.add_trace(go.Scatter(x=view.time_s,y=view[col],name=name,mode="lines",
                                 line=dict(color=color,width=2)))
    fig.add_hline(y=threshold,line_dash="dot",line_color="gray")
    fig.add_vline(x=float(now),line_color="#f97316",line_dash="dash")
    fig.update_layout(height=250,margin=dict(l=15,r=15,t=20,b=25),
                      xaxis_title="Recording time (s)",yaxis=dict(title="P(stance)",range=[0,1]),
                      legend=dict(orientation="h",y=1.18,x=0),hovermode="x unified")
    return fig

with st.sidebar:
    st.header("Recording and model")
    model_file=st.file_uploader("CNN checkpoint (.pt)",type=["pt","pth"])
    csv_file=st.file_uploader("Recorded IMU data (.csv)",type=["csv"])
    threshold=st.slider("Stance threshold",0.05,0.95,0.50,0.01)
    rate=st.number_input("Sampling rate if time_s is missing (Hz)",
                         min_value=1.0,max_value=2000.0,value=148.15)
    st.caption("CSV sensors must match checkpoint features; FSR labels are optional for evaluation.")

if model_file is None or csv_file is None:
    st.info("Upload one trained .pt checkpoint and one recorded .csv file in the sidebar. Dense inference will run once, followed by interactive recorded playback.")
    st.stop()

try:
    raw,info=run_inference(csv_file.getvalue(),model_file.getvalue(),float(rate))
    pred=add_binary_predictions(raw,threshold)
    signals,sample_times=get_signals(csv_file.getvalue(),tuple(info["features"]),float(rate))
except Exception as ex:
    st.error(f"Could not load the recording or model: {ex}")
    st.stop()

# One-second playback ticks are selected from the already computed dense predictions.
# This does NOT mean that the 148-sample window shifts by 148 rows in training.
first_time=float(pred.time_s.iloc[0]);last_time=float(pred.time_s.iloc[-1])
seconds = np.arange(first_time,last_time + 1e-7,1.0)
if seconds.size==0:
    seconds=np.array([first_time])
indices=np.searchsorted(pred.time_s.to_numpy(),seconds,side="left")
indices=np.clip(indices,0,len(pred)-1)
# Include final time if farther than 0.5 s after last playback frame.
if last_time-float(pred.time_s.iloc[indices[-1]])>.5:
    indices=np.append(indices,len(pred)-1)
indices=np.unique(indices)
frames=pred.iloc[indices].reset_index(drop=True)

identifier=(model_file.name,model_file.size,csv_file.name,csv_file.size)
if st.session_state.get("recording_identifier")!=identifier:
    st.session_state.recording_identifier=identifier
    st.session_state.frame_pos=0
    st.session_state.playing=False

with st.expander("Recording and model information"):
    st.write(f"**Model:** {info['mode']} · {info['n_channels']} IMU channels · {info['window']} samples/window · checkpoint epochs: {info['epochs']}")
    st.write(f"**Input IMUs:** {', '.join(info['sensors'])} · **Inference device:** {info['device']}")
    st.write(f"**Recording:** {info['recording_rows']:,} samples · **Dense CNN predictions:** {len(pred):,} · **Playback updates:** {len(frames):,} (~1/s)")
    st.caption("CNN prediction is stance/swing for each leg, not continuous 0–100% gait-cycle phase.")

controls=st.container()

# The fragment reruns once per second while playing, without redoing inference.
@st.fragment(run_every="1s")
def playback():
    with controls:
        st.subheader("Playback controls")
        c1,c2,c3,c4=st.columns([1,1,2,2])
        with c1:
            if st.button("⏸ Pause" if st.session_state.playing else "▶ Play",use_container_width=True):
                st.session_state.playing=not st.session_state.playing
                st.rerun(scope="fragment")
        with c2:
            if st.button("↺ Restart",use_container_width=True):
                st.session_state.frame_pos=0
                st.session_state.playing=False
                st.rerun(scope="fragment")
        with c3:
            time_idx=st.slider("Position (one-second steps)",0,len(frames)-1,
                               value=int(st.session_state.frame_pos))
            if int(time_idx)!=int(st.session_state.frame_pos):
                st.session_state.frame_pos=int(time_idx)
                st.session_state.playing=False
        with c4:
            signal_choice=st.selectbox("IMU channel",info["features"],
                                       index=next((j for j,x in enumerate(info["features"]) if x.endswith("gyr_y")),0))

        frame=int(st.session_state.frame_pos)
        current=frames.iloc[frame]
        t=float(current.time_s)
        end_row=int(current.row)
        left_state="STANCE" if current.L_pred else "SWING"
        right_state="STANCE" if current.R_pred else "SWING"
        st.caption(f"Playback: **{t:.2f} s** of **{last_time:.2f} s** · sample row {end_row:,} · CNN window [{end_row-info['window']+1:,} – {end_row:,}]")
        a,b,c=st.columns(3)
        a.metric("Left leg",left_state,f"P(stance) {current.L_prob_stance:.1%}",delta_color="off")
        b.metric("Right leg",right_state,f"P(stance) {current.R_prob_stance:.1%}",delta_color="off")
        c.metric("Playback step",f"{frame+1:,}/{len(frames):,}","1 second/step",delta_color="off")

        st.subheader("1 · IMU signal and highlighted CNN input window")
        st.caption("Blue: measured IMU channel. Green shading: the exact 148-sample history used for the displayed prediction. Orange cursor: prediction time.")
        st.plotly_chart(signal_plot(signals,sample_times,info["features"],end_row,info["window"],signal_choice,7),use_container_width=True,key="imu_chart")
        st.subheader("2 · Predicted left/right gait states")
        st.caption("0 = swing, 1 = stance. Dashed lighter lines are FSR reference states when available; missing FSR labels are not treated as swing.")
        st.plotly_chart(state_plot(pred,t,12,threshold),use_container_width=True,key="states_chart")
        st.subheader("3 · CNN stance probabilities")
        st.plotly_chart(probability_plot(pred,t,12,threshold),use_container_width=True,key="prob_chart")
        st.subheader("4 · Gait summary up to current playback time")
        so_far=pred[pred.time_s<=t]
        elapsed=float(so_far.time_s.iloc[-1]-so_far.time_s.iloc[0]) if len(so_far)>1 else 0.0
        rows=so_far.row.to_numpy()
        times=so_far.time_s.to_numpy()
        le=transition_times(so_far.L_pred.to_numpy(),times,rows)
        re=transition_times(so_far.R_pred.to_numpy(),times,rows)
        # Use heel strikes only after enough data have elapsed; no measured stride length available.
        cadence=(len(le)+len(re))*60/elapsed if elapsed>=10 and len(le)+len(re)>=4 else None
        ls=float(np.median(np.diff(le))) if len(le)>=3 else None
        rs=float(np.median(np.diff(re))) if len(re)>=3 else None
        d1,d2,d3,d4=st.columns(4)
        d1.metric("Estimated cadence",metric_str(cadence,".1f"," steps/min"))
        d2.metric("Left stride time",metric_str(ls,".2f"," s"))
        d3.metric("Right stride time",metric_str(rs,".2f"," s"))
        both=np.mean((so_far.L_pred.to_numpy()==1)&(so_far.R_pred.to_numpy()==1))*100
        d4.metric("Double support",metric_str(both,".1f","%"))
        st.caption("Preliminary predictions-derived estimates. Cadence requires at least 10 seconds and four detected events; stride times require three events per leg. Stride length requires additional distance/speed estimation.")

        # A play tick is a display update, NOT a new CNN model evaluation.
        if st.session_state.playing:
            if st.session_state.frame_pos<len(frames)-1:
                st.session_state.frame_pos+=1
            else:
                st.session_state.playing=False

playback()

st.divider()
st.subheader("Whole-recording analysis and export")
summary=calculate_metrics(pred)
cols=st.columns(4)
for col,key,label,unit in [(cols[0],"cadence_spm","Cadence"," steps/min"),
                           (cols[1],"left_stride_time_s","Left stride time"," s"),
                           (cols[2],"right_stride_time_s","Right stride time"," s"),
                           (cols[3],"double_support_pct","Double support","%")]:
    col.metric(label,metric_str(summary.get(key),".2f",unit))
if info["has_reference_labels"]:
    l=summary.get("left_classification",{});r=summary.get("right_classification",{})
    st.write(f"**FSR comparison (dense):** left accuracy {metric_str(l.get('accuracy'),'.1%', '')}, right accuracy {metric_str(r.get('accuracy'),'.1%', '')}.")
    st.caption("A held-out subject checkpoint is required for unbiased evaluation. Never use a model on recordings included in its training fold for a final accuracy claim.")
st.download_button("Download dense predictions (.csv)",data=pred.to_csv(index=False).encode(),
                   file_name="gait_dense_predictions.csv",mime="text/csv")
st.warning("Research demonstration only; not a clinically validated gait assessment device.")
