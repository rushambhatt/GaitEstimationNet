#!/usr/bin/env python3
"""
label_gait.py - Build IMU training CSVs with FSR-derived gait labels from LabRecorder .xdf files.

Labels (one set per leg, so the network can use either leg or both):
    L_stance, R_stance : binary  1 = stance (foot on ground), 0 = swing, -1 = unlabeled
    phase_code         : 2-bit binary code "LR" as an integer 0..3  (= 2*L_stance + R_stance)
                           3 = "11" double support, 2 = "10" left single stance,
                           1 = "01" right single stance, 0 = "00" no foot down (should not occur in walking)
                          -1 = unlabeled (either leg unlabeled)
    L_gait_pct, R_gait_pct : 0-100 % of the gait cycle, heel strike to next heel strike of the same foot
                           (for RMSE evaluation; NaN when unlabeled)
    L_HS, L_TO, R_HS, R_TO : 1 on the IMU sample nearest each FSR heel-strike / toe-off event, else 0

Each row is one IMU sample (~148 Hz). FSR data are mapped onto IMU time by timestamp, never by row.

Standing period and IMU calibration:
    Each trial starts with the subject standing still. The script finds the quietest window in that
    period (STATIC_WIN_S long, ending at least 1 s before walking starts) and, per sensor:
      1. gyro bias  = mean gyro while still            -> subtracted (true rate is 0 when still)
      2. acc scale  = 1 / |mean acc while still|        -> |acc| = 1 g when still
      3. tilt       = smallest rotation taking the still gravity vector onto the sensor's vertical
                      axis (default: the axis closest to gravity, e.g. +y on the shanks/thighs/torso,
                      +z on the foot; use --vertical-axis z to force +z on every sensor). The same
                      rotation is applied to the gyro so acc and gyro stay in one frame.
    After calibration a still sensor reads acc = (0, 1, 0) or (0, 0, 1) g and gyro = (0, 0, 0) deg/s.
    The standing period is then removed: the CSV starts at walking onset (time_s = 0).
    The calibration (bias, scale, rotation matrix) is saved per trial in <trial>_imu_calibration.json so the
    same correction can be applied to live data in the real-time system.

Usage:
    Easiest: put label_gait.py in the same folder as the .xdf files and press Run in VS Code
    (or: python label_gait.py). Every .xdf in that folder is processed and the results go to a
    "labeled_output" subfolder there.
    python label_gait.py trial1.xdf trial2.xdf ... --out labeled_output
    python label_gait.py *.xdf --out out --trim 20261030_2_0mph=15:135
         (--trim NAME=START:END restricts labeling to START..END seconds of that trial, measured from
          walking onset; NAME is matched as a substring of the file name. Rows outside are unlabeled.)
    --vertical-axis z      force gravity onto +z for every sensor (default: nearest axis)
    --no-imu-cal           skip the calibration (raw IMU values)
    --keep-static          keep the standing period in the CSV (unlabeled)

Requires only numpy and matplotlib (an XDF reader is built in, so pyxdf is not needed).
"""
import argparse, json, os, struct, sys, xml.etree.ElementTree as ET
import numpy as np

# ----------------------------------------------------------------------------- configuration
SENSOR_MAP = {            # Trigno serial -> body location (from Sensor_IDs_serial_numbers.txt)
    "52731": "LeftShank",
    "52691": "RightShank",
    "52696": "LeftThigh",
    "52701": "RightThigh",
    "52708": "Torso",
    "52675": "LeftFoot",   # Trigno slot 6
    "52733": "RightFoot",  # Trigno slot 8
}
FSR_FEET = {               # FSR channel -> foot (L1 = left, L2 = right, verified against shank IMUs)
    "L": ["L1_Heel", "L1_Mid", "L1_Toe"],
    "R": ["L2_Heel", "L2_Mid", "L2_Toe"],
}
CONTACT_ON = 0.20          # contact starts when any FSR of a foot rises above this fraction of its range
CONTACT_OFF = 0.12         # ...and ends when all fall below this fraction (hysteresis)
MIN_GAP_FRAC = 0.12        # merge contacts separated by less than this fraction of the median stride (swing is ~40 %)
MIN_CONTACT_FRAC = 0.15    # drop contacts shorter than this fraction of the median stride
STRIDE_TOL = 0.25          # valid stride: within +/-25 % of the trial's median stride time
STANCE_PCT_RANGE = (45.0, 80.0)  # valid stride: stance between these % of the stride
CHANNEL_RESPONSE = 0.25    # a channel "responded" in a stride if its peak > this fraction of its typical peak
WALK_GYRO_DPS = 30.0       # walking onset: smoothed max |gyro| (bias-removed) over all IMUs exceeds this...
WALK_ACTIVE_FRAC = 0.3     # ...and stays above it for at least this fraction of the next 3 s
STATIC_WIN_S = 4.0         # length of the still window used for calibration (shorter if standing was shorter)
STATIC_MARGIN_S = 1.0      # the still window ends at least this long before walking onset


# ----------------------------------------------------------------------------- IMU calibration
def walking_onset(t, imu, n_sensors, fs):
    """Time when sustained movement starts. Uses a rough bias estimate from the first 2 s."""
    k0 = t < t[0] + 2.0
    gn = np.array([np.linalg.norm(imu[:, 6 * i + 3:6 * i + 6] - imu[k0, 6 * i + 3:6 * i + 6].mean(0), axis=1)
                   for i in range(n_sensors)]).max(0)
    w = max(int(0.25 * fs), 1); gs = np.convolve(gn, np.ones(w) / w, "same")
    above = gs > WALK_GYRO_DPS; n3 = int(3 * fs)
    for i in np.where(above)[0]:
        if above[i:i + n3].mean() >= WALK_ACTIVE_FRAC:
            return t[i]
    return t[0]


def static_window(t, imu, n_sensors, fs, onset):
    """Quietest window (lowest summed gyro SD over all sensors) between the start and onset - margin."""
    end = onset - STATIC_MARGIN_S
    L = min(STATIC_WIN_S, end - t[0] - 0.5)
    if L < 1.0:
        return None
    step = int(0.25 * fs); n = int(L * fs); best = None
    i_end = np.searchsorted(t, end)
    for i in range(np.searchsorted(t, t[0] + 0.5), i_end - n, step):
        sd = sum(imu[i:i + n, 6 * s + 3:6 * s + 6].std(0).sum() + 50 * imu[i:i + n, 6 * s:6 * s + 3].std(0).sum()
                 for s in range(n_sensors))
        if best is None or sd < best[0]:
            best = (sd, i)
    i = best[1]
    return t[i], t[i + n - 1]


def rot_between(u, v):
    """Smallest rotation matrix R with R @ u = v (u, v unit vectors), via Rodrigues' formula."""
    c = float(np.dot(u, v)); ax = np.cross(u, v); s = np.linalg.norm(ax)
    if s < 1e-12:
        return np.eye(3) if c > 0 else -np.eye(3)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]]) / s
    th = np.arctan2(s, c)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def calibrate_imu(imu, locs, mask, vertical="nearest"):
    """Per-sensor gyro bias removal, accelerometer scale to 1 g, and tilt alignment from a still window.
    Returns calibrated copy and a dict of parameters + before/after checks."""
    out = imu.copy(); cal = {}
    for i, loc in enumerate(locs):
        A = imu[mask, 6 * i:6 * i + 3]; G = imu[mask, 6 * i + 3:6 * i + 6]
        g0 = A.mean(0); mag = float(np.linalg.norm(g0)); u = g0 / mag
        bias = G.mean(0)
        if vertical == "z":
            ax_i, sign = 2, 1.0
        else:
            ax_i = int(np.argmax(np.abs(u))); sign = float(np.sign(u[ax_i]))
        e = np.zeros(3); e[ax_i] = sign
        R = rot_between(u, e)
        acc = (imu[:, 6 * i:6 * i + 3] / mag) @ R.T
        gyr = (imu[:, 6 * i + 3:6 * i + 6] - bias) @ R.T
        out[:, 6 * i:6 * i + 3] = acc; out[:, 6 * i + 3:6 * i + 6] = gyr
        cal[loc] = {
            "vertical_axis": ("+" if sign > 0 else "-") + "xyz"[ax_i],
            "gyro_bias_dps": np.round(bias, 3).tolist(),
            "acc_scale": round(1 / mag, 5),
            "tilt_correction_deg": round(float(np.degrees(np.arccos(np.clip(np.dot(u, e), -1, 1)))), 2),
            "rotation_matrix": np.round(R, 6).tolist(),
            "static_acc_before_g": np.round(g0, 4).tolist(), "static_acc_mag_before_g": round(mag, 4),
            "static_acc_after_g": np.round(acc[mask].mean(0), 4).tolist(),
            "static_gyro_after_dps": np.round(gyr[mask].mean(0), 3).tolist(),
            "static_gyro_sd_dps": np.round(G.std(0), 2).tolist(),
        }
    return out, cal


# ----------------------------------------------------------------------------- XDF reader
_FMT = {"float32": ("f", 4), "double64": ("d", 8), "int32": ("i", 4), "int16": ("h", 2),
        "int8": ("b", 1), "int64": ("q", 8)}


def _varlen(b, p):
    n = b[p]; p += 1
    return int.from_bytes(b[p:p + n], "little"), p + n


def load_xdf(fn):
    """Minimal XDF 1.0 reader. Returns {stream_id: dict(name, type, srate, labels, ts, data, offsets)}."""
    b = open(fn, "rb").read()
    if b[:4] != b"XDF:":
        raise ValueError(f"{fn} is not an XDF file")
    p, S = 4, {}
    while p < len(b):
        L, p = _varlen(b, p); end = p + L
        tag = struct.unpack("<H", b[p:p + 2])[0]; q = p + 2
        if tag in (2, 3, 4, 6):
            sid = struct.unpack("<I", b[q:q + 4])[0]; q += 4
        if tag == 2:                                   # stream header
            x = ET.fromstring(b[q:end].decode("utf-8", "replace"))
            S[sid] = dict(name=x.findtext("name"), type=x.findtext("type"),
                          srate=float(x.findtext("nominal_srate") or 0),
                          nch=int(x.findtext("channel_count")), fmt=x.findtext("channel_format"),
                          labels=[c.findtext("label") for c in x.iter("channel")],
                          ts=[], data=[], offsets=[])
        elif tag == 3:                                 # samples
            s = S[sid]; n, q = _varlen(b, q); c, sz = _FMT.get(s["fmt"], (None, 0))
            for _ in range(n):
                if b[q] == 8:
                    t = struct.unpack("<d", b[q + 1:q + 9])[0]; q += 9
                else:
                    t = None; q += 1
                if c:
                    v = struct.unpack("<%d%s" % (s["nch"], c), b[q:q + sz * s["nch"]]); q += sz * s["nch"]
                else:                                  # string stream
                    v = []
                    for _ in range(s["nch"]):
                        l, q = _varlen(b, q); v.append(b[q:q + l].decode("utf-8", "replace")); q += l
                s["ts"].append(t); s["data"].append(v)
        elif tag == 4:                                 # clock offset
            S[sid]["offsets"].append(struct.unpack("<dd", b[q:q + 16]))
        p = end
    for s in S.values():
        ts, dt = s["ts"], (1.0 / s["srate"] if s["srate"] else 0.0)
        for i in range(len(ts)):
            if ts[i] is None:
                ts[i] = ts[i - 1] + dt if i else 0.0
        s["ts"] = np.asarray(ts, float)
        try:
            s["data"] = np.asarray(s["data"], float)
        except (ValueError, TypeError):
            pass
        s["offsets"] = np.asarray(s["offsets"], float).reshape(-1, 2)
    return S


# ----------------------------------------------------------------------------- time handling
def apply_clock_offsets(s):
    """Map stream time onto the recording PC's clock using LabRecorder clock offsets (linear fit)."""
    o = s["offsets"]
    if len(o) == 0:
        return s["ts"].copy()
    if len(o) < 3:
        return s["ts"] + np.median(o[:, 1])
    tc, ov = o[:, 0], o[:, 1]
    keep = np.abs(ov - np.median(ov)) < 5 * (np.median(np.abs(ov - np.median(ov))) + 1e-6)
    a, c = np.polyfit(tc[keep], ov[keep], 1)
    return s["ts"] + (a * s["ts"] + c)


def dejitter(ts, srate, gap_samples=5):
    """Replace jittery timestamps by a linear fit within each gap-free segment."""
    if srate <= 0 or len(ts) < 10:
        return ts
    out = ts.copy(); d = np.diff(ts)
    breaks = np.where(d > gap_samples / srate)[0] + 1
    for seg in np.split(np.arange(len(ts)), breaks):
        if len(seg) > 10:
            a, c = np.polyfit(seg, ts[seg], 1); out[seg] = a * seg + c
    return out


def fsr_time_from_arduino(ts_lsl, t_ms):
    """Rebuild FSR time from the Arduino's own millisecond counter (removes serial jitter, keeps
    the LSL mean latency). Robust linear fit of LSL time vs Arduino time."""
    x = (t_ms - t_ms[0]) / 1000.0
    a, c = np.polyfit(x, ts_lsl, 1)
    for _ in range(3):
        r = ts_lsl - (a * x + c); k = np.abs(r - np.median(r)) < 4 * np.std(r) + 1e-9
        a, c = np.polyfit(x[k], ts_lsl[k], 1)
    drift_ppm = (1.0 / a - 1.0) * 1e6
    jitter_ms = 1000 * np.std(ts_lsl - (a * x + c))
    return a * x + c, drift_ppm, jitter_ms


# ----------------------------------------------------------------------------- QC helpers
def flat_segments(x, fs, min_s=0.5):
    """Total seconds where a signal is exactly constant for >= min_s."""
    same = np.r_[False, np.diff(x) == 0]
    tot, run = 0, 0
    for v in same:
        run = run + 1 if v else 0
        if run == int(min_s * fs):
            tot += run
        elif run > int(min_s * fs):
            tot += 1
    return tot / fs


def imu_qc(imu, locs, fs, expected):
    q = {"sensors_present": locs,
         "sensors_missing": [f"{v} (SID{k})" for k, v in expected.items() if v not in locs],
         "duplicate_sensor_pairs": [], "flat_seconds": {}, "max_abs_acc_g": {}, "max_abs_gyr_dps": {}}
    n = len(locs)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = imu[:, 6 * i:6 * i + 6], imu[:, 6 * j:6 * j + 6]
            if np.allclose(a, b) or np.mean([abs(np.corrcoef(a[:, k], b[:, k])[0, 1]) for k in range(6)]) > 0.995:
                q["duplicate_sensor_pairs"].append(f"{locs[i]} = {locs[j]}")
    for i, loc in enumerate(locs):
        q["flat_seconds"][loc] = round(max(flat_segments(imu[:, 6 * i + k], fs) for k in range(6)), 2)
        q["max_abs_acc_g"][loc] = round(float(np.abs(imu[:, 6 * i:6 * i + 3]).max()), 2)
        q["max_abs_gyr_dps"][loc] = round(float(np.abs(imu[:, 6 * i + 3:6 * i + 6]).max()), 1)
    return q


# ----------------------------------------------------------------------------- contact detection
def normalize(x):
    lo, hi = np.percentile(x, 5), np.percentile(x, 99)
    return (x - lo) / max(hi - lo, 1e-9)


def contacts(t, load):
    """Hysteresis threshold on a 0..1 foot-load signal -> list of (on_time, off_time), with
    sub-sample interpolation of the crossing times."""
    on = False; ev = []; start = None
    for i in range(1, len(load)):
        if not on and load[i] >= CONTACT_ON and load[i - 1] < CONTACT_ON:
            f = (CONTACT_ON - load[i - 1]) / (load[i] - load[i - 1]); start = t[i - 1] + f * (t[i] - t[i - 1]); on = True
        elif on and load[i] < CONTACT_OFF and load[i - 1] >= CONTACT_OFF:
            f = (load[i - 1] - CONTACT_OFF) / (load[i - 1] - load[i]); ev.append([start, t[i - 1] + f * (t[i] - t[i - 1])]); on = False
    return np.array(ev).reshape(-1, 2)


def clean_contacts(ev):
    if len(ev) < 4:
        return ev
    med = np.median(np.diff(ev[:, 0]))
    merged = [list(ev[0])]
    for a, b in ev[1:]:
        if a - merged[-1][1] < MIN_GAP_FRAC * med:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    ev = np.array(merged)
    med = np.median(np.diff(ev[:, 0]))
    return ev[(ev[:, 1] - ev[:, 0]) > MIN_CONTACT_FRAC * med]


def strides_from_contacts(ev, t_fsr, chans, names, win):
    """One stride = HS_k .. HS_k+1 with TO_k inside. Returns list of dicts with validity checks."""
    out = []
    if len(ev) < 3:
        return out
    st = np.diff(ev[:, 0]); mid = st[len(st) // 4: 3 * len(st) // 4]
    med = np.median(mid if len(mid) else st)
    typ_peak = {n: np.percentile(c, 99) - np.percentile(c, 5) for n, c in zip(names, chans)}
    for k in range(len(ev) - 1):
        hs, to, nhs = ev[k, 0], ev[k, 1], ev[k + 1, 0]
        T = nhs - hs; stance = 100 * (to - hs) / T
        m = (t_fsr >= hs) & (t_fsr < nhs)
        silent = [n for n, c in zip(names, chans)
                  if m.any() and (c[m].max() - np.percentile(c, 5)) < CHANNEL_RESPONSE * typ_peak[n]]
        reasons = []
        if abs(T - med) > STRIDE_TOL * med: reasons.append("stride_time")
        if not (STANCE_PCT_RANGE[0] <= stance <= STANCE_PCT_RANGE[1]): reasons.append("stance_pct")
        if hs < win[0] or nhs > win[1]: reasons.append("outside_trim")
        out.append(dict(hs=hs, to=to, next_hs=nhs, stride_s=T, stance_pct=stance,
                        valid=not reasons, reason=";".join(reasons), silent_fsr=";".join(silent)))
    return out


def label_leg(t, strides):
    stance = np.full(len(t), -1, int); pct = np.full(len(t), np.nan)
    hs_f = np.zeros(len(t), int); to_f = np.zeros(len(t), int)
    for s in strides:
        if not s["valid"]:
            continue
        m = (t >= s["hs"]) & (t < s["next_hs"])
        stance[m] = (t[m] < s["to"]).astype(int)
        pct[m] = 100 * (t[m] - s["hs"]) / s["stride_s"]
        for key, arr in (("hs", hs_f), ("to", to_f)):
            i = np.searchsorted(t, s[key]); i = min(max(i, 0), len(t) - 1)
            if i > 0 and abs(t[i - 1] - s[key]) < abs(t[i] - s[key]): i -= 1
            arr[i] = 1
    return stance, pct, hs_f, to_f


# ----------------------------------------------------------------------------- main per-trial routine
def process(fn, outdir, trim=None, vertical="nearest", do_cal=True, keep_static=False):
    name = os.path.splitext(os.path.basename(fn))[0]
    S = load_xdf(fn)
    imus = [s for s in S.values() if s["type"] == "IMU"]
    fsrs = [s for s in S.values() if s["type"] == "FSR"]
    if not imus or not fsrs:
        raise RuntimeError(f"{name}: needs one IMU and one FSR stream, found {[s['name'] for s in S.values()]}")
    imu_s = max(imus, key=lambda s: len(s["ts"])); fsr_s = max(fsrs, key=lambda s: len(s["ts"]))
    for st in (imu_s, fsr_s):
        if len(st["ts"]) < 100:
            raise RuntimeError(f"{name}: the {st['name']} stream has only {len(st['ts'])} samples - "
                               f"it was not recording (check that LabRecorder saw this stream)")
    qc = {"trial": name, "streams": [f"{s['name']} ({s['type']}, {len(s['ts'])} samples)" for s in S.values()]}
    os.makedirs(outdir, exist_ok=True)

    # --- time base
    t_imu = dejitter(apply_clock_offsets(imu_s), imu_s["srate"])
    fs_imu = 1 / np.median(np.diff(t_imu))
    fl = fsr_s["labels"]; fd = fsr_s["data"]
    t_fsr_lsl = apply_clock_offsets(fsr_s)
    if "arduino_t_ms" in fl:
        t_fsr, drift, jit = fsr_time_from_arduino(t_fsr_lsl, fd[:, fl.index("arduino_t_ms")])
        qc["fsr_clock"] = {"arduino_drift_ppm": round(drift), "serial_jitter_ms": round(jit, 2)}
    else:
        t_fsr = dejitter(t_fsr_lsl, fsr_s["srate"])
    qc["imu_rate_hz"] = round(fs_imu, 2)
    qc["imu_max_gap_ms"] = round(1000 * np.diff(t_imu).max(), 1)

    # --- IMU channels -> locations
    labels = imu_s["labels"]; imu = imu_s["data"]
    sids = list(dict.fromkeys(l.split("_")[1].replace("SID", "") for l in labels))
    locs = [SENSOR_MAP.get(s, f"SID{s}") for s in sids]
    imu_cols = [f"{SENSOR_MAP.get(l.split('_')[1].replace('SID', ''), l.split('_')[1])}_{'_'.join(l.split('_')[2:])}"
                for l in labels]
    qc["imu"] = imu_qc(imu, locs, fs_imu, SENSOR_MAP)

    # --- standing period: find walking onset and a still window, calibrate, then trim
    onset = walking_onset(t_imu, imu, len(locs), fs_imu)
    sw = static_window(t_imu, imu, len(locs), fs_imu, onset)
    qc["recording_start_to_walking_onset_s"] = round(onset - t_imu[0], 2)
    qc["static_window_s_from_recording_start"] = [round(float(sw[0] - t_imu[0]), 2), round(float(sw[1] - t_imu[0]), 2)] if sw else None
    if do_cal and sw:
        imu, cal = calibrate_imu(imu, locs, (t_imu >= sw[0]) & (t_imu <= sw[1]), vertical)
        qc["imu_calibration"] = cal
        with open(os.path.join(outdir, f"{name}_imu_calibration.json"), "w") as f:
            json.dump({"trial": name, "static_window_lsl_time": [float(x) for x in sw], "vertical_axis_mode": vertical,
                       "apply_as": "acc_cal = R @ (acc * acc_scale);  gyr_cal = R @ (gyr - gyro_bias_dps)",
                       "sensors": {k: {kk: v[kk] for kk in ("vertical_axis", "gyro_bias_dps", "acc_scale",
                                                             "rotation_matrix", "tilt_correction_deg")}
                                   for k, v in cal.items()}}, f, indent=2)
    else:
        qc["imu_calibration"] = "skipped" if not do_cal else "no still period long enough (>= 2 s) before walking"
    T0 = t_imu[0] if keep_static else onset
    if not keep_static:
        k = t_imu >= onset
        t_imu, imu = t_imu[k], imu[k]
        kf = t_fsr >= onset - 0.5          # FSR thresholds are computed from walking data only
        t_fsr, fd = t_fsr[kf], fd[kf]
    qc["duration_s"] = round(t_imu[-1] - T0, 1)
    qc["time_zero"] = "walking onset" if not keep_static else "recording start"

    # --- FSR QC and contacts per foot
    win = (T0 + trim[0], T0 + trim[1]) if trim else (t_imu[0], np.inf)
    qc["fsr"] = {}; legs = {}
    for foot, names in FSR_FEET.items():
        chans = [fd[:, fl.index(n)] for n in names]
        cq = {}
        for n, c in zip(names, chans):
            rng = np.percentile(c, 99) - np.percentile(c, 5)
            others = [(m, np.corrcoef(c, fd[:, fl.index(m)])[0, 1]) for m in fl if m not in (n, "arduino_t_ms")]
            m_best, r_best = max(others, key=lambda x: x[1])
            cq[n] = {"range_counts": round(float(rng), 1), "baseline": round(float(np.percentile(c, 5)), 1),
                     "most_similar": m_best, "r": round(float(r_best), 3),
                     "suspect": bool(rng < 15 or r_best > 0.98)}
        usable = [i for i, n in enumerate(names) if cq[n]["range_counts"] >= 15 and cq[n]["r"] <= 0.98]
        if not usable:
            detail = "; ".join(f"{n}: range {cq[n]['range_counts']} counts, r = {cq[n]['r']} with {cq[n]['most_similar']}"
                               for n in names)
            raise RuntimeError(f"no usable FSR channel on the {'left' if foot == 'L' else 'right'} foot "
                               f"(each needs range >= 15 counts and r <= 0.98 with every other channel). {detail}")
        load = np.max([normalize(chans[i]) for i in usable], axis=0)
        ev = clean_contacts(contacts(t_fsr, load))
        st = strides_from_contacts(ev, t_fsr, [chans[i] for i in usable], [names[i] for i in usable], win)
        # treadmill start/stop: keep only from the first to the last run of 3 consecutive valid strides
        v = np.array([s["valid"] for s in st])
        runs = [i for i in range(len(v) - 2) if v[i:i + 3].all()]
        if runs:
            for i, s in enumerate(st):
                if i < runs[0] or i > runs[-1] + 2:
                    if s["valid"]:
                        s["valid"] = False; s["reason"] = "startup_or_end"
        legs[foot] = st
        good = [s for s in st if s["valid"]]
        silent_counts = {}
        for s in good:
            for n in filter(None, s["silent_fsr"].split(";")):
                silent_counts[n] = silent_counts.get(n, 0) + 1
        qc["fsr"][foot] = {"channels": cq, "used_for_contact": [names[i] for i in usable],
                           "strides_detected": len(st), "strides_valid": len(good),
                           "stride_s_mean_sd": [round(np.mean([s["stride_s"] for s in good]), 3),
                                                round(np.std([s["stride_s"] for s in good]), 3)] if good else None,
                           "stance_pct_mean_sd": [round(np.mean([s["stance_pct"] for s in good]), 1),
                                                  round(np.std([s["stance_pct"] for s in good]), 1)] if good else None,
                           "labeled_window_s": [round(good[0]["hs"] - T0, 1), round(good[-1]["next_hs"] - T0, 1)] if good else None,
                           "valid_strides_with_silent_channel": silent_counts}

    # --- left/right sanity check: shank gyro should be quieter in stance than in swing
    chk = {}
    for foot, loc in (("L", "LeftShank"), ("R", "RightShank")):
        if loc in locs:
            i = locs.index(loc); g = np.linalg.norm(imu[:, 6 * i + 3:6 * i + 6], axis=1)
            stc, _, _, _ = label_leg(t_imu, legs[foot])
            if (stc == 1).any() and (stc == 0).any():
                chk[f"{foot}_shank_gyro_swing/stance_ratio"] = round(float(g[stc == 0].mean() / g[stc == 1].mean()), 2)
    qc["side_check"] = chk
    qc["side_check_note"] = "ratio > 1 means that foot's FSR stance matches its own shank IMU (expected ~2-4)"

    # --- labels on IMU time
    L_st, L_pct, L_hs, L_to = label_leg(t_imu, legs["L"])
    R_st, R_pct, R_hs, R_to = label_leg(t_imu, legs["R"])
    code = np.where((L_st >= 0) & (R_st >= 0), 2 * L_st + R_st, -1)
    lab = (code >= 0)
    qc["labeled_rows_pct"] = round(100 * lab.mean(), 1)
    qc["phase_code_share_pct"] = {k: round(100 * np.mean(code[lab] == v), 1) for k, v in
                                  (("11_double_support", 3), ("10_left_single", 2), ("01_right_single", 1), ("00_flight", 0))} if lab.any() else {}

    fsr_on_imu = np.column_stack([np.interp(t_imu, t_fsr, fd[:, fl.index(n)]) for n in FSR_FEET["L"] + FSR_FEET["R"]])

    # --- write CSVs
    os.makedirs(outdir, exist_ok=True)
    header = (["time_s", "lsl_time"] + imu_cols +
              ["L_stance", "R_stance", "phase_code", "L_gait_pct", "R_gait_pct", "L_HS", "L_TO", "R_HS", "R_TO"] +
              ["fsr_" + n for n in FSR_FEET["L"] + FSR_FEET["R"]])
    M = np.column_stack([t_imu - T0, t_imu, imu, L_st, R_st, code, L_pct, R_pct, L_hs, L_to, R_hs, R_to, fsr_on_imu])
    fmt = ["%.5f", "%.5f"] + ["%.6g"] * imu.shape[1] + ["%d"] * 3 + ["%.2f"] * 2 + ["%d"] * 4 + ["%.1f"] * 6
    np.savetxt(os.path.join(outdir, f"{name}_labeled.csv"), M, delimiter=",", header=",".join(header),
               comments="", fmt=fmt)
    with open(os.path.join(outdir, f"{name}_strides.csv"), "w") as f:
        f.write("leg,stride,hs_time_s,to_time_s,next_hs_time_s,stride_s,stance_pct,valid,reason,silent_fsr\n")
        for foot in ("L", "R"):
            for k, s in enumerate(legs[foot]):
                f.write(f"{foot},{k},{s['hs'] - T0:.4f},{s['to'] - T0:.4f},{s['next_hs'] - T0:.4f},"
                        f"{s['stride_s']:.4f},{s['stance_pct']:.2f},{int(s['valid'])},{s['reason']},{s['silent_fsr']}\n")
    with open(os.path.join(outdir, f"{name}_qc.json"), "w") as f:
        json.dump(qc, f, indent=2)
    qc_plot(name, outdir, t_imu - T0, imu, locs, t_fsr - T0, fd, fl, L_st, R_st, legs, T0)
    return qc


def qc_plot(name, outdir, t, imu, locs, tf, fd, fl, L_st, R_st, legs, T0):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(5, 1, figsize=(14, 13))
    for foot, a in zip(("L", "R"), ax[:2]):
        for n in FSR_FEET[foot]:
            a.plot(tf, fd[:, fl.index(n)], lw=0.6, label=n)
        for s in legs[foot]:
            a.axvspan(s["hs"] - T0, s["to"] - T0, color="g" if s["valid"] else "r", alpha=0.12, lw=0)
        a.set_title(f"{name}: {foot} foot FSRs (green = valid stance, red = rejected stride)"); a.legend(loc="upper right", fontsize=7)
    a = ax[2]
    for foot in ("L", "R"):
        st = legs[foot]; v = [s for s in st if s["valid"]]; x = [s for s in st if not s["valid"]]
        a.plot([s["hs"] - T0 for s in v], [s["stride_s"] for s in v], "o", ms=3, label=f"{foot} valid")
        a.plot([s["hs"] - T0 for s in x], [s["stride_s"] for s in x], "x", ms=4, label=f"{foot} rejected")
    a.set_ylabel("stride time (s)"); a.legend(fontsize=7); a.set_title("Stride times")
    for i, loc in enumerate(locs):
        ax[3].plot(t, np.linalg.norm(imu[:, 6 * i + 3:6 * i + 6], axis=1) + 0 * i, lw=0.4, label=loc)
    ax[3].set_title("IMU |gyro| (deg/s), whole trial"); ax[3].legend(fontsize=7, ncol=7)
    # zoom: 10 s in the middle with labels
    mid = t[len(t) // 2]; m = (t > mid) & (t < mid + 8)
    a = ax[4]
    for loc in ("LeftShank", "RightShank"):
        if loc in locs:
            i = locs.index(loc); g = imu[m, 6 * i + 3:6 * i + 6]; j = np.argmax(g.std(0))
            a.plot(t[m], g[:, j] / np.abs(g[:, j]).max(), lw=0.8, label=f"{loc} gyr (norm.)")
    a.plot(t[m], np.where(L_st[m] >= 0, L_st[m], np.nan) * 1.1 + 1.3, "k", lw=1.2, label="L_stance")
    a.plot(t[m], np.where(R_st[m] >= 0, R_st[m], np.nan) * 1.1 + 2.7, "m", lw=1.2, label="R_stance")
    a.set_title("8 s zoom: labels vs shank gyro"); a.legend(fontsize=7, ncol=4); a.set_xlabel("time (s)")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, f"{name}_qc.png"), dpi=80); plt.close(fig)


def summary_md(qcs, path):
    L = ["# Gait labeling QC summary\n"]
    L.append("| trial | labeled rows | L valid strides | R valid strides | stride (s) L / R | stance % L / R | missing IMUs | duplicate IMUs |")
    L.append("|---|---|---|---|---|---|---|---|")
    for q in qcs:
        f = q["fsr"]
        fmt = lambda v: f"{v[0]} ± {v[1]}" if v else "-"
        L.append(f"| {q['trial']} | {q['labeled_rows_pct']} % | {f['L']['strides_valid']}/{f['L']['strides_detected']} | "
                 f"{f['R']['strides_valid']}/{f['R']['strides_detected']} | {fmt(f['L']['stride_s_mean_sd'])} / {fmt(f['R']['stride_s_mean_sd'])} | "
                 f"{fmt(f['L']['stance_pct_mean_sd'])} / {fmt(f['R']['stance_pct_mean_sd'])} | "
                 f"{', '.join(q['imu']['sensors_missing']) or 'none'} | {', '.join(q['imu']['duplicate_sensor_pairs']) or 'none'} |")
    L.append("\n## FSR channel flags (suspect = range < 15 counts or r > 0.98 with another channel)\n")
    for q in qcs:
        for foot in ("L", "R"):
            for n, c in q["fsr"][foot]["channels"].items():
                if c["suspect"] or c["r"] > 0.9:
                    L.append(f"- {q['trial']} {n}: range {c['range_counts']}, r = {c['r']} with {c['most_similar']}"
                             f"{' (excluded)' if c['suspect'] else ' (kept)'}")
            s = q["fsr"][foot]["valid_strides_with_silent_channel"]
            if s:
                L.append(f"- {q['trial']} {foot} foot: channels silent in some valid strides: {s}")
    L.append("\n## Left/right check (shank |gyro| swing/stance ratio, expected > 1)\n")
    for q in qcs:
        L.append(f"- {q['trial']}: {q['side_check']}")
    L.append("\n## Standing period and IMU calibration\n")
    L.append("Standing period removed: each CSV starts at walking onset (time_s = 0). Calibration uses the quietest "
             "window of the standing period. 'before' is the raw still reading, 'after' the calibrated one.\n")
    L.append("| trial | sensor | vertical axis | gyro bias (deg/s) | still acc before (g) | |acc| before | tilt fix (deg) | still acc after (g) | still gyro after (deg/s) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for q in qcs:
        cal = q.get("imu_calibration")
        hdr = f"{q['trial']} (onset {q['recording_start_to_walking_onset_s']} s, still window {q['static_window_s_from_recording_start']} s)"
        if not isinstance(cal, dict):
            L.append(f"| {hdr} | - | {cal} | | | | | | |"); continue
        for j, (loc, c) in enumerate(cal.items()):
            L.append(f"| {hdr if j == 0 else ''} | {loc} | {c['vertical_axis']} | {c['gyro_bias_dps']} | {c['static_acc_before_g']} | "
                     f"{c['static_acc_mag_before_g']} | {c['tilt_correction_deg']} | {c['static_acc_after_g']} | {c['static_gyro_after_dps']} |")
    open(path, "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help=".xdf files or wildcards; default: every .xdf in this script's folder")
    ap.add_argument("--out", default=None, help="output folder (default: labeled_output next to the .xdf files)")
    ap.add_argument("--trim", action="append", default=[], help="NAME=START:END in seconds (repeatable)")
    ap.add_argument("--vertical-axis", choices=["nearest", "z"], default="nearest")
    ap.add_argument("--no-imu-cal", action="store_true")
    ap.add_argument("--keep-static", action="store_true")
    a = ap.parse_args()

    # Expand wildcards ourselves (Windows PowerShell/cmd pass "*.xdf" through literally).
    import glob
    files = []
    for f in (a.files or [os.path.join(os.path.dirname(os.path.abspath(__file__)), "*.xdf")]):
        hits = sorted(glob.glob(f)) if any(ch in f for ch in "*?[") else [f]
        files += [h for h in hits if h.lower().endswith(".xdf")]
    files = list(dict.fromkeys(os.path.abspath(f) for f in files))
    if not files:
        where = a.files or [os.path.dirname(os.path.abspath(__file__))]
        sys.exit(f"No .xdf files found in {where}. Put the .xdf files in the same folder as label_gait.py, "
                 f"or pass them on the command line.")
    out = a.out or os.path.join(os.path.dirname(files[0]), "labeled_output")
    print(f"Found {len(files)} .xdf file(s):"); [print("  ", os.path.basename(f)) for f in files]

    trims = {}
    for t in a.trim:
        k, v = t.split("="); s, e = v.split(":"); trims[k] = (float(s), float(e))
    qcs = []
    for fn in files:
        trim = next((v for k, v in trims.items() if k in os.path.basename(fn)), None)
        try:
            q = process(fn, out, trim, a.vertical_axis, not a.no_imu_cal, a.keep_static)
        except Exception as e:                      # keep going if one file is bad
            print(f"{os.path.basename(fn)}: FAILED - {type(e).__name__}: {e}")
            continue
        qcs.append(q)
        print(f"{q['trial']}: labeled {q['labeled_rows_pct']} % of rows, "
              f"valid strides L {q['fsr']['L']['strides_valid']} R {q['fsr']['R']['strides_valid']}")
    if qcs:
        summary_md(qcs, os.path.join(out, "QC_summary.md"))
    print(f"Wrote outputs to {out}")


if __name__ == "__main__":
    main()
