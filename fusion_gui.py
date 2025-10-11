#!/usr/bin/env python3
"""
fusion_gui.py
Single-file implementation of RK-LSI-EF Energy-efficient Fusion GUI module.

Features:
- Input: model prediction x_m and sensor measurement x_s (scalar or vector via comma-separated)
- Fusion math: residual, Mahalanobis, Huber-weight, exp decay, smoothing (alpha), P_min
- State machine: IDLE / ACTIVE / LEARNING with thresholds, hysteresis and anti-flap timers
- Logging to CSV
- Embedded matplotlib plots in Tkinter GUI (live updating)
- Simple simulator mode (for test) and manual input mode
- Configurable parameters are at top of script (easy to adjust)

Run: python fusion_gui.py
"""

import sys
import os
import time
import math
import json
import csv
import threading
from collections import deque
from datetime import datetime

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import matplotlib
# Use TkAgg when running locally with GUI. For headless demo, Agg can be used.
try:
    matplotlib.use('TkAgg')
except Exception:
    matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ----------------------
# CONFIG (editable)
# ----------------------
CONFIG = {
    "P_min": 0.05,
    "beta": 3.0,               # sensitivity for exp decay
    "alpha": 0.9,              # smoothing for P_m
    "huber_delta": 0.05,       # huber threshold (in same units as x)
    "eps_idle_enter": 0.01,    # relative to dynamic_range (will multiply later)
    "eps_idle_exit": 0.02,
    "eps_learn_enter": 0.05,
    "eps_learn_exit": 0.035,
    "s_idle": 0.005,           # std threshold for idle
    "s_learn_enter": 0.02,
    "tau_idle": 2.0,           # seconds required to confirm idle entry
    "tau_learn": 1.0,          # seconds required to confirm learn entry
    "T_learn_min": 5.0,        # minimum learning time in seconds
    "control_dt": 0.2,         # control loop period (s)
    "dynamic_range": 1.0,      # default normalization range; adjust per signal
    "log_file": "fusion_log.csv",
    "max_points": 400,
    "default_fusion_gain": 0.5
}

# ----------------------
# Utilities: parsing vectors
# ----------------------
def parse_vec(s):
    s = str(s).strip()
    if not s:
        return None
    if "," in s:
        parts = [float(p) for p in s.split(",")]
        return np.array(parts, dtype=float)
    else:
        return float(s)

def vec_to_list(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    else:
        return float(x)

# ----------------------
# Fusion Math
# ----------------------
def huber_weight(d, delta):
    # For scalar or vector (use norm for vector)
    if isinstance(d, np.ndarray):
        normd = np.linalg.norm(d)
    else:
        normd = abs(d)
    if normd <= delta:
        return 1.0
    else:
        return delta / normd

def mahalanobis_distance(r, S):
    # r: vector or scalar, S: covariance matrix or scalar variance
    if isinstance(r, np.ndarray):
        r = r.reshape(-1,1)
        if S is None:
            S = np.eye(len(r))
        Sinv = np.linalg.pinv(S)
        d2 = float(r.T.dot(Sinv).dot(r))
        return math.sqrt(max(d2, 0.0))
    else:
        var = S if (S is not None and not isinstance(S, np.ndarray)) else 1.0
        if var <= 0:
            var = 1.0
        return abs(r) / math.sqrt(var)

class FusionEngine:
    def __init__(self, cfg):
        self.cfg = cfg.copy()
        self.Pm = 0.9
        self.Ps = 0.1
        self.x_hat = 0.0
        self.d = 0.0
        self.prev_Pm = self.Pm

    def compute_confidences(self, x_model, x_sensor, S=None):
        # compute residual
        if isinstance(x_model, np.ndarray) or isinstance(x_sensor, np.ndarray):
            # ensure same shape
            xm = np.array(x_model, dtype=float)
            xs = np.array(x_sensor, dtype=float)
            r = xs - xm
            d = mahalanobis_distance(r, S)
        else:
            xm = float(x_model)
            xs = float(x_sensor)
            r = xs - xm
            d = mahalanobis_distance(r, S)

        # huber weight
        w = huber_weight(r, self.cfg["huber_delta"])

        # raw score (exp decay)
        beta = self.cfg["beta"]
        s = math.exp(-beta * d) * w  # in [0,1]

        # raw confidences
        Pm_raw = max(s, self.cfg["P_min"])
        Ps_raw = max(1.0 - s, self.cfg["P_min"])

        # normalize
        ssum = Pm_raw + Ps_raw
        Pm_raw /= ssum
        Ps_raw = 1.0 - Pm_raw

        # smoothing (inertia)
        alpha = self.cfg["alpha"]
        Pm = alpha * self.prev_Pm + (1.0 - alpha) * Pm_raw
        Pm = max(Pm, self.cfg["P_min"])
        Ps = max(1.0 - Pm, self.cfg["P_min"])

        self.prev_Pm = Pm
        self.Pm = Pm
        self.Ps = Ps
        self.d = d
        self.r = r
        return Pm, Ps, d, r

    def fuse(self, x_model, x_sensor, S=None, fusion_gain=None):
        Pm, Ps, d, r = self.compute_confidences(x_model, x_sensor, S)
        # Weighted average fusion (scalar or vector)
        if fusion_gain is None:
            fusion_gain = self.cfg.get("default_fusion_gain", 0.5)
        if isinstance(x_model, np.ndarray) or isinstance(x_sensor, np.ndarray):
            xm = np.array(x_model, dtype=float)
            xs = np.array(x_sensor, dtype=float)
            xhat = (Pm * xm + Ps * xs) / (Pm + Ps)
        else:
            xm = float(x_model)
            xs = float(x_sensor)
            xhat = (Pm * xm + Ps * xs) / (Pm + Ps)
        self.x_hat = xhat
        return xhat, Pm, Ps, d, r

# ----------------------
# State machine
# ----------------------
class ModeManager:
    def __init__(self, cfg):
        self.cfg = cfg.copy()
        self.state = "ACTIVE"  # start active
        self.t_enter = time.time()
        self.last_transition_time = time.time()
        self.idle_hold_start = None
        self.learn_hold_start = None
        self.learning_start_time = None

    def update_metrics(self, window_d):
        if len(window_d) == 0:
            return 0.0, 0.0
        arr = np.array(window_d, dtype=float)
        return float(np.mean(arr)), float(np.std(arr))

    def should_enter_idle(self, d_mean, d_std, delta_rate):
        c = self.cfg
        cond = (d_mean < c["eps_idle_enter"] and d_std < c["s_idle"] and delta_rate <  c["s_idle"])
        return cond

    def should_enter_learning(self, d_mean, d_std):
        c = self.cfg
        cond = (d_mean > c["eps_learn_enter"] or d_std > c["s_learn_enter"])
        return cond

    def step(self, d_mean, d_std, delta_rate):
        now = time.time()
        c = self.cfg
        prev = self.state

        if self.state == "ACTIVE":
            if self.should_enter_idle(d_mean, d_std, delta_rate):
                if self.idle_hold_start is None:
                    self.idle_hold_start = now
                elif now - self.idle_hold_start >= c["tau_idle"]:
                    self.state = "IDLE"
                    self.last_transition_time = now
                    self.idle_hold_start = None
            else:
                self.idle_hold_start = None

            if self.should_enter_learning(d_mean, d_std):
                if self.learn_hold_start is None:
                    self.learn_hold_start = now
                elif now - self.learn_hold_start >= c["tau_learn"]:
                    self.state = "LEARNING"
                    self.learning_start_time = now
                    self.last_transition_time = now
                    self.learn_hold_start = None

        elif self.state == "IDLE":
            if not (d_mean < c["eps_idle_exit"] and d_std < c["s_idle"] and delta_rate < c["s_idle"]):
                self.state = "ACTIVE"
                self.last_transition_time = now

        elif self.state == "LEARNING":
            if self.learning_start_time is None:
                self.learning_start_time = now
            if now - self.learning_start_time >= c["T_learn_min"]:
                if d_mean < c["eps_learn_exit"] and d_std < c["s_learn_enter"]:
                    self.state = "ACTIVE"
                    self.last_transition_time = now
                    self.learning_start_time = None
                else:
                    pass

        return self.state

# ----------------------
# Logger
# ----------------------
class DataLogger:
    def __init__(self, filename):
        self.filename = filename
        with open(self.filename, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp","xm","xs","xhat","Pm","Ps","d","state"])

    def log(self, xm, xs, xhat, Pm, Ps, d, state):
        with open(self.filename, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().isoformat(), json.dumps(vec_to_list(xm)), json.dumps(vec_to_list(xs)), json.dumps(vec_to_list(xhat)), Pm, Ps, d, state])

# ----------------------
# Plotter (matplotlib embedded in tkinter)
# ----------------------
class Plotter:
    def __init__(self, parent, max_points=400):
        self.max_points = max_points
        self.times = deque(maxlen=max_points)
        self.xm = deque(maxlen=max_points)
        self.xs = deque(maxlen=max_points)
        self.xhat = deque(maxlen=max_points)
        self.Pm = deque(maxlen=max_points)
        self.Ps = deque(maxlen=max_points)
        self.d = deque(maxlen=max_points)
        self.state = deque(maxlen=max_points)

        self.fig = Figure(figsize=(8,6))
        self.ax1 = self.fig.add_subplot(311)
        self.ax2 = self.fig.add_subplot(312)
        self.ax3 = self.fig.add_subplot(313)
        self.fig.tight_layout(pad=2.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

    def update(self, t, xm, xs, xhat, Pm, Ps, d, state):
        self.times.append(t)
        self.xm.append(vec_to_list(xm))
        self.xs.append(vec_to_list(xs))
        self.xhat.append(vec_to_list(xhat))
        self.Pm.append(Pm)
        self.Ps.append(Ps)
        self.d.append(d)
        self.state.append(state)

    def redraw(self):
        self.ax1.clear(); self.ax2.clear(); self.ax3.clear()
        self.ax1.set_title("States (model, sensor, fused)")
        self.ax1.plot(self.times, self.xm, label='x_m'); self.ax1.plot(self.times, self.xs, label='x_s'); self.ax1.plot(self.times, self.xhat, label='x_hat')
        self.ax1.legend(loc='upper right')
        self.ax2.set_title("Residual d(t)")
        self.ax2.plot(self.times, self.d, label='d'); self.ax2.legend(loc='upper right')
        self.ax3.set_title("Confidences (P_m, P_s) and Mode")
        self.ax3.plot(self.times, self.Pm, label='P_m'); self.ax3.plot(self.times, self.Ps, label='P_s')
        for i,t in enumerate(self.times):
            if self.state[i] == 'IDLE':
                self.ax3.axvspan(t - 0.001, t + 0.001, color='green', alpha=0.05)
            elif self.state[i] == 'LEARNING':
                self.ax3.axvspan(t - 0.001, t + 0.001, color='red', alpha=0.05)
        self.ax3.legend(loc='upper right')
        self.canvas.draw_idle()

# ----------------------
# GUI Application
# ----------------------
class FusionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RK-LSI-EF Fusion Module")
        self.root.geometry("1000x700")

        self.cfg = CONFIG.copy()
        dr = self.cfg["dynamic_range"]
        self.cfg["eps_idle_enter"] = self.cfg["eps_idle_enter"] * dr
        self.cfg["eps_idle_exit"] = self.cfg["eps_idle_exit"] * dr
        self.cfg["eps_learn_enter"] = self.cfg["eps_learn_enter"] * dr
        self.cfg["eps_learn_exit"] = self.cfg["eps_learn_exit"] * dr
        self.cfg["s_idle"] = self.cfg["s_idle"] * dr
        self.cfg["s_learn_enter"] = self.cfg["s_learn_enter"] * dr

        self.engine = FusionEngine(self.cfg)
        self.state_manager = ModeManager(self.cfg)
        self.logger = DataLogger(self.cfg["log_file"])
        top = ttk.Frame(self.root); top.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        bottom = ttk.Frame(self.root); bottom.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        ttk.Label(top, text="Model prediction x_m:").grid(row=0, column=0, sticky=tk.W)
        self.xm_entry = ttk.Entry(top, width=20); self.xm_entry.grid(row=0, column=1, padx=3)
        ttk.Label(top, text="Sensor x_s:").grid(row=0, column=2, sticky=tk.W)
        self.xs_entry = ttk.Entry(top, width=20); self.xs_entry.grid(row=0, column=3, padx=3)
        ttk.Button(top, text="Apply", command=self.manual_apply).grid(row=0, column=4, padx=6)

        self.run_flag = False
        self.start_btn = ttk.Button(top, text="Start Stream", command=self.start_stream); self.start_btn.grid(row=1, column=0, pady=6)
        self.stop_btn = ttk.Button(top, text="Stop", command=self.stop_stream); self.stop_btn.grid(row=1, column=1, pady=6)
        ttk.Button(top, text="Export Log", command=self.export_log).grid(row=1, column=2, pady=6)
        ttk.Button(top, text="Simulate Step", command=self.simulate_step).grid(row=1, column=3, pady=6)

        self.mode_var = tk.StringVar(value="ACTIVE")
        ttk.Label(top, text="Mode:").grid(row=2, column=0, sticky=tk.W)
        ttk.Label(top, textvariable=self.mode_var, foreground='blue').grid(row=2, column=1, sticky=tk.W)

        self.pm_var = tk.StringVar(value="Pm: -")
        self.ps_var = tk.StringVar(value="Ps: -")
        self.d_var = tk.StringVar(value="d: -")
        ttk.Label(top, textvariable=self.pm_var).grid(row=2, column=2)
        ttk.Label(top, textvariable=self.ps_var).grid(row=2, column=3)
        ttk.Label(top, textvariable=self.d_var).grid(row=2, column=4)

        self.plotter = Plotter(bottom, max_points=self.cfg["max_points"])

        self.d_buffer = deque(maxlen=int(2.0 / self.cfg["control_dt"]) * 10 + 10)
        self.delta_x_buffer = deque(maxlen=int(2.0 / self.cfg["control_dt"]) * 10 + 10)

        self.thread = None
        self.sim_t = 0.0
        self.sim_mode = False
        self.prev_xhat = 0.0

    def manual_apply(self):
        xm = parse_vec(self.xm_entry.get())
        xs = parse_vec(self.xs_entry.get())
        if xm is None or xs is None:
            messagebox.showwarning("Input missing", "Please supply both x_m and x_s.")
            return
        xhat, Pm, Ps, d, r = self.engine.fuse(xm, xs)
        self._post_update(xm, xs, xhat, Pm, Ps, d)

    def start_stream(self):
        if self.run_flag:
            return
        self.run_flag = True
        self.sim_mode = False
        self.thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.thread.start()

    def stop_stream(self):
        self.run_flag = False

    def simulate_step(self):
        self.sim_mode = True
        def sim():
            for k in range(60):
                if not self.sim_mode:
                    break
                t = k * self.cfg["control_dt"]
                xm = math.sin(0.1 * t)
                xs = xm + (0.0 if t < 3.0 else 0.3) + np.random.normal(scale=0.02)
                xhat, Pm, Ps, d, r = self.engine.fuse(xm, xs)
                self._post_update(xm, xs, xhat, Pm, Ps, d)
                time.sleep(self.cfg["control_dt"])
        threading.Thread(target=sim, daemon=True).start()

    def export_log(self):
        fname = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files","*.csv")])
        if not fname:
            return
        try:
            with open(self.cfg["log_file"], "rb") as fsrc, open(fname, "wb") as fdst:
                fdst.write(fsrc.read())
            messagebox.showinfo("Export", f"Log exported to {fname}")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    def _stream_loop(self):
        while self.run_flag:
            try:
                xm_val = self.xm_entry.get().strip()
                xs_val = self.xs_entry.get().strip()
                if xm_val and xs_val:
                    xm = parse_vec(xm_val)
                    xs = parse_vec(xs_val)
                else:
                    t = self.sim_t
                    xm = math.sin(0.5 * t)
                    xs = xm + np.random.normal(scale=0.03)
                    if int(t) % 20 == 0 and (t - int(t)) < self.cfg["control_dt"]:
                        xs += 0.4
                    self.sim_t += self.cfg["control_dt"]

                xhat, Pm, Ps, d, r = self.engine.fuse(xm, xs)
                self._post_update(xm, xs, xhat, Pm, Ps, d)
            except Exception as e:
                print("Stream error:", e, file=sys.stderr)
            time.sleep(self.cfg["control_dt"])

    def _post_update(self, xm, xs, xhat, Pm, Ps, d):
        self.d_buffer.append(d)
        delta_x = abs(xhat - getattr(self, "prev_xhat", 0.0))
        self.delta_x_buffer.append(delta_x)
        self.prev_xhat = xhat

        d_mean = float(np.mean(self.d_buffer)) if len(self.d_buffer) > 0 else 0.0
        d_std = float(np.std(self.d_buffer)) if len(self.d_buffer) > 0 else 0.0
        delta_rate = float(np.mean(self.delta_x_buffer)) if len(self.delta_x_buffer) > 0 else 0.0

        state = self.state_manager.step(d_mean, d_std, delta_rate)
        self.mode_var.set(state)
        self.pm_var.set(f"Pm: {Pm:.3f}")
        self.ps_var.set(f"Ps: {Ps:.3f}")
        self.d_var.set(f"d: {d:.4f} (mean {d_mean:.4f})")

        tnow = time.time()
        self.logger.log(xm, xs, xhat, Pm, Ps, d, state)
        self.plotter.update(tnow, xm, xs, xhat, Pm, Ps, d, state)
        try:
            self.plotter.redraw()
        except Exception as e:
            pass

    def on_close(self):
        self.run_flag = False
        self.root.quit()

def main():
    root = tk.Tk()
    app = FusionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
