# adaptive_rk_interactive.py
from dataclasses import dataclass, field
import numpy as np
import math
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import threading
import time
import json
import csv
import os
from typing import List, Dict, Any, Optional


# Config

class Config:
    tol_abs: float = 1e-6
    tol_rel: float = 1e-6
    h_min: float = 1e-6
    h_max: float = 0.05
    safety: float = 0.9
    K_map: float = 1.0
    eta: float = 1.0
    lambda_M: float = 0.3
    r_base: float = 2.0
    mu_up: float = 0.05
    mu_down: float = 0.001
    eps_S: float = 1e-12
    window_N: int = 50
    dt_sampling: float = 0.01
    max_retries: int = 8
    M_cap: float = 50.0
    beta_softmax: float = 1.0
    default_weights: Dict[str, float] = field(default_factory=lambda: {
        'e':0.6, 'delta':0.8, 'r':0.6, 'f':0.4, 'c':0.2, 'm':0.3,
        'D':0.2, 'v':0.15, 'a':0.1, 'u_rms':0.2, 'u_rate':0.5
    })

def clamp(x, lo, hi):
    return max(lo, min(hi, x))


# TF -> state-space 

def tf_to_ss(num: List[float], den: List[float]):
    an = np.asarray(den, dtype=float)
    bn = np.asarray(num, dtype=float)
    if an.size == 0:
        raise ValueError("Denominator empty.")
    if abs(an[0]) < 1e-20:
        raise ValueError("Leading denominator coefficient is zero.")
    bn = bn / an[0]
    an = an / an[0]
    n = len(an) - 1
    A = np.zeros((n, n), dtype=float)
    if n > 0:
        A[0, :] = -an[1:]
        if n > 1:
            A[1:, 0:-1] = np.eye(n-1)
    B = np.zeros((n, 1), dtype=float) if n > 0 else np.zeros((0,1))
    if n > 0:
        B[0,0] = 1.0
    m = len(bn) - 1
    if m < n:
        bn = np.concatenate([np.zeros(n - m), bn])
    elif m > n:
        q, r = np.polydiv(bn, an)
        bn = r
    if n == 0:
        A = np.zeros((0,0))
        B = np.zeros((0,1))
        C = np.zeros((1,0))
        D = float(bn[-1]) if len(bn) > 0 else 0.0
        return A, B, C, D
    bn = np.asarray(bn, dtype=float)
    C = (bn[1:] - bn[0] * an[1:]).reshape(1, -1)
    D = float(bn[0])
    return A, B, C, D


# Cash-Karp RK45

class CashKarp45:
    def __init__(self, dynamics):
        self.f = dynamics
        self.c = np.array([0.0, 1/5, 3/10, 3/5, 1.0, 7/8], dtype=float)
        self.a = np.zeros((6,6), dtype=float)
        self.a[1,0] = 1/5
        self.a[2,0:2] = np.array([3/40, 9/40])
        self.a[3,0:3] = np.array([3/10, -9/10, 6/5])
        self.a[4,0:4] = np.array([-11/54, 5/2, -70/27, 35/27])
        self.a[5,0:5] = np.array([1631/55296, 175/512, 575/13824, 44275/110592, 253/4096])
        self.b_high = np.array([37/378, 0.0, 250/621, 125/594, 0.0, 512/1771], dtype=float)
        self.b_low  = np.array([2825/27648, 0.0, 18575/48384, 13525/55296, 277/14336, 1/4], dtype=float)

    def step(self, t, x, u, h, params=None):
        x = np.asarray(x, dtype=float)
        k = []
        for i in range(6):
            ti = t + self.c[i] * h
            xi = x.copy()
            if i > 0:
                inc = np.zeros_like(x)
                for j in range(i):
                    inc += self.a[i,j] * k[j]
                xi = x + h * inc
            fi = self.f(ti, xi, u)
            k.append(fi)
        x_high = x + h * sum(self.b_high[i] * k[i] for i in range(6))
        x_low  = x + h * sum(self.b_low[i]  * k[i] for i in range(6))
        e_vec = x_high - x_low
        e_est = float(np.linalg.norm(e_vec, ord=2))
        return x_high, x_low, e_est


# PID controller

class PID:
    def __init__(self, Kp=0.0, Ki=0.0, Kd=0.0, anti_windup=True):
        self.Kp = float(Kp); self.Ki = float(Ki); self.Kd = float(Kd)
        self.I = 0.0
        self.last_err = None
        self.anti_windup = anti_windup

    def reset(self):
        self.I = 0.0
        self.last_err = None

    def control(self, t, y, ref, dt):
        err = ref - y
        if self.last_err is None:
            d = 0.0
        else:
            d = (err - self.last_err) / max(dt, 1e-12)
        self.I += err * dt
        if self.anti_windup:
            self.I = clamp(self.I, -1e6, 1e6)
        u = self.Kp * err + self.Ki * self.I + self.Kd * d
        self.last_err = err
        return float(u)


# MetricsWindow (support v and a)

class MetricsWindow:
    def __init__(self, N:int, dt:float):
        self.N = max(2, int(N))
        self.dt = dt
        self.u_hist: List[float] = []
        self.delta_hist: List[float] = []
        self.v_hist: List[float] = []
        self.a_hist: List[float] = []
        self.last_u = None
        self.last_v = None

    def push(self, u: float):
        mag = float(abs(u))
        if self.last_u is None:
            self.delta_hist.append(0.0)
            self.v_hist.append(0.0)
            self.a_hist.append(0.0)
        else:
            delta = abs(mag - self.last_u)
            self.delta_hist.append(delta)
            v = (mag - self.last_u) / max(self.dt, 1e-12)
            self.v_hist.append(v)
            if self.last_v is None:
                a = 0.0
            else:
                a = (v - self.last_v) / max(self.dt, 1e-12)
            self.a_hist.append(a)
            self.last_v = v
        self.last_u = mag
        self.u_hist.append(mag)
        # trim to window
        if len(self.u_hist) > self.N: self.u_hist.pop(0)
        if len(self.delta_hist) > self.N: self.delta_hist.pop(0)
        if len(self.v_hist) > self.N: self.v_hist.pop(0)
        if len(self.a_hist) > self.N: self.a_hist.pop(0)

    def current_delta(self):
        return float(self.delta_hist[-1]) if self.delta_hist else 0.0
    def delta_rms(self):
        a = np.asarray(self.delta_hist) if self.delta_hist else np.array([0.0])
        return float(np.sqrt(np.mean(a**2)))
    def delta_cum(self):
        a = np.asarray(self.delta_hist) if self.delta_hist else np.array([0.0])
        return float(np.sum(np.abs(a)))
    def delta_max(self):
        a = np.asarray(self.delta_hist) if self.delta_hist else np.array([0.0])
        return float(np.max(np.abs(a)))
    def delta_freq(self, thr: float):
        a = np.asarray(self.delta_hist) if self.delta_hist else np.array([0.0])
        if a.size == 0: return 0.0
        count = float(np.sum(a > thr))
        return count / (len(a) * self.dt) if (len(a) * self.dt) > 0 else 0.0
    def duty_cycle(self, eps: float):
        u = np.asarray(self.u_hist) if self.u_hist else np.array([0.0])
        if u.size == 0: return 0.0
        return float(np.sum(np.abs(u) > eps)) / float(u.size)
    def v_rms(self):
        a = np.asarray(self.v_hist) if self.v_hist else np.array([0.0])
        return float(np.sqrt(np.mean(a**2)))
    def a_rms(self):
        a = np.asarray(self.a_hist) if self.a_hist else np.array([0.0])
        return float(np.sqrt(np.mean(a**2)))
    def u_rms(self):
        a = np.asarray(self.u_hist) if self.u_hist else np.array([0.0])
        return float(np.sqrt(np.mean(a**2)))


# Normalizer & Aggregator

class Normalizer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        # initialize S values for metrics (EMA baseline)
        self.S = {k: max(1e-6, 1.0) for k in cfg.default_weights.keys()}
        self.S['e'] = 1e-3

    def observe_and_update(self, key: str, v: float):
        v = abs(float(v))
        S_old = self.S.get(key, self.cfg.eps_S)
        mu = self.cfg.mu_up if v > S_old else self.cfg.mu_down
        self.S[key] = max(self.cfg.eps_S, (1.0 - mu) * S_old + mu * v)

    def normalize(self, key: str, v: float):
        Sval = max(self.S.get(key, self.cfg.eps_S), self.cfg.eps_S)
        til = abs(v) / Sval
        return float(min(til, self.cfg.M_cap))

class Aggregator:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def softmax_weights(self, weights: Dict[str, float]):
        keys = list(weights.keys())
        vals = np.array([weights[k] for k in keys], dtype=float)
        beta = float(self.cfg.beta_softmax)
        ex = np.exp(beta * (vals - np.max(vals)))
        w = ex / (np.sum(ex) + 1e-16)
        return dict(zip(keys, w))

    def compute_M(self, tilde: Dict[str, float], weights: Dict[str, float], tilde_d: float = 0.0):
        w_norm = self.softmax_weights(weights)
        M_L = 0.0
        for k, v in tilde.items():
            w = float(w_norm.get(k, 0.0))
            M_L += w * v
        Gamma = 1.0 + self.cfg.eta * tilde_d
        M_raw = Gamma * M_L
        M = max(0.0, min(M_raw, self.cfg.M_cap))
        return M, M_raw, M_L, Gamma, w_norm


# StepAdapter (adaptive rate limiting)

class StepAdapter:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.M_filt = 0.0
        self.h_last = None
        self.prev_M = 0.0

    def classical_h_rk(self, h_old, e_est, x, x_high, p=4):
        tol_vec = self.cfg.tol_abs + self.cfg.tol_rel * np.maximum(np.abs(x), np.abs(x_high))
        scale = np.linalg.norm(tol_vec, ord=2) + 1e-16
        if e_est <= 0:
            return float(min(h_old * self.cfg.r_base, self.cfg.h_max))
        exponent = 1.0 / float(p + 1.0)
        h_new = h_old * ((self.cfg.safety * scale) / (e_est + 1e-16))**exponent
        return float(clamp(h_new, self.cfg.h_min, self.cfg.h_max))

    def rational_map(self, h_old, M):
        return float(clamp((self.cfg.K_map / (1.0 + M + 1e-16)) * h_old, self.cfg.h_min, self.cfg.h_max))

    def exp_map(self, h_old, M):
        return float(clamp(h_old * math.exp(-self.cfg.eta * M), self.cfg.h_min, self.cfg.h_max))

    def lowpass_M(self, M_new):
        lam = max(0.0, min(1.0, self.cfg.lambda_M))
        self.M_filt = (1.0 - lam) * self.M_filt + lam * M_new
        return self.M_filt

    def adaptive_rate_limit(self, h_candidate, h_old, M):
        dM = abs(M - self.prev_M)
        lower = max(self.cfg.h_min, h_old * (1.0 / (1.0 + 2.0 * self.M_filt)))
        upper = min(self.cfg.h_max, h_old * (1.0 + 0.5 / (1.0 + 10.0 * dM)))
        self.prev_M = M
        return float(clamp(h_candidate, lower, upper))

    def adapt(self, h_old, e_est, x, x_high, M):
        h_rk = self.classical_h_rk(h_old, e_est, x, x_high)
        h1 = self.rational_map(h_old, M)
        h2 = self.exp_map(h_old, M)
        h_cand = min(h_rk, h1, h2)
        h_cand = self.adaptive_rate_limit(h_cand, h_old, M)
        if self.h_last is None:
            self.h_last = h_cand
        else:
            alpha = 0.2
            self.h_last = alpha * h_cand + (1 - alpha) * self.h_last
        return float(self.h_last), h_rk


# AdaptiveRKSimulator with CSV logging & interactive callback

class AdaptiveRKSimulator:
    def __init__(self, A, B, C, D, cfg: Config, csv_path: str = None, weights: Dict[str, float] = None):
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float).reshape(-1,1) if np.asarray(B).size else np.asarray(B)
        self.C = np.asarray(C, dtype=float).reshape(1,-1) if np.asarray(C).size else np.asarray(C)
        self.D = float(D)
        self.n = self.A.shape[0] if self.A.size else 0
        self.cfg = cfg
        self.rk = CashKarp45(self.dynamics)
        self.metrics = MetricsWindow(cfg.window_N, cfg.dt_sampling)
        self.norm = Normalizer(cfg)
        self.agg = Aggregator(cfg)
        self.adapter = StepAdapter(cfg)
        self.csv_path = csv_path or 'adaptive_parameters_log.csv'
        self.weights = weights if weights is not None else dict(cfg.default_weights)
        self.csv_lock = threading.Lock()
        self._open_csv_if_needed()

    def _open_csv_if_needed(self):
        # ensure directory exists
        base_dir = os.path.dirname(os.path.abspath(self.csv_path))
        if not os.path.isdir(base_dir):
            try:
                os.makedirs(base_dir, exist_ok=True)
            except Exception:
                pass
        file_exists = os.path.exists(self.csv_path)
        # open for append
        self.csv_file = open(self.csv_path, 'a', newline='')
        # prepare header dynamically (includes weight_<k> columns)
        weight_keys = list(self.weights.keys())
        header = [
            't', 'h_prev', 'h', 'h_rk', 'y', 'u', 'x_json', 'e_est', 'e_norm',
            'delta', 'r', 'c', 'm', 'f', 'D',
            'u_rms', 'u_rate', 'v_rms', 'a_rms',
            # tilde fields
            'tilde_delta', 'tilde_r', 'tilde_f', 'tilde_c', 'tilde_m', 'tilde_D', 'tilde_v', 'tilde_a', 'tilde_u_rms', 'tilde_u_rate', 'tilde_e',
            'M_raw', 'M_filt', 'M_L', 'Gamma'
        ]
        # add raw weight columns and normalized weights JSON
        for k in weight_keys:
            header.append(f'weight_{k}')
        header += ['weights_norm_json', 'poles_cont_json', 'poles_disc_json',
                   # interactive related
                   'interactive', 'interactive_u', 'interactive_x_json', 'interactive_auto']
        self.csv_writer = csv.writer(self.csv_file)
        should_write = False
        if not file_exists:
            should_write = True
        else:
            try:
                if os.path.getsize(self.csv_path) == 0:
                    should_write = True
            except Exception:
                should_write = True
        if should_write:
            self.csv_writer.writerow(header)
            self.csv_file.flush()

    def close_csv(self):
        try:
            self.csv_file.close()
        except Exception:
            pass

    def dynamics(self, t, x, u, params=None):
        x = np.asarray(x, dtype=float)
        if self.n == 0:
            return np.zeros(0)
        return (self.A.dot(x) + (self.B.flatten() * float(u)))

    def output(self, x, u):
        y_arr = self.C.dot(x)
        try:
            y0 = float(y_arr.item())
        except Exception:
            y0 = float(np.asarray(y_arr).ravel()[0])
        return y0 + self.D * float(u)

    def _continuous_poles(self):
        if self.n == 0:
            return np.array([])
        return np.linalg.eigvals(self.A)

    def _discrete_poles_for_h(self, h):
        lam = self._continuous_poles()
        if lam.size == 0:
            return np.array([])
        return np.exp(lam * h)

    def simulate(self, x0: np.ndarray, pid: PID, ref_func, h0: float, Tfinal: float,
                 progress_callback=None, meta_save=True, ref_sequence: Optional[List[float]] = None,
                 interactive_cb: Optional[callable] = None, stop_event: Optional[threading.Event] = None):
        # Save metadata JSON at simulation start for reproducibility
        if meta_save:
            meta = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                'csv_path': os.path.abspath(self.csv_path),
                'A_shape': list(self.A.shape),
                'B_shape': list(self.B.shape),
                'C_shape': list(self.C.shape),
                'D': self.D,
                'cfg': self.cfg.__dict__,
                'weights': self.weights,
                'ref_sequence': ref_sequence
            }
            meta_path = self.csv_path + '.meta.json'
            try:
                with open(meta_path, 'w') as mf:
                    json.dump(meta, mf, indent=2)
            except Exception:
                pass

        t = 0.0
        x = np.asarray(x0, dtype=float).reshape(-1)
        if self.n == 0:
            raise RuntimeError("Zero-order system not supported in simulate.")
        h = float(clamp(h0, self.cfg.h_min, self.cfg.h_max))
        history = {'t': [], 'x': [], 'y': [], 'u': [], 'h': [], 'M': [], 'poles_cont': [], 'poles_disc': []}
        iter_count = 0
        max_iters = int(1e6)
        last_u = 0.0
        # main loop
        while t < Tfinal and iter_count < max_iters:
            if stop_event is not None and stop_event.is_set():
                break
            # measure using last applied u (ZOH)
            y = self.output(x, last_u)
            # decide reference: prefer ref_sequence (one per accepted step) else call ref_func(t)
            if ref_sequence is not None:
                idx = len(history['t'])  # index for next accepted step
                ref = ref_sequence[min(idx, len(ref_sequence)-1)]
            else:
                ref = ref_func(t)
            # compute control (uses current step size h)
            u = pid.control(t, y, ref, h)
            retries = 0
            h_try = h
            h_prev = h
            while True:
                if stop_event is not None and stop_event.is_set():
                    break
                x_high, x_low, e_est = self.rk.step(t, x, u, h_try)
                # update metrics (based on applied u)
                self.metrics.push(u)
                raw = {
                    'delta': self.metrics.current_delta(),
                    'r': self.metrics.delta_rms(),
                    'f': self.metrics.delta_freq(thr=1e-6),
                    'c': self.metrics.delta_cum(),
                    'm': self.metrics.delta_max(),
                    'D': self.metrics.duty_cycle(eps=1e-6),
                    'v': self.metrics.v_rms(),
                    'a': self.metrics.a_rms(),
                    'u_rms': self.metrics.u_rms(),
                    'u_rate': (abs(self.metrics.u_hist[-1] - self.metrics.u_hist[-2]) / self.metrics.dt) if len(self.metrics.u_hist) > 1 else 0.0
                }
                # update normalizer scales
                for k, val in raw.items():
                    self.norm.observe_and_update(k, val)
                self.norm.observe_and_update('e', e_est)
                # tilde normalization
                tilde = {k: self.norm.normalize(k, raw[k]) for k in ['delta', 'r', 'f', 'c', 'm', 'D', 'v', 'a', 'u_rms', 'u_rate']}
                tilde['e'] = self.norm.normalize('e', e_est)
                tilde_d = 0.0
                M, M_raw, M_L, Gamma, w_norm = self.agg.compute_M(tilde, self.weights, tilde_d)
                M_filt = self.adapter.lowpass_M(M)
                h_new, h_rk = self.adapter.adapt(h_try, e_est, x, x_high, M_filt)
                tol_vec = self.cfg.tol_abs + self.cfg.tol_rel * np.maximum(np.abs(x), np.abs(x_high))
                scale = np.linalg.norm(tol_vec, ord=2) + 1e-16
                e_norm = e_est / (scale + 1e-16)
                accept = (e_norm <= 1.0)
                if accept or retries >= self.cfg.max_retries:
                    t_next = t + h_try
                    # note: we do NOT commit x yet until interactive handling completes.
                    x_post = x_high.copy()
                    y_next = self.output(x_post, u)
                    poles_cont = self._continuous_poles()
                    poles_disc = self._discrete_poles_for_h(h_try)
                    # prepare serializations for possible logging
                    weights_json = json.dumps(w_norm)
                    poles_cont_json = json.dumps([[float(np.real(p)), float(np.imag(p))] for p in poles_cont.tolist()])
                    poles_disc_json = json.dumps([[float(np.real(p)), float(np.imag(p))] for p in poles_disc.tolist()])
                    x_json = json.dumps([float(xi) for xi in x_post.tolist()])
                    # interactive handling (if provided)
                    interactive_info = {'interactive': False, 'interactive_u': None, 'interactive_x_json': None, 'interactive_auto': False}
                    if interactive_cb is not None:
                        # call interactive callback with relevant info; this will block until GUI responds
                        try:
                            response = interactive_cb({
                                't_next': t_next,
                                'h_try': h_try,
                                'h_rk': h_rk,
                                'y_next': float(y_next),
                                'u_current': float(u),
                                'x_post': x_post.copy(),
                                'M_filt': float(M_filt),
                                'raw': raw,
                                'tilde': tilde,
                                'w_norm': w_norm,
                                'poles_cont': poles_cont.copy(),
                                'poles_disc': poles_disc.copy()
                            })
                        except Exception:
                            response = None
                        if isinstance(response, dict):
                            # response keys: 'set_u' (float or None), 'set_x' (list or None), 'auto' (bool)
                            interactive_info['interactive'] = bool(response.get('changed', False))
                            if response.get('set_u') is not None:
                                interactive_info['interactive_u'] = float(response.get('set_u'))
                            if response.get('set_x') is not None:
                                try:
                                    xi = [float(xx) for xx in response.get('set_x')]
                                    interactive_info['interactive_x_json'] = json.dumps(xi)
                                except Exception:
                                    interactive_info['interactive_x_json'] = None
                            interactive_info['interactive_auto'] = bool(response.get('auto', False))
                            # apply changes if present
                            if response.get('set_x') is not None:
                                # override full state vector
                                try:
                                    new_x = np.asarray(response.get('set_x'), dtype=float).reshape(-1)
                                    if new_x.size == self.n:
                                        x_post = new_x.copy()
                                    else:
                                        # if lengths mismatch, pad or trim
                                        if new_x.size < self.n:
                                            new_x_full = np.zeros(self.n)
                                            new_x_full[:new_x.size] = new_x
                                            x_post = new_x_full
                                        else:
                                            x_post = new_x[:self.n]
                                except Exception:
                                    pass
                            if response.get('set_u') is not None:
                                last_u = float(response.get('set_u'))
                            else:
                                last_u = u
                        else:
                            # no valid response
                            last_u = u
                    else:
                        last_u = u
                    # commit state and history
                    x = x_post.copy()
                    history['t'].append(t_next)
                    history['x'].append(x.copy())
                    history['y'].append(float(y_next))
                    history['u'].append(float(last_u))
                    history['h'].append(float(h_try))
                    history['M'].append(float(M_filt))
                    history['poles_cont'].append(poles_cont.copy())
                    history['poles_disc'].append(poles_disc.copy())
                    # build row and write to CSV (include interactive info)
                    row = [
                        round(t_next, 9), round(h_prev, 12), round(h_try, 12), float(h_rk), float(y_next), float(last_u), x_json, float(e_est), float(e_norm),
                        float(raw['delta']), float(raw['r']), float(raw['c']), float(raw['m']), float(raw['f']), float(raw['D']),
                        float(raw['u_rms']), float(raw['u_rate']), float(raw['v']), float(raw['a']),
                        float(tilde['delta']), float(tilde['r']), float(tilde['f']), float(tilde['c']), float(tilde['m']), float(tilde['D']), float(tilde['v']), float(tilde['a']), float(tilde['u_rms']), float(tilde['u_rate']), float(tilde['e']),
                        float(M_raw), float(M_filt), float(M_L), float(Gamma)
                    ]
                    for k in list(self.weights.keys()):
                        row.append(float(self.weights.get(k, 0.0)))
                    row += [weights_json, poles_cont_json, poles_disc_json,
                            int(interactive_info.get('interactive', False)),
                            ('' if interactive_info.get('interactive_u') is None else float(interactive_info.get('interactive_u'))),
                            (interactive_info.get('interactive_x_json') or ''),
                            int(interactive_info.get('interactive_auto', False))]
                    # thread-safe CSV write
                    try:
                        with self.csv_lock:
                            self.csv_writer.writerow(row)
                            self.csv_file.flush()
                    except Exception:
                        pass
                    # advance time and step-size
                    t = t_next
                    h = clamp(h_new, self.cfg.h_min, self.cfg.h_max)
                    last_u = last_u
                    break
                else:
                    retries += 1
                    h_try = max(self.cfg.h_min, 0.5 * h_try)
            iter_count += 1
            if progress_callback and (iter_count % 50 == 0):
                try:
                    progress_callback(t, Tfinal)
                except Exception:
                    pass
        return history

# ---------------------------
# GUI Application
# ---------------------------
class App:
    def __init__(self, root):
        self.root = root
        root.title("Adaptive RK Interactive Simulator")
        self.cfg = Config()
        self.weights = dict(self.cfg.default_weights)
        # left: controls
        left = ttk.Frame(root, padding=6)
        left.grid(row=0, column=0, sticky='nw')
        ttk.Label(left, text='Numerator coeffs (high->low), comma:').grid(row=0, column=0, sticky='w')
        self.num_entry = ttk.Entry(left, width=30); self.num_entry.insert(0, '1')
        self.num_entry.grid(row=1, column=0, pady=2)
        ttk.Label(left, text='Denominator coeffs (high->low), comma:').grid(row=2, column=0, sticky='w')
        self.den_entry = ttk.Entry(left, width=30); self.den_entry.insert(0, '1, 2, 1')
        self.den_entry.grid(row=3, column=0, pady=2)
        ttk.Label(left, text='PID Kp, Ki, Kd:').grid(row=4, column=0, sticky='w')
        self.pid_entry = ttk.Entry(left, width=30); self.pid_entry.insert(0, '1.0, 0.0, 0.0')
        self.pid_entry.grid(row=5, column=0, pady=2)
        ttk.Label(left, text='Initial state x0 (comma):').grid(row=6, column=0, sticky='w')
        self.x0_entry = ttk.Entry(left, width=30); self.x0_entry.insert(0, '0, 0')
        self.x0_entry.grid(row=7, column=0, pady=2)
        ttk.Label(left, text='Initial step h0:').grid(row=8, column=0, sticky='w')
        self.h0_entry = ttk.Entry(left, width=30); self.h0_entry.insert(0, '0.01')
        self.h0_entry.grid(row=9, column=0, pady=2)
        ttk.Label(left, text='Simulate until T (s):').grid(row=10, column=0, sticky='w')
        self.T_entry = ttk.Entry(left, width=30); self.T_entry.insert(0, '2.0')
        self.T_entry.grid(row=11, column=0, pady=2)
        ttk.Label(left, text='Reference(s) (constant or comma-list):').grid(row=12, column=0, sticky='w')
        self.ref_entry = ttk.Entry(left, width=30); self.ref_entry.insert(0, '1.0')
        self.ref_entry.grid(row=13, column=0, pady=2)
        ttk.Label(left, text='CSV file name:').grid(row=14, column=0, sticky='w')
        # keep same CSV default name as earlier versions
        self.csv_entry = ttk.Entry(left, width=30); self.csv_entry.insert(0, 'adaptive_parameters_log.csv')
        self.csv_entry.grid(row=15, column=0, pady=2)
        self.run_btn = ttk.Button(left, text='Run', command=self.run_sim); self.run_btn.grid(row=16, column=0, pady=6)
        self.stop_btn = ttk.Button(left, text='Stop', command=self._stop_requested); self.stop_btn.grid(row=17, column=0, pady=2)
        # weights editor
        ttk.Label(left, text='Weights (editable)').grid(row=18, column=0, sticky='w', pady=(8,0))
        self.weights_frame = ttk.Frame(left)
        self.weights_frame.grid(row=19, column=0, sticky='w')
        self.weight_vars: Dict[str, tk.StringVar] = {}
        r = 0
        for k, v in self.weights.items():
            ttk.Label(self.weights_frame, text=k).grid(row=r, column=0, sticky='w')
            var = tk.StringVar(value=str(v))
            self.weight_vars[k] = var
            e = ttk.Entry(self.weights_frame, width=8, textvariable=var)
            e.grid(row=r, column=1, padx=4)
            r += 1
        ttk.Button(left, text='Load weights JSON', command=self._load_weights).grid(row=19, column=0, sticky='e')
        ttk.Button(left, text='Save weights JSON', command=self._save_weights).grid(row=20, column=0, sticky='e')
        # right: plots (y, u, x, h, M, poles)
        right = ttk.Frame(root, padding=6)
        right.grid(row=0, column=1)
        self.fig, axes = plt.subplots(6, 1, figsize=(9, 12))
        (self.ax_y, self.ax_u, self.ax_x, self.ax_h, self.ax_M, self.ax_poles) = axes
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=1)
        # status
        self.status = tk.Text(root, height=10, width=140)
        self.status.grid(row=1, column=0, columnspan=2, padx=6, pady=6)
        self._stop_event = threading.Event()
        self.sim_thread = None
        # interactive synchronization
        self._interaction_data = None
        self._interaction_wait: Optional[threading.Event] = None
        self._interaction_response = None
        self._auto_run = False

    def _append_status(self, s: str):
        try:
            stamp = time.strftime('%H:%M:%S')
            self.status.insert(tk.END, f'[{stamp}] {s}')
            self.status.see(tk.END)
        except Exception:
            pass

    def _parse_list(self, s: str):
        try:
            return [float(p.strip()) for p in s.split(',') if p.strip() != '']
        except Exception as e:
            raise ValueError(f'Failed to parse list: {e}')

    def _read_weights_from_gui(self):
        w = {}
        for k, var in self.weight_vars.items():
            try:
                w[k] = float(var.get())
            except Exception:
                w[k] = float(self.weights.get(k, 0.0))
        self.weights = w
        return w

    def _load_weights(self):
        path = filedialog.askopenfilename(title='Load weights JSON', filetypes=[('JSON files', '*.json'), ('All', '*.*')])
        if not path:
            return
        try:
            with open(path) as f:
                data = json.load(f)
            for k in data:
                if k in self.weight_vars:
                    self.weight_vars[k].set(str(data[k]))
            self._append_status(f'Loaded weights from {path}\n')
        except Exception as e:
            messagebox.showerror('Error', str(e))

    def _save_weights(self):
        path = filedialog.asksaveasfilename(title='Save weights JSON', defaultextension='.json', filetypes=[('JSON', '*.json')])
        if not path:
            return
        w = self._read_weights_from_gui()
        try:
            with open(path, 'w') as f:
                json.dump(w, f, indent=2)
            self._append_status(f'Saved weights to {path}\n')
        except Exception as e:
            messagebox.showerror('Error', str(e))

    def _stop_requested(self):
        self._stop_event.set()
        self._append_status('Stop requested...\n')

    def run_sim(self):
        try:
            num = self._parse_list(self.num_entry.get())
            den = self._parse_list(self.den_entry.get())
            Kp, Ki, Kd = self._parse_list(self.pid_entry.get())
            x0 = self._parse_list(self.x0_entry.get())
            h0 = float(self.h0_entry.get())
            Tfinal = float(self.T_entry.get())
            # parse reference: either single float or comma-list of floats
            ref_text = self.ref_entry.get().strip()
            ref_list = None
            if ',' in ref_text:
                ref_list = self._parse_list(ref_text)
                ref_val = ref_list[0] if ref_list else 0.0
            else:
                ref_val = float(ref_text) if ref_text != '' else 0.0
            csv_name = self.csv_entry.get().strip() or 'adaptive_parameters_log.csv'
        except Exception as e:
            messagebox.showerror('Input error', str(e))
            return
        try:
            A, B, C, D = tf_to_ss(num, den)
        except Exception as e:
            messagebox.showerror('TF error', str(e))
            return
        n = A.shape[0]
        if len(x0) != n:
            if len(x0) == 1 and n > 1:
                x0 = [x0[0]] + [0.0] * (n - 1)
            else:
                messagebox.showerror('x0 error', f'x0 length {len(x0)} != system order {n}')
                return
        pid = PID(Kp, Ki, Kd)
        self._read_weights_from_gui()
        sim = AdaptiveRKSimulator(A, B, C, D, self.cfg, csv_path=csv_name, weights=self.weights)
        self._append_status('Starting simulation...\n')
        try:
            self.run_btn.config(state='disabled')
        except Exception:
            pass
        self._stop_event.clear()
        self._auto_run = False

        def progress_cb(t_cur, T_tot):
            self.root.after(0, lambda: self._append_status('sim time {:.4f}/{:.4f}\n'.format(t_cur, T_tot)))

        # interactive callback invoked from sim thread; must schedule modal in main thread and wait
        def interactive_cb(step_info):
            # step_info contains t_next, h_try, h_rk, y_next, u_current, x_post, M_filt, raw, tilde, w_norm, poles...
            # store data for dialog and wait for GUI input
            self._interaction_data = step_info
            self._interaction_wait = threading.Event()
            self._interaction_response = None
            # schedule dialog in GUI thread
            try:
                self.root.after(0, self._open_interactive_dialog)
            except Exception:
                pass
            # wait until GUI sets the response or stop requested
            while True:
                if self._interaction_wait.wait(timeout=0.1):
                    break
                if self._stop_event.is_set():
                    # if stop requested while waiting, return no-change
                    self._interaction_response = {'changed': False, 'set_u': None, 'set_x': None, 'auto': False}
                    break
            resp = self._interaction_response
            # ensure a valid dict
            if not isinstance(resp, dict):
                resp = {'changed': False, 'set_u': None, 'set_x': None, 'auto': False}
            # if auto-run checkbox was set, update flag to skip future dialogs
            if resp.get('auto', False):
                self._auto_run = True
            return resp

        def thread_fn():
            try:
                if ref_list is not None:
                    hist = sim.simulate(np.array(x0), pid, lambda tt: ref_val, h0, Tfinal,
                                        progress_callback=progress_cb, meta_save=True, ref_sequence=ref_list,
                                        interactive_cb=(interactive_cb if not self._auto_run else None),
                                        stop_event=self._stop_event)
                else:
                    hist = sim.simulate(np.array(x0), pid, lambda tt: ref_val, h0, Tfinal,
                                        progress_callback=progress_cb, meta_save=True, ref_sequence=None,
                                        interactive_cb=(interactive_cb if not self._auto_run else None),
                                        stop_event=self._stop_event)
                # done -> schedule plotting & status update on main thread
                self.root.after(0, lambda: self._on_sim_done(hist))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror('Simulation error', str(e)))
            finally:
                sim.close_csv()
                # re-enable run button safely
                def _enable_run():
                    try:
                        self.run_btn.config(state='normal')
                    except Exception:
                        pass
                try:
                    self.root.after(0, _enable_run)
                except Exception:
                    pass

        t = threading.Thread(target=thread_fn, daemon=True)
        t.start()
        self.sim_thread = t

    def _open_interactive_dialog(self):
        # Called in main thread. Create modal dialog showing self._interaction_data
        data = self._interaction_data or {}
        t_next = data.get('t_next', 0.0)
        h_try = data.get('h_try', 0.0)
        y_next = data.get('y_next', 0.0)
        u_current = data.get('u_current', 0.0)
        x_post = data.get('x_post', np.array([]))
        M_filt = data.get('M_filt', 0.0)
        raw = data.get('raw', {})
        tilde = data.get('tilde', {})
        w_norm = data.get('w_norm', {})

        dlg = tk.Toplevel(self.root)
        dlg.transient(self.root)
        dlg.grab_set()
        dlg.title(f'Interactive step @ t={t_next:.6f}s')
        # info labels
        info = ttk.Frame(dlg, padding=6)
        info.grid(row=0, column=0, sticky='nsew')
        ttk.Label(info, text=f't = {t_next:.6f} s    h = {h_try:.6g}    y = {y_next:.6g}    u = {u_current:.6g}    M_filtered = {M_filt:.6g}').grid(row=0, column=0, sticky='w', pady=(0,6))
        ttk.Label(info, text=f'x (state) = {np.array2string(np.asarray(x_post), precision=6, separator=", ")}').grid(row=1, column=0, sticky='w')
        ttk.Label(info, text=f'metrics (delta={raw.get("delta",0):.6g}, r={raw.get("r",0):.6g}, u_rms={raw.get("u_rms",0):.6g})').grid(row=2, column=0, sticky='w', pady=(0,6))
        # entries for new values
        fields = ttk.Frame(dlg, padding=6)
        fields.grid(row=1, column=0, sticky='nsew')
        ttk.Label(fields, text='New u (leave empty to keep):').grid(row=0, column=0, sticky='w')
        u_var = tk.StringVar(value='')
        u_entry = ttk.Entry(fields, width=30, textvariable=u_var)
        u_entry.grid(row=0, column=1, sticky='w', padx=4, pady=2)
        ttk.Label(fields, text='Override x (comma list) (leave empty to keep):').grid(row=1, column=0, sticky='w')
        x_var = tk.StringVar(value='')
        x_entry = ttk.Entry(fields, width=40, textvariable=x_var)
        x_entry.grid(row=1, column=1, sticky='w', padx=4, pady=2)
        auto_var = tk.BooleanVar(value=self._auto_run)
        ttk.Checkbutton(fields, text='Auto-run from now on (no more dialogs)', variable=auto_var).grid(row=2, column=0, columnspan=2, sticky='w', pady=(4,0))
        # buttons
        btns = ttk.Frame(dlg, padding=6)
        btns.grid(row=2, column=0, sticky='e')
        def on_apply_and_continue():
            # collect inputs and set response
            resp = {'changed': False, 'set_u': None, 'set_x': None, 'auto': bool(auto_var.get())}
            u_text = u_var.get().strip()
            x_text = x_var.get().strip()
            if u_text != '':
                try:
                    resp['set_u'] = float(u_text)
                    resp['changed'] = True
                except Exception:
                    messagebox.showerror('Input error', 'Failed to parse new u as float.')
                    return
            if x_text != '':
                try:
                    parts = [float(p.strip()) for p in x_text.split(',') if p.strip() != '']
                    resp['set_x'] = parts
                    resp['changed'] = True
                except Exception:
                    messagebox.showerror('Input error', 'Failed to parse x override (comma-separated floats).')
                    return
            # store the response and notify waiting thread
            self._interaction_response = resp
            try:
                if self._interaction_wait is not None:
                    self._interaction_wait.set()
            except Exception:
                pass
            try:
                dlg.grab_release()
                dlg.destroy()
            except Exception:
                pass

        def on_continue_only():
            resp = {'changed': False, 'set_u': None, 'set_x': None, 'auto': bool(auto_var.get())}
            self._interaction_response = resp
            try:
                if self._interaction_wait is not None:
                    self._interaction_wait.set()
            except Exception:
                pass
            try:
                dlg.grab_release()
                dlg.destroy()
            except Exception:
                pass

        def on_cancel_stop():
            # treat as continue but also request stop
            resp = {'changed': False, 'set_u': None, 'set_x': None, 'auto': bool(auto_var.get())}
            self._interaction_response = resp
            try:
                if self._interaction_wait is not None:
                    self._interaction_wait.set()
            except Exception:
                pass
            self._stop_event.set()
            try:
                dlg.grab_release()
                dlg.destroy()
            except Exception:
                pass

        ttk.Button(btns, text='Apply & Continue', command=on_apply_and_continue).grid(row=0, column=0, padx=4)
        ttk.Button(btns, text='Continue (no change)', command=on_continue_only).grid(row=0, column=1, padx=4)
        ttk.Button(btns, text='Continue & Stop simulation', command=on_cancel_stop).grid(row=0, column=2, padx=4)
        # focus
        u_entry.focus_set()

    def _on_sim_done(self, hist):
        ts = np.array(hist['t']) if hist['t'] else np.array([])
        ys = np.array(hist['y']) if hist['y'] else np.array([])
        xs = np.array(hist['x']) if hist['x'] else np.array([])
        hs = np.array(hist['h']) if hist['h'] else np.array([])
        Ms = np.array(hist['M']) if hist['M'] else np.array([])
        us = np.array(hist['u']) if hist['u'] else np.array([])

        # clear axes
        self.ax_y.cla(); self.ax_u.cla(); self.ax_x.cla(); self.ax_h.cla(); self.ax_M.cla(); self.ax_poles.cla()

        # y(t)
        if ys.size > 0:
            self.ax_y.plot(ts, ys, marker='o', label='y(t)')
            self.ax_y.set_title('Output response y(t)')
            self.ax_y.set_xlabel('Time [s]'); self.ax_y.set_ylabel('Output y'); self.ax_y.grid(True); self.ax_y.legend()
        else:
            self.ax_y.set_title('Output y(t) - empty')

        # u(t)
        if us.size > 0:
            self.ax_u.plot(ts, us, marker='.', label='u(t) [control]')
            self.ax_u.set_title('Control signal u(t)')
            self.ax_u.set_xlabel('Time [s]'); self.ax_u.set_ylabel('Control u'); self.ax_u.grid(True); self.ax_u.legend()
        else:
            self.ax_u.set_title('Control signal u(t) - empty')

        # states x(t)
        if xs.size > 0:
            if xs.ndim == 2:
                for i in range(xs.shape[1]):
                    self.ax_x.plot(ts, xs[:, i], label=f'x[{i}]')
            else:
                # 1D sequence of scalars
                self.ax_x.plot(ts, xs, label='x[0]')
            self.ax_x.set_title('State variables x(t)')
            self.ax_x.set_xlabel('Time [s]'); self.ax_x.set_ylabel('States x'); self.ax_x.grid(True); self.ax_x.legend()
        else:
            self.ax_x.set_title('States x(t) - empty')

        # h(t)
        if hs.size > 0:
            self.ax_h.plot(ts, hs, marker='x', label='h(t)')
            self.ax_h.set_title('Adaptive step size h(t)')
            self.ax_h.set_xlabel('Time [s]'); self.ax_h.set_ylabel('Step size h'); self.ax_h.grid(True); self.ax_h.legend()
        else:
            self.ax_h.set_title('Step size h(t) - empty')

        # M(t)
        if Ms.size > 0:
            self.ax_M.plot(ts, Ms, marker='.', label='M_filtered(t)')
            self.ax_M.set_title('Adaptive metric M_filtered(t)')
            self.ax_M.set_xlabel('Time [s]'); self.ax_M.set_ylabel('M value'); self.ax_M.grid(True); self.ax_M.legend()
        else:
            self.ax_M.set_title('M_filtered(t) - empty')

        # Pole plot (last)
        try:
            if hist['poles_cont']:
                last_cont = np.asarray(hist['poles_cont'][-1])
                last_disc = np.asarray(hist['poles_disc'][-1])
                if last_cont.size > 0:
                    self.ax_poles.scatter(np.real(last_cont), np.imag(last_cont), marker='x', label='continuous poles')
                if last_disc.size > 0:
                    self.ax_poles.scatter(np.real(last_disc), np.imag(last_disc), marker='o', label='predicted discrete poles')
                self.ax_poles.axvline(0, color='gray', linewidth=0.5); self.ax_poles.axhline(0, color='gray', linewidth=0.5)
                self.ax_poles.set_title('Pole locations: continuous (x) vs predicted discrete (o)')
                self.ax_poles.set_xlabel('Real axis'); self.ax_poles.set_ylabel('Imag axis'); self.ax_poles.grid(True); self.ax_poles.legend()
            else:
                self.ax_poles.set_title('Pole plot - empty')
        except Exception:
            self.ax_poles.set_title('Pole plot error')

        self.fig.tight_layout()
        try:
            self.canvas.draw()
        except Exception:
            pass
        csv_path = os.path.abspath(self.csv_entry.get().strip() or 'adaptive_parameters_log.csv')
        self._append_status(f'Simulation finished. Steps: {len(ts)}. CSV written to {csv_path}\n')

# ---------------------------
# Run
# ---------------------------
if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()
