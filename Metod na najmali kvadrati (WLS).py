import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import time
import csv

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except Exception:
    PANDAS_AVAILABLE = False

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from collections import deque

# =========================
# Класи за лизгачки прозорец, Хубер Лос и методот на најмали квадрати (WLS)
# =========================
class SlidingWindow:
    """Efficient buffer for real-time data. Gi cuva poslednite N merenja."""
    def __init__(self, window_size):
        self.window_size = window_size
        self.data = deque(maxlen=window_size)

    def add(self, input_vector, output_value):
        self.data.append((input_vector, output_value))

    def get_data(self):
        return list(self.data)

    def clear(self):
        self.data.clear()

    def __len__(self):
        return len(self.data)


class HuberLoss:
    """Robust loss function used in regression. Се намалува влијанието на outliers."""
    def __init__(self, delta):
        self.delta = delta

    def compute(self, residuals):
        losses = []
        for a in residuals:
            abs_a = abs(a)
            if abs_a <= self.delta:
                losses.append(0.5 * a * a)
            else:
                losses.append(self.delta * (abs_a - 0.5 * self.delta))
        return losses

    def derivative(self, residuals):
        derivatives = []
        for a in residuals:
            abs_a = abs(a)
            if abs_a <= self.delta:
                derivatives.append(a)
            else:
                derivatives.append(self.delta * (1 if a > 0 else -1))
        return derivatives

    def weights(self, residuals, epsilon=1e-8):
        weights = []
        for a in residuals:
            abs_a = abs(a)
            if abs_a <= self.delta:
                weights.append(1.0)
            else:
                weights.append(self.delta / (abs_a + epsilon))
        return weights


class LeastSquaresIdentifier:
    """Метод на најмали квадрати + Лизгачки прозорец + Хубер Лос"""
    def __init__(self, sliding_window, huber_loss, n_params):
        self.window = sliding_window
        self.huber = huber_loss
        self.n_params = n_params
        self.parameters = np.zeros(n_params)

    def model_predict(self, input_vector, parameters=None):
        if parameters is None:
            parameters = self.parameters
        return np.dot(parameters, input_vector)

    def calc_residuals(self, parameters=None):
        if parameters is None:
            parameters = self.parameters
        residuals = []
        for input_vector, y_measured in self.window.get_data():
            y_pred = self.model_predict(input_vector, parameters)
            residuals.append(y_measured - y_pred)
        return np.array(residuals)

    def fit(self):
        if len(self.window) < self.n_params:
            return self.parameters

        X = np.array([x for x, y in self.window.get_data()])
        y = np.array([y for x, y in self.window.get_data()])

        residuals = self.calc_residuals(self.parameters)
        weights = self.huber.weights(residuals)
        W = np.diag(weights)

        try:
            XtWX = X.T @ W @ X
            XtWy = X.T @ W @ y
            new_params = np.linalg.solve(XtWX, XtWy)
        except np.linalg.LinAlgError:
            new_params = np.linalg.lstsq(X, y, rcond=None)[0]

        self.parameters = new_params
        return self.parameters

    def update(self, input_vector, output_value):
        self.window.add(input_vector, output_value)
        if len(self.window) >= self.n_params:
            self.fit()
        return self.parameters

    def get_parameters(self):
        return self.parameters


# =========================
# GUI
# =========================
class IdentGUI:
    def __init__(self, master):
        self.master = master
        master.title("Identification GUI — SlidingWindow + Huber + LSI")

        # Default config
        self.default_window = 20
        self.default_nparams = 3
        self.default_delta = 1.0

        # Frame: Configuration
        cfg_frame = ttk.LabelFrame(master, text="Configuration", padding=6)
        cfg_frame.grid(row=0, column=0, sticky="nw", padx=6, pady=6)

        ttk.Label(cfg_frame, text="Window size:").grid(row=0, column=0, sticky="w")
        self.win_entry = ttk.Entry(cfg_frame, width=8); self.win_entry.insert(0, str(self.default_window))
        self.win_entry.grid(row=0, column=1, sticky="w", padx=4)

        ttk.Label(cfg_frame, text="n_params:").grid(row=0, column=2, sticky="w")
        self.np_entry = ttk.Entry(cfg_frame, width=8); self.np_entry.insert(0, str(self.default_nparams))
        self.np_entry.grid(row=0, column=3, sticky="w", padx=4)

        ttk.Label(cfg_frame, text="Huber delta:").grid(row=0, column=4, sticky="w")
        self.delta_entry = ttk.Entry(cfg_frame, width=8); self.delta_entry.insert(0, str(self.default_delta))
        self.delta_entry.grid(row=0, column=5, sticky="w", padx=4)

        self.create_btn = ttk.Button(cfg_frame, text="Create / Reset Identifier", command=self.create_identifier)
        self.create_btn.grid(row=1, column=0, columnspan=2, pady=4, sticky="w")

        self.clear_btn = ttk.Button(cfg_frame, text="Clear Window", command=self.clear_window)
        self.clear_btn.grid(row=1, column=2, columnspan=2, pady=4, sticky="w")

        # Frame: Add sample
        sample_frame = ttk.LabelFrame(master, text="Add sample (input_vector, output_value)", padding=6)
        sample_frame.grid(row=1, column=0, sticky="nw", padx=6, pady=6)

        ttk.Label(sample_frame, text="Input vector (comma):").grid(row=0, column=0, sticky="w")
        self.input_entry = ttk.Entry(sample_frame, width=40)
        self.input_entry.grid(row=0, column=1, sticky="w", padx=4)
        self.input_entry.insert(0, "1.0,0.0,0.0")

        ttk.Label(sample_frame, text="Output y:").grid(row=1, column=0, sticky="w")
        self.y_entry = ttk.Entry(sample_frame, width=20)
        self.y_entry.grid(row=1, column=1, sticky="w", padx=4)
        self.y_entry.insert(0, "0.0")

        self.add_btn = ttk.Button(sample_frame, text="Add sample (+fit)", command=self.add_sample_and_fit)
        self.add_btn.grid(row=2, column=0, pady=6, sticky="w")
        self.add_nofit_btn = ttk.Button(sample_frame, text="Add sample (no fit)", command=self.add_sample_no_fit)
        self.add_nofit_btn.grid(row=2, column=1, pady=6, sticky="w")
        self.fit_now_btn = ttk.Button(sample_frame, text="Fit now", command=self.fit_now)
        self.fit_now_btn.grid(row=2, column=2, pady=6, sticky="w")

        opt_frame = ttk.Frame(sample_frame)
        opt_frame.grid(row=3, column=0, columnspan=3, sticky="w")
        self.autofit_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt_frame, text="Auto-fit on add", variable=self.autofit_var).grid(row=0, column=0, sticky="w", padx=2)

        # Frame: Plots
        plot_frame = ttk.LabelFrame(master, text="Plots", padding=4)
        plot_frame.grid(row=0, column=1, rowspan=3, sticky="nsew", padx=6, pady=6)
        master.grid_columnconfigure(1, weight=1)
        master.grid_rowconfigure(2, weight=1)

        self.fig, axes = plt.subplots(4, 1, figsize=(6, 10))
        self.ax_meas, self.ax_res, self.ax_params, self.ax_weights = axes
        plt.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Frame: Bottom
        bottom_frame = ttk.Frame(master, padding=6)
        bottom_frame.grid(row=3, column=0, columnspan=2, sticky="ew")
        ttk.Label(bottom_frame, text="Current parameters:").grid(row=0, column=0, sticky="w")
        self.param_text = tk.Text(bottom_frame, height=2, width=80)
        self.param_text.grid(row=1, column=0, columnspan=3, sticky="w")

        self.export_btn = ttk.Button(bottom_frame, text="Export window -> CSV", command=self.export_csv)
        self.export_btn.grid(row=0, column=2, sticky="e", padx=6)
        self.load_btn = ttk.Button(bottom_frame, text="Load CSV", command=self.load_csv)
        self.load_btn.grid(row=1, column=2, sticky="e", padx=6)
        self.status_label = ttk.Label(bottom_frame, text="Ready")
        self.status_label.grid(row=0, column=1, sticky="w", padx=6)

        # Initialize
        self.identifier = None
        self.param_history = []
        self.pred_history = []
        self.y_history = []
        self.time_history = []
        self.all_data = []  # <<< Сите внесени податоци
        self.create_identifier()

    # =========================
    # GUI methods
    # =========================
    def create_identifier(self):
        try:
            window_size = int(self.win_entry.get())
            n_params = int(self.np_entry.get())
            delta = float(self.delta_entry.get())
        except Exception as e:
            messagebox.showerror("Input error", f"Invalid configuration: {e}")
            return
        self.window = SlidingWindow(window_size)
        self.huber = HuberLoss(delta)
        self.identifier = LeastSquaresIdentifier(self.window, self.huber, n_params)
        self.param_history.clear()
        self.pred_history.clear()
        self.y_history.clear()
        self.time_history.clear()
        self.all_data.clear()
        self.update_status(f"Created identifier: window={window_size}, n_params={n_params}, delta={delta}")
        self.redraw_plots()

    def clear_window(self):
        if self.identifier is None:
            return
        self.identifier.window.clear()
        self.param_history.clear()
        self.pred_history.clear()
        self.y_history.clear()
        self.time_history.clear()
        self.all_data.clear()
        self.update_status("Window cleared.")
        self.redraw_plots()
        self.update_parameters_text()

    def parse_input_vector(self, s):
        return np.array([float(p.strip()) for p in s.split(",") if p.strip() != ""])

    def add_sample_and_fit(self):
        self._add_sample(do_fit=True)

    def add_sample_no_fit(self):
        self._add_sample(do_fit=False)

    def _add_sample(self, do_fit=True):
        if self.identifier is None:
            messagebox.showerror("Error", "Identifier not created.")
            return
        try:
            x = self.parse_input_vector(self.input_entry.get())
            y = float(self.y_entry.get())
        except Exception as e:
            messagebox.showerror("Input parse error", str(e))
            return
        # Pad or trim
        if x.size < self.identifier.n_params:
            x = np.concatenate([x, np.zeros(self.identifier.n_params - x.size)])
        elif x.size > self.identifier.n_params:
            x = x[:self.identifier.n_params]

        self.all_data.append((x, y))  # <<< Сочувај сите податоци

        params = self.identifier.update(x, y) if do_fit else self.identifier.get_parameters()
        self.param_history.append(params.copy())
        self.pred_history.append(np.array([self.identifier.model_predict(d[0]) for d in self.window.get_data()]))
        self.y_history.append(np.array([d[1] for d in self.window.get_data()]))
        self.time_history.append(time.time())
        self.update_status(f"Added sample x={x.tolist()}, y={y:.6g}  | window len={len(self.identifier.window)}")
        self.update_parameters_text()
        self.redraw_plots()

    def fit_now(self):
        if self.identifier is None:
            messagebox.showerror("Error", "Identifier not created.")
            return
        if len(self.identifier.window) < self.identifier.n_params:
            messagebox.showinfo("Not enough data", f"Need at least {self.identifier.n_params} samples.")
            return
        self.identifier.fit()
        self.param_history.append(self.identifier.get_parameters().copy())
        self.pred_history.append(np.array([self.identifier.model_predict(d[0]) for d in self.window.get_data()]))
        self.y_history.append(np.array([d[1] for d in self.window.get_data()]))
        self.time_history.append(time.time())
        self.update_parameters_text()
        self.redraw_plots()
        self.update_status("Fitted parameters.")

    def update_parameters_text(self):
        if self.identifier is None:
            return
        params = self.identifier.get_parameters()
        self.param_text.delete("1.0", tk.END)
        self.param_text.insert(tk.END, np.array2string(params, precision=6, separator=", "))

    def redraw_plots(self):
        if self.identifier is None:
            return
        self.ax_meas.cla()
        self.ax_res.cla()
        self.ax_params.cla()
        self.ax_weights.cla()

        data = self.identifier.window.get_data()
        if len(data) > 0:
            X = np.vstack([d[0] for d in data])
            y = np.array([d[1] for d in data])
            y_pred = np.array([self.identifier.model_predict(d[0]) for d in data])
            residuals = y - y_pred
            weights = self.huber.weights(residuals)

            # Measured vs Predicted
            self.ax_meas.plot(range(len(y)), y, 'o-', label='measured')
            self.ax_meas.plot(range(len(y_pred)), y_pred, 'x--', label='predicted')
            self.ax_meas.set_title("Measured vs Predicted")
            self.ax_meas.legend()
            self.ax_meas.grid(True)

            # Residuals
            self.ax_res.plot(range(len(residuals)), residuals, 'o-', label='residual')
            self.ax_res.axhline(0, color='k', linewidth=0.5)
            self.ax_res.set_title("Residuals")
            self.ax_res.grid(True)

            # Parameter evolution
            if self.param_history:
                P = np.vstack(self.param_history)
                for i in range(P.shape[1]):
                    self.ax_params.plot(range(P.shape[0]), P[:, i], label=f"theta[{i}]")
                self.ax_params.set_title("Parameter evolution")
                self.ax_params.legend()
                self.ax_params.grid(True)

            # Weights
            self.ax_weights.plot(range(len(weights)), weights, 'o-', label='Huber weight')
            self.ax_weights.set_title("Huber weights")
            self.ax_weights.grid(True)

        self.canvas.draw()

    def export_csv(self):
        if self.identifier is None or len(self.all_data) == 0:
            messagebox.showinfo("Empty", "No data to export.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv")
        if not path:
            return

        data = list(self.all_data[-self.window.window_size:])
        start_idx = len(self.all_data) - len(data)

        rows = []
        for i, (x, y) in enumerate(data):
            param_idx = start_idx + i
            if param_idx < len(self.param_history):
                params = self.param_history[param_idx]
            else:
                params = self.identifier.get_parameters()
            y_pred = float(np.dot(params, x))
            residual = float(y - y_pred)
            row = {f"x{j}": float(val) for j, val in enumerate(x)}
            row["y"] = float(y)
            row["y_pred"] = y_pred
            row["residual"] = residual
            weights = self.huber.weights([residual])
            row["weight"] = float(weights[0])
            for k, theta in enumerate(params):
                row[f"theta{k}"] = float(theta)
            rows.append(row)

        try:
            if PANDAS_AVAILABLE:
                pd.DataFrame(rows).to_csv(path, index=False)
            else:
                keys = rows[0].keys()
                with open(path, "w", newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    for r in rows:
                        writer.writerow(r)
            self.update_status(f"Exported {len(rows)} rows to {path}")
            messagebox.showinfo("Exported", f"Exported {len(rows)} rows to\n{path}")
        except Exception as e:
            messagebox.showerror("Export error", str(e))

    def load_csv(self):
        if self.identifier is None:
            return
        path = filedialog.askopenfilename()
        if not path:
            return
        try:
            with open(path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    x = np.array([float(row[f"x{i}"]) for i in range(self.identifier.n_params)])
                    y = float(row["y"])
                    self.all_data.append((x, y))
                    self.identifier.update(x, y)
                    self.param_history.append(self.identifier.get_parameters().copy())
            self.redraw_plots()
            self.update_parameters_text()
            self.update_status(f"Loaded data from {path}")
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    def update_status(self, s):
        self.status_label.config(text=s)
        self.master.update_idletasks()


if __name__ == "__main__":
    root = tk.Tk()
    app = IdentGUI(root)
    root.mainloop()
