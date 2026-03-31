import os
import pickle
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Patch
from scipy.signal import butter, filtfilt
from scipy.stats import skew, kurtosis


WINDOW_SECONDS = 5
SAMPLE_RATE = 100
SAMPLES_PER_WIN = WINDOW_SECONDS * SAMPLE_RATE
SMA_WINDOW = 5
HP_CUTOFF = 0.25
HP_ORDER = 4
MODEL_PATH = "step6_model.pkl"

LABEL_MAP = {0: "walking", 1: "jumping"}
COLOUR_MAP = {"walking": "#2196F3", "jumping": "#FF5722"}


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    return df.ffill().bfill()


def apply_sma(df: pd.DataFrame, window: int = SMA_WINDOW) -> pd.DataFrame:
    out = df.copy()
    for col in ("Ax", "Ay", "Az"):
        out[col] = df[col].rolling(window, center=True).mean()
    return out.ffill().bfill()


def apply_highpass(df: pd.DataFrame,
                   cutoff: float = HP_CUTOFF,
                   fs: int = SAMPLE_RATE,
                   order: int = HP_ORDER) -> pd.DataFrame:
    out = df.copy()
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="high", analog=False)
    for col in ("Ax", "Ay", "Az"):
        mean_val = df[col].mean()
        out[col] = filtfilt(b, a, df[col].values) + mean_val
    return out


def preprocess(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = fill_missing(df_raw)
    df = apply_sma(df)
    df = apply_highpass(df)
    return df


def extract_features(df: pd.DataFrame) -> dict:
    feats = {}
    df = df.copy()
    df["Magnitude"] = np.sqrt(df["Ax"]**2 + df["Ay"]**2 + df["Az"]**2)

    for axis in ("Ax", "Ay", "Az", "Magnitude"):
        col = df[axis]
        feats[f"{axis}_mean"] = col.mean()
        feats[f"{axis}_median"] = col.median()
        feats[f"{axis}_std"] = col.std()
        feats[f"{axis}_var"] = col.var()
        feats[f"{axis}_max"] = col.max()
        feats[f"{axis}_min"] = col.min()
        feats[f"{axis}_range"] = col.max() - col.min()
        feats[f"{axis}_rms"] = np.sqrt(np.mean(col**2))
        feats[f"{axis}_skew"] = skew(col)
        feats[f"{axis}_kurtosis"] = kurtosis(col)

    return feats


def load_model(path: str):
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["scaler"]


def prepare_input_csv(df_raw: pd.DataFrame) -> pd.DataFrame:
    rename = {}

    for c in df_raw.columns:
        cl = c.lower().strip()
        if "time" in cl:
            rename[c] = "Time"
        elif "acceleration x" in cl or "accel x" in cl or cl == "ax":
            rename[c] = "Ax"
        elif "acceleration y" in cl or "accel y" in cl or cl == "ay":
            rename[c] = "Ay"
        elif "acceleration z" in cl or "accel z" in cl or cl == "az":
            rename[c] = "Az"

    df_raw = df_raw.rename(columns=rename)

    required = ["Time", "Ax", "Ay", "Az"]
    missing = [c for c in required if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df_raw[required].copy()


def segment_and_classify(df_proc: pd.DataFrame, model, scaler) -> pd.DataFrame:
    num_windows = len(df_proc) // SAMPLES_PER_WIN
    if num_windows == 0:
        raise ValueError("File is too short. Need at least 5 seconds of data (500 samples).")

    records = []

    for i in range(num_windows):
        s = i * SAMPLES_PER_WIN
        e = s + SAMPLES_PER_WIN
        seg = df_proc.iloc[s:e].copy()

        feature_dict = extract_features(seg)
        feature_vec = np.array(list(feature_dict.values())).reshape(1, -1)
        feature_scaled = scaler.transform(feature_vec)
        pred_num = int(model.predict(feature_scaled)[0])

        records.append({
            "window_index": i,
            "start_time_s": round(float(seg["Time"].iloc[0]), 4),
            "end_time_s": round(float(seg["Time"].iloc[-1]), 4),
            "label": LABEL_MAP[pred_num],
            "label_numeric": pred_num
        })

    return pd.DataFrame(records)


def build_figure(df_proc: pd.DataFrame, results_df: pd.DataFrame) -> Figure:
    fig = Figure(figsize=(11, 7), dpi=100)
    axes = fig.subplots(3, 1, sharex=True)
    fig.patch.set_facecolor("#0F1117")

    for ax, col, ylabel in zip(
        axes,
        ("Ax", "Ay", "Az"),
        ("Ax (m/s²)", "Ay (m/s²)", "Az (m/s²)")
    ):
        ax.set_facecolor("#161B22")
        ax.tick_params(colors="#AAAAAA")
        ax.yaxis.label.set_color("#CCCCCC")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2A2F3A")
        ax.set_ylabel(ylabel, fontsize=9, color="#CCCCCC")
        ax.grid(True, color="#2A2F3A", linewidth=0.5)

        for _, row in results_df.iterrows():
            mask = (df_proc["Time"] >= row["start_time_s"]) & (df_proc["Time"] <= row["end_time_s"])
            seg = df_proc[mask]
            ax.plot(seg["Time"], seg[col], color=COLOUR_MAP[row["label"]], linewidth=0.8)

    axes[-1].set_xlabel("Time (s)", fontsize=9, color="#CCCCCC")
    axes[-1].tick_params(axis="x", colors="#AAAAAA")

    legend_elems = [
        Patch(facecolor=COLOUR_MAP["walking"], label="Walking"),
        Patch(facecolor=COLOUR_MAP["jumping"], label="Jumping")
    ]
    axes[0].legend(handles=legend_elems, loc="upper right",
                   facecolor="#1E2230", labelcolor="white",
                   framealpha=0.8, fontsize=9)

    fig.suptitle("Activity Classification – Accelerometer Signal",
                 fontsize=13, color="white", fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


class ActivityClassifierApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Activity Classifier – Step 7")
        self.geometry("1000x720")
        self.configure(bg="#0F1117")

        self.model = None
        self.scaler = None
        self.results_df = None
        self.df_proc = None
        self.canvas_widget = None
        self.toolbar = None

        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar(value="labelled_results.csv")
        self.status_var = tk.StringVar(value="Idle")

        self.build_ui()
        self.try_auto_load_model()

    def build_ui(self):
        top = tk.Frame(self, bg="#0F1117")
        top.pack(fill="x", padx=20, pady=12)

        tk.Label(top, text="Activity Classifier", font=("Arial", 18, "bold"),
                 bg="#0F1117", fg="white").pack(anchor="w")

        controls = tk.Frame(self, bg="#0F1117")
        controls.pack(fill="x", padx=20, pady=8)

        tk.Button(controls, text="Load CSV", command=self.browse_input).pack(side="left", padx=5)
        tk.Button(controls, text="Run Classification", command=self.run_clicked).pack(side="left", padx=5)
        tk.Button(controls, text="Export Results CSV", command=self.export_clicked).pack(side="left", padx=5)

        tk.Label(self, textvariable=self.input_path, bg="#0F1117", fg="#AAAAAA").pack(anchor="w", padx=25)
        tk.Label(self, textvariable=self.status_var, bg="#0F1117", fg="#AAAAAA").pack(anchor="w", padx=25, pady=(0, 8))

        self.summary_lbl = tk.Label(self, text="No results yet.", justify="left",
                                    bg="#0F1117", fg="white")
        self.summary_lbl.pack(anchor="w", padx=25, pady=(0, 8))

        self.plot_frame = tk.Frame(self, bg="#161B22")
        self.plot_frame.pack(fill="both", expand=True, padx=20, pady=10)

    def try_auto_load_model(self):
        if os.path.exists(MODEL_PATH):
            self.model, self.scaler = load_model(MODEL_PATH)
            self.status_var.set("Model loaded successfully.")

    def browse_input(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if path:
            self.input_path.set(path)
            self.output_path.set(os.path.splitext(path)[0] + "_labelled.csv")

    def run_clicked(self):
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded.")
            return
        if not self.input_path.get():
            messagebox.showerror("Error", "Please choose a CSV file first.")
            return

        threading.Thread(target=self.pipeline, daemon=True).start()

    def pipeline(self):
        try:
            self.status_var.set("Reading CSV...")
            df_raw = pd.read_csv(self.input_path.get())
            df_raw = prepare_input_csv(df_raw)

            self.status_var.set("Preprocessing...")
            df_proc = preprocess(df_raw)

            self.status_var.set("Classifying...")
            results_df = segment_and_classify(df_proc, self.model, self.scaler)

            self.df_proc = df_proc
            self.results_df = results_df

            n_walk = (results_df["label"] == "walking").sum()
            n_jump = (results_df["label"] == "jumping").sum()
            total = len(results_df)

            summary = (
                f"Total windows: {total}\n"
                f"Walking: {n_walk}\n"
                f"Jumping: {n_jump}\n"
                f"Window length: {WINDOW_SECONDS}s"
            )
            self.summary_lbl.config(text=summary)

            fig = build_figure(df_proc, results_df)
            self.after(0, lambda: self.show_figure(fig))
            self.status_var.set("Done.")
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.status_var.set(f"Error: {e}")

    def show_figure(self, fig: Figure):
        if self.canvas_widget is not None:
            self.canvas_widget.get_tk_widget().destroy()
        if self.toolbar is not None:
            self.toolbar.destroy()

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()

        self.toolbar = NavigationToolbar2Tk(canvas, self.plot_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side="bottom", fill="x")

        canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas_widget = canvas

    def export_clicked(self):
        if self.results_df is None:
            messagebox.showwarning("Warning", "No results to export yet.")
            return

        try:
            self.results_df.to_csv(self.output_path.get(), index=False)
            messagebox.showinfo("Saved", f"Saved to:\n{self.output_path.get()}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))


if __name__ == "__main__":
    app = ActivityClassifierApp()
    app.mainloop()
