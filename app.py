import io
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import streamlit as st


# ============================
# Robust I/O: CSV + Excel with encoding/delimiter fallback
# ============================
@st.cache_data(show_spinner=False)
def load_table(uploaded_file) -> pd.DataFrame:
    """
    Read CSV or Excel with robust fallbacks:
    - .xlsx with openpyxl
    - .xls with xlrd
    - CSV with auto delimiter and encoding retries (utf-8-sig, cp1252, latin-1)
    """
    name = uploaded_file.name.lower()

    # Excel directly
    if name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file, engine="openpyxl")
    if name.endswith(".xls"):
        return pd.read_excel(uploaded_file, engine="xlrd")

    # CSV or unknown extension
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, sep=None, engine="python", encoding=enc)
        except Exception:
            pass

    # Fallback: some files may actually be Excel with a CSV extension
    uploaded_file.seek(0)
    try:
        return pd.read_excel(uploaded_file, engine="openpyxl")
    except Exception:
        uploaded_file.seek(0)
        return pd.read_excel(uploaded_file, engine="xlrd")


@st.cache_data(show_spinner=False)
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# ============================
# Flitt & Schweinsberg model
# ============================
@dataclass
class FlittParams:
    # log10 of exchange currents (A)
    log_i0_a: float = -6.0     # anodic Fe dissolution
    log_i0_H: float = -6.5     # hydrogen evolution
    log_i0_O2: float = -7.0    # oxygen reduction

    # Tafel slopes (V/dec)
    b_a: float = 0.10
    b_H: float = 0.12
    b_O2: float = 0.12

    # Equilibrium potentials (V vs the reference used in the file)
    Eeq_a: float = -0.45
    Eeq_H: float = 0.00
    Eeq_O2: float = 1.00

    # Diffusion limit magnitude for ORR (A, positive)
    i_L: float = 1e-3

    # Series resistance and current offset
    R_s: float = 0.0
    i_offset: float = 0.0


# Prevent overflow: 10**x where x is clipped to [-300, 300] (double precision safe)
def pow10_clip(x):
    return np.power(10.0, np.clip(x, -300.0, 300.0))


def tafels_anodic(E, log_i0, b, Eeq):
    # i_a > 0
    return pow10_clip(log_i0 + (E - Eeq) / b)


def tafels_cathodic_abs(E, log_i0, b, Eeq):
    # returns positive magnitude of cathodic kinetic current
    return pow10_clip(log_i0 - (E - Eeq) / b)


def orr_with_diffusion(E, log_i0, b, Eeq, i_L):
    # Koutecký–Levich mix of kinetic and diffusion limit for ORR
    ik_abs = tafels_cathodic_abs(E, log_i0, b, Eeq)
    denom = (1.0 / (ik_abs + 1e-300)) + (1.0 / (abs(i_L) + 1e-300))
    i_abs = 1.0 / denom
    return -i_abs  # cathodic sign


def partial_currents_at_internal_potential(Eint: np.ndarray, p: FlittParams,
                                           include_HER: bool = True,
                                           include_ORR: bool = True):
    ia = tafels_anodic(Eint, p.log_i0_a, p.b_a, p.Eeq_a)
    iH = -tafels_cathodic_abs(Eint, p.log_i0_H, p.b_H, p.Eeq_H) if include_HER else 0.0
    iO2 = orr_with_diffusion(Eint, p.log_i0_O2, p.b_O2, p.Eeq_O2, p.i_L) if include_ORR else 0.0
    components = {
        "anodic_Fe": ia,
        "HER": (iH if include_HER else np.zeros_like(Eint)),
        "ORR": (iO2 if include_ORR else np.zeros_like(Eint)),
    }
    imix = ia + (iH if include_HER else 0.0) + (iO2 if include_ORR else 0.0)
    return imix, components


def predict_current_for_measured_potential(Em: np.ndarray, p: FlittParams, n_iter: int = 5,
                                           include_HER: bool = True,
                                           include_ORR: bool = True):
    # Iterate to include ohmic drop: Eint = Em - i*R_s
    Eint = Em.copy()
    i_pred = np.zeros_like(Em)
    comps = {}
    for _ in range(max(1, n_iter)):
        i_pred, comps = partial_currents_at_internal_potential(Eint, p, include_HER, include_ORR)
        Eint = Em - i_pred * p.R_s
    i_pred = i_pred + p.i_offset

    # Ensure finite
    i_pred = np.nan_to_num(i_pred, nan=0.0, posinf=1e308, neginf=-1e308)
    for k in list(comps.keys()):
        comps[k] = np.nan_to_num(comps[k], nan=0.0, posinf=1e308, neginf=-1e308)
    return i_pred, comps


def pack_params(p: FlittParams) -> np.ndarray:
    return np.array([
        p.log_i0_a, p.log_i0_H, p.log_i0_O2,
        p.b_a, p.b_H, p.b_O2,
        p.Eeq_a, p.Eeq_H, p.Eeq_O2,
        p.i_L, p.R_s, p.i_offset
    ], dtype=float)


def unpack_params(x: np.ndarray) -> FlittParams:
    return FlittParams(
        log_i0_a=float(x[0]),
        log_i0_H=float(x[1]),
        log_i0_O2=float(x[2]),
        b_a=float(x[3]),
        b_H=float(x[4]),
        b_O2=float(x[5]),
        Eeq_a=float(x[6]),
        Eeq_H=float(x[7]),
        Eeq_O2=float(x[8]),
        i_L=float(x[9]),
        R_s=float(x[10]),
        i_offset=float(x[11]),
    )


def estimate_Ecorr(E, I):
    if I.size == 0:
        return 0.0
    return float(E[np.nanargmin(np.abs(I))])


def guess_iL(E, I):
    Ec = estimate_Ecorr(E, I)
    cath_mask = E < (Ec - 0.15)
    if np.any(cath_mask):
        return float(np.nanpercentile(np.abs(I[cath_mask]), 95))
    return float(0.3 * np.nanmax(np.abs(I))) if I.size else 1e-6


def default_bounds(I: np.ndarray, E: Optional[np.ndarray] = None):
    i_abs = np.nanmax(np.abs(I)) if I.size and np.isfinite(np.nanmax(np.abs(I))) else 1.0
    i_abs = max(i_abs, 1.0)

    # Keep Eeq near observed potentials
    if E is not None and E.size:
        Emin, Emax = float(np.nanmin(E)), float(np.nanmax(E))
        E_margin = max(0.15, 0.2 * (Emax - Emin))
        E_low, E_high = Emin - E_margin, Emax + E_margin
    else:
        E_low, E_high = -1.5, 1.5

    # Tighter bounds to avoid unphysical blow-up
    lb = np.array([
        -10.0, -12.0, -12.0,  # log_i0_a, log_i0_H, log_i0_O2
        0.03,  0.06,  0.06,   # b_a, b_H, b_O2 (V/dec)
        E_low, E_low, E_low,  # Eeqs
        1e-9,  0.0,  -i_abs   # i_L, R_s, i_offset
    ], dtype=float)
    ub = np.array([
        -3.5,  -8.0,  -6.0,
        0.15,  0.18,  0.18,
        E_high, E_high, E_high,
        max(10*i_abs, 1e-6), 300.0, i_abs
    ], dtype=float)
    return lb, ub


def objective(x: np.ndarray, Em: np.ndarray, Iobs: np.ndarray, w_lin: float, w_log: float,
              include_HER: bool, include_ORR: bool) -> np.ndarray:
    p = unpack_params(x)
    Ipred, _ = predict_current_for_measured_potential(Em, p,
                                                      include_HER=include_HER,
                                                      include_ORR=include_ORR)
    # Bound predictions to keep residuals stable
    cap = max(1.0, 1e6 * (np.nanmax(np.abs(Iobs)) if Iobs.size else 1.0))
    Ipred = np.clip(Ipred, -cap, cap)

    res_lin = (Ipred - Iobs)
    # log residual on |I|
    eps = 1e-20
    mask = (np.abs(Iobs) > 10*eps) & (np.abs(Ipred) > 10*eps)
    res_log = np.zeros_like(Iobs)
    res_log[mask] = (np.log10(np.abs(Ipred[mask])) - np.log10(np.abs(Iobs[mask])))

    scale = max(np.nanmax(np.abs(Iobs)), 1.0)
    res = np.concatenate([w_lin * res_lin / scale, w_log * res_log])

    # Ensure finite residuals
    return np.nan_to_num(res, nan=1e6, posinf=1e6, neginf=-1e6)


def fit_flitt(Em: np.ndarray,
              Iobs: np.ndarray,
              p0: FlittParams,
              bounds,
              w_lin: float = 1.0,
              w_log: float = 1.0,
              include_HER: bool = True,
              include_ORR: bool = True):
    x0 = pack_params(p0)
    res = least_squares(
        objective, x0, bounds=bounds,
        args=(Em, Iobs, w_lin, w_log, include_HER, include_ORR),
        method="trf", max_nfev=5000, ftol=1e-10, xtol=1e-10
    )
    p_fit = unpack_params(res.x)
    info = {"success": res.success, "message": res.message, "cost": res.cost, "nfev": res.nfev}
    # Approximate parameter SEs from Jacobian
    try:
        J = res.jac
        _, s, VT = np.linalg.svd(J, full_matrices=False)
        threshold = np.finfo(float).eps * max(J.shape) * s[0]
        s = s[s > threshold]
        VT = VT[:s.size]
        cov = (VT.T / (s**2)) @ VT
        dof = max(1, len(res.fun) - len(res.x))
        residual_var = 2 * res.cost / dof
        cov *= residual_var
        info["param_stderr"] = np.sqrt(np.diag(cov))
    except Exception:
        info["param_stderr"] = None
    return p_fit, info


# ============================
# Streamlit UI
# ============================
st.set_page_config(page_title="Polarization Fit (Flitt & Schweinsberg)", page_icon="🧪", layout="wide")
st.title("🧪 Polarization Curve Fit — Flitt & Schweinsberg deconstruction")

st.markdown(
    "Upload your LSV file (CSV or Excel), select the potential and current columns, "
    "and fit a mixed-current model: anodic Fe dissolution (Tafel), HER (Tafel), "
    "and ORR (Tafel mixed with diffusion limit). Include optional iR-drop."
)

# ---- Upload
uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
if uploaded is None:
    st.info("Please upload a file to begin.")
    st.stop()

df = load_table(uploaded)
st.success(f"Loaded {uploaded.name} — {df.shape[0]} rows × {df.shape[1]} columns")

# ---- Column selection
default_pot_cols = ["WE(1).Potential (V)", "Potential applied (V)"]
default_cur_cols = ["WE(1).Current (A)"]
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

pot_guess = next((c for c in default_pot_cols if c in df.columns), (num_cols[0] if num_cols else df.columns[0]))
# fallback for current column
if len(df.columns) > 1:
    cur_guess = next((c for c in default_cur_cols if c in df.columns),
                     (num_cols[1] if len(num_cols) > 1 else df.columns[1]))
else:
    cur_guess = pot_guess  # degenerate case; user must choose

c1, c2, c3 = st.columns([1.6, 1.6, 1])
pot_col = c1.selectbox("Potential column", options=df.columns.tolist(), index=df.columns.get_loc(pot_guess))
cur_col = c2.selectbox("Current column", options=df.columns.tolist(), index=df.columns.get_loc(cur_guess))
invert_I = c3.checkbox("Invert current sign", value=False, help="Tick if cathodic currents appear positive in your data.")

# ---- Pre-processing
row_range = st.slider("Row range to include", 0, len(df)-1, (0, len(df)-1))
use_current_density = st.checkbox("Use current density (A/cm²)")
area_cm2 = st.number_input("Electrode area (cm²)", value=1.0, min_value=1e-9, step=0.1,
                           disabled=not use_current_density, format="%.6f")

E_all = pd.to_numeric(df[pot_col], errors="coerce").to_numpy()
I_all = pd.to_numeric(df[cur_col], errors="coerce").to_numpy()
mask = np.isfinite(E_all) & np.isfinite(I_all)
E_all, I_all = E_all[mask], I_all[mask]
E = E_all[row_range[0]:row_range[1] + 1]
I = I_all[row_range[0]:row_range[1] + 1]
if invert_I:
    I = -I
if use_current_density:
    I = I / float(area_cm2)

if E.size < 5:
    st.error("Not enough valid points after filtering. Check your column selection and row range.")
    st.stop()

with st.expander("Data preview"):
    prev = df.iloc[row_range[0]:row_range[1]+1][[pot_col, cur_col]].head(20).copy()
    if invert_I:
        prev[cur_col] = -prev[cur_col]
    st.dataframe(prev, use_container_width=True)

# ---- Include/exclude processes
st.subheader("Processes to include")
c_inc1, c_inc2 = st.columns(2)
include_HER = c_inc1.checkbox("Include HER", value=True)
include_ORR = c_inc2.checkbox("Include ORR", value=True)

# ---- Initial guesses
p0 = FlittParams()
# i_L guess from cathodic tail
p0.i_L = max(1e-9, guess_iL(E, I))

st.subheader("Model parameters (set and optionally lock)")
def param_input(label, value, step, fmt="%.5f", help_txt="", lock_key=None):
    cols = st.columns([2.2, 1])
    with cols[0]:
        val = st.number_input(label, value=value, step=step, format=fmt, help=help_txt, key=f"val_{label}")
    with cols[1]:
        fix = st.checkbox("lock", value=False, key=lock_key or f"lock_{label}")
    return val, fix

# log10(i0)
p0.log_i0_a, lock_log_i0_a = param_input("log10(i0_a)", p0.log_i0_a, 0.1, fmt="%.2f", help_txt="Fe dissolution")
p0.log_i0_H, lock_log_i0_H = param_input("log10(i0_H)", p0.log_i0_H, 0.1, fmt="%.2f", help_txt="Hydrogen evolution")
p0.log_i0_O2, lock_log_i0_O2 = param_input("log10(i0_O2)", p0.log_i0_O2, 0.1, fmt="%.2f", help_txt="Oxygen reduction")
# Tafel slopes
p0.b_a, lock_b_a = param_input("b_a (V/dec)", p0.b_a, 0.01)
p0.b_H, lock_b_H = param_input("b_H (V/dec)", p0.b_H, 0.01)
p0.b_O2, lock_b_O2 = param_input("b_O2 (V/dec)", p0.b_O2, 0.01)
# Eeq
p0.Eeq_a, lock_Eeq_a = param_input("Eeq_a (V)", p0.Eeq_a, 0.01)
p0.Eeq_H, lock_Eeq_H = param_input("Eeq_H (V)", p0.Eeq_H, 0.01, help_txt="≈ 0.000 − 0.0591·pH vs SHE (adjust for your reference)")
p0.Eeq_O2, lock_Eeq_O2 = param_input("Eeq_O2 (V)", p0.Eeq_O2, 0.01, help_txt="≈ 1.229 − 0.0591·pH vs SHE (adjust for your reference)")
# iL, Rs, offset
p0.i_L, lock_i_L = param_input("i_L (A)", float(p0.i_L), max(1e-9, 0.1*abs(p0.i_L) if p0.i_L else 1e-9), fmt="%.6e",
                               help_txt="Magnitude (>0) of ORR limiting current")
p0.R_s, lock_R_s = param_input("R_s (Ohm)", p0.R_s, 0.1)
p0.i_offset, lock_i_offset = param_input("i_offset (A)", p0.i_offset, 1e-6, fmt="%.6e")

# Loss weights
cW1, cW2 = st.columns(2)
w_lin = cW1.slider("Weight: linear residual", 0.0, 5.0, 1.0, 0.1)
w_log = cW2.slider("Weight: log10(|I|) residual", 0.0, 5.0, 1.0, 0.1)

# Bounds and locks
lb, ub = default_bounds(I, E)
names = ["log_i0_a","log_i0_H","log_i0_O2","b_a","b_H","b_O2","Eeq_a","Eeq_H","Eeq_O2","i_L","R_s","i_offset"]
locks = [lock_log_i0_a, lock_log_i0_H, lock_log_i0_O2, lock_b_a, lock_b_H, lock_b_O2,
         lock_Eeq_a, lock_Eeq_H, lock_Eeq_O2, lock_i_L, lock_R_s, lock_i_offset]
for i, (name, locked) in enumerate(zip(names, locks)):
    if locked:
        val = getattr(p0, name)
        lb[i] = val
        ub[i] = val

# ---- Fit
cfit1, cfit2 = st.columns([1, 1])
run = cfit1.button("Run fit", type="primary", use_container_width=True)
if not run:
    st.stop()

with st.spinner("Fitting model..."):
    p_fit, info = fit_flitt(E, I, p0=p0, bounds=(lb, ub),
                            w_lin=w_lin, w_log=w_log,
                            include_HER=include_HER, include_ORR=include_ORR)
st.success(info.get("message", "Done"))

# ---- Results table
vals = pack_params(p_fit)
stderr = info.get("param_stderr")
rows = []
for i, name in enumerate(names):
    se = (stderr[i] if (stderr is not None and i < len(stderr) and np.isfinite(stderr[i])) else np.nan)
    rows.append({"parameter": name, "value": vals[i], "stderr": se})
st.subheader("Fitted parameters")
st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ---- Plots and diagnostics
Ipred, comps = predict_current_for_measured_potential(E, p_fit,
                                                      include_HER=include_HER,
                                                      include_ORR=include_ORR)

fig = plt.figure(figsize=(11, 8))
# Linear I–E
ax1 = plt.subplot(2, 2, 1)
ax1.plot(E, I, "k.", ms=3, label="Data")
ax1.plot(E, Ipred, "r-", lw=2, label="Fit")
ax1.set_xlabel("Potential (V)"); ax1.set_ylabel("Current (A/cm²)" if use_current_density else "Current (A)")
ax1.legend(); ax1.grid(True, alpha=0.3); ax1.set_title("Polarization fit")
# Semilog |I|
ax2 = plt.subplot(2, 2, 2)
ax2.semilogy(E, np.abs(I)+1e-30, "k.", ms=3, label="|Data|")
ax2.semilogy(E, np.abs(Ipred)+1e-30, "r-", lw=2, label="|Fit|")
ax2.set_xlabel("Potential (V)"); ax2.set_ylabel("|Current| (A/cm²)" if use_current_density else "|Current| (A)")
ax2.legend(); ax2.grid(True, which="both", alpha=0.3)
# Components
ax3 = plt.subplot(2, 2, 3)
for name, arr in comps.items():
    ax3.plot(E, arr, lw=1.8, label=name)
ax3.plot(E, Ipred, "k--", lw=1.2, label="Total")
ax3.set_xlabel("Potential (V)"); ax3.set_ylabel("Current (A/cm²)" if use_current_density else "Current (A)")
ax3.legend(); ax3.grid(True, alpha=0.3); ax3.set_title("Partial currents")
# Residuals
ax4 = plt.subplot(2, 2, 4)
ax4.plot(E, Ipred - I, "b-", lw=1.2)
ax4.axhline(0, color="k", lw=0.8)
ax4.set_xlabel("Potential (V)"); ax4.set_ylabel("Residual (A/cm²)" if use_current_density else "Residual (A)")
ax4.grid(True, alpha=0.3)

plt.tight_layout()
st.subheader("Diagnostics")
st.pyplot(fig, clear_figure=False)

# Ecorr/Icorr estimate from the fitted model
try:
    Egrid = np.linspace(min(E)-0.5, max(E)+0.5, 4000)
    Igrid, _ = predict_current_for_measured_potential(Egrid, p_fit,
                                                      include_HER=include_HER,
                                                      include_ORR=include_ORR)
    idx = np.argmin(np.abs(Igrid))
    Ecorr_est, Icorr_est = Egrid[idx], Igrid[idx]
    st.info(f"Estimated E_corr ≈ {Ecorr_est:.6f} V, I_corr ≈ {Icorr_est:.3e} "
            f"{'A/cm²' if use_current_density else 'A'} (model)")
except Exception:
    pass

# ---- Downloads
st.subheader("Downloads")
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
buf.seek(0)

out_df = pd.DataFrame({"E": E, "I_obs": I, "I_fit": Ipred, **{k: v for k, v in comps.items()}})
param_series = pd.Series({k: v for k, v in zip(names, vals)})

cD1, cD2, cD3 = st.columns(3)
with cD1:
    st.download_button("Plot (PNG)", data=buf, file_name="polarization_fit.png", mime="image/png")
with cD2:
    st.download_button("Parameters (JSON)", data=param_series.to_json(indent=2).encode("utf-8"),
                       file_name="fit_params.json", mime="application/json")
with cD3:
    st.download_button("Fitted curve (CSV)", data=out_df.to_csv(index=False).encode("utf-8"),
                       file_name="fit_curve.csv", mime="text/csv")

st.caption("Tip: Lock parameters you know (e.g., R_s from EIS, i_L from hydrodynamics, Eeq from pH/reference).")
