"""SCLC mobility analysis of dark I-V data (hole-only or electron-only devices).

Workflow (same order as the SCLC theory figures):
    1. Choose carrier type + data file, convert I -> J and take |J|.
    2. Work entirely in log10(V) - log10(J) space (both axes logarithmic).
    3. Fit, in this order and at increasing voltage
         (a) Ohmic region          : log-log slope fixed to 1
         (b) Trap-filled region    : log-log slope m > 2 (free fit)
         (c) Trap-free SCLC region : log-log slope fixed to 2
    4. V_TFL = transition from the trap-filled region to the trap-free SCLC
       region = intersection of the trap-filled fit and the SCLC fit in the
       log-log plot (as in the second reference figure).
    5. Mobility from the Mott-Gurney law  J = 9/8 * eps0*eps_r * mu * V^2 / L^3
       using the SCLC fit, and trap density N_t = 2*eps0*eps_r*V_TFL/(q*L^2).

Edit the DEFAULT_* device parameters (or type them in the start-up window).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Device defaults (can be changed in the start-up window)
# --------------------------------------------------------------------------
DEFAULT_AREA_CM2 = 0.01
DEFAULT_THICKNESS_NM = 100.0
DEFAULT_REL_PERMITTIVITY = 3.5

EPSILON_0_F_M = 8.8541878128e-12
ELEMENTARY_CHARGE_C = 1.602176634e-19

# --------------------------------------------------------------------------
# Analysis settings
# --------------------------------------------------------------------------
BINS_PER_DECADE = 40        # data are averaged on a uniform log10(V) grid so
                            # every voltage decade carries equal weight
SLOPE_HALF_WINDOW = 3       # local slope: regression over 2*N+1 grid points
MIN_VOLTAGE_V = 0.0         # ignore |V| below this (offset-dominated points)

OHMIC_SLOPE_TOL = 0.15      # fitted slope must be 1 +/- tol
OHMIC_LOCAL_BAND = 0.30     # every local slope in the window within 1 +/- band
OHMIC_MIN_DECADES = 0.30

TRAP_SLOPE_MIN = 2.10       # "slope > 2": local slope must exceed this
TRAP_MIN_DECADES = 0.07

SCLC_SLOPE_TOL = 0.15       # fitted slope must be 2 +/- tol
SCLC_LOCAL_BAND = 0.30
SCLC_MIN_DECADES = 0.20
MIN_R2 = 0.99               # minimum R^2 of a free log-log fit in a window

# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------
@dataclass
class Region:
    """A fitted straight line  log10(J) = slope*log10(V) + intercept."""

    name: str
    start: int                 # index into the log-grid arrays (inclusive)
    end: int                   # index (exclusive)
    slope: float               # slope used for the line (1, m, or 2)
    intercept: float
    free_slope: float          # unconstrained slope, diagnostic
    r_squared: float           # R^2 of the line that is drawn
    v_min: float
    v_max: float

    def log_j(self, log_v):
        return self.slope * np.asarray(log_v) + self.intercept


@dataclass
class BranchAnalysis:
    polarity: int
    voltage: np.ndarray          # raw |V| (sorted)
    current_density: np.ndarray  # raw |J| in mA/cm^2
    log_v: np.ndarray            # log-grid
    log_j: np.ndarray            # (mean log10 J in A/m^2 per bin)
    local_slope: np.ndarray
    ohmic: Region | None
    trap: Region | None
    sclc: Region | None


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------
def read_iv_file(filename: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the first two numeric columns (voltage, current); skip text headers."""
    try:
        table = pd.read_csv(
            filename, sep=None, engine="python", header=None, comment="#",
            encoding="utf-8-sig",
        )
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        raise ValueError(f"Could not read the selected data file: {exc}") from exc
    if table.shape[1] < 2:
        raise ValueError("The data file must contain voltage and current columns.")

    v = pd.to_numeric(table.iloc[:, 0], errors="coerce").to_numpy(float)
    i = pd.to_numeric(table.iloc[:, 1], errors="coerce").to_numpy(float)
    ok = np.isfinite(v) & np.isfinite(i)
    v, i = v[ok], i[ok]
    if v.size < 20:
        raise ValueError("Fewer than 20 valid voltage/current rows were found.")
    return v, i


# --------------------------------------------------------------------------
# Numerical helpers (all in log10 space)
# --------------------------------------------------------------------------
def _linfit(x: np.ndarray, y: np.ndarray, slope: float | None = None):
    """Least-squares line; slope=None -> free fit, else slope held fixed.

    Returns (slope, intercept, R^2 of the returned line)."""
    if slope is None:
        slope, intercept = np.polyfit(x, y, 1)
    else:
        intercept = float(np.mean(y - slope * x))
    resid = y - (slope * x + intercept)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 if ss_tot == 0 else 1.0 - float(np.sum(resid**2)) / ss_tot
    return float(slope), float(intercept), r2


def _bin_on_log_grid(v: np.ndarray, j: np.ndarray):
    """Average log10(V), log10(J) inside uniform log10(V) bins."""
    lv, lj = np.log10(v), np.log10(j)
    idx = np.floor((lv - lv.min()) * BINS_PER_DECADE).astype(int)
    xs, ys = [], []
    for b in np.unique(idx):
        sel = idx == b
        xs.append(lv[sel].mean())
        ys.append(lj[sel].mean())
    return np.array(xs), np.array(ys)


def _local_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """d(log J)/d(log V) from a sliding linear regression."""
    s = np.full(x.size, np.nan)
    for k in range(x.size):
        a, b = max(0, k - SLOPE_HALF_WINDOW), min(x.size, k + SLOPE_HALF_WINDOW + 1)
        if b - a >= 3:
            s[k] = np.polyfit(x[a:b], y[a:b], 1)[0]
    return s


def _make_region(name, x, y, v_lin, start, end, fixed_slope):
    slope, intercept, r2 = _linfit(x[start:end], y[start:end], fixed_slope)
    free_slope = _linfit(x[start:end], y[start:end], None)[0]
    return Region(
        name=name, start=start, end=end, slope=slope, intercept=intercept,
        free_slope=free_slope, r_squared=r2,
        v_min=float(10 ** x[start]), v_max=float(10 ** x[end - 1]),
    )


def _find_window(x, y, s, lo, hi, target, slope_tol, local_band, min_decades):
    """Longest window [a, b) inside [lo, hi) whose free log-log fit has
    slope = target +/- slope_tol, R^2 >= MIN_R2 and whose local slopes all stay
    within target +/- local_band.  Ties -> fit slope closest to target."""
    best, best_key = None, None
    for a in range(lo, hi):
        if not np.isfinite(s[a]) or abs(s[a] - target) > local_band:
            continue
        for b in range(a + 4, hi + 1):
            if not np.all(np.abs(s[a:b] - target) <= local_band):
                break
            if x[b - 1] - x[a] < min_decades:
                continue
            m, _, r2 = _linfit(x[a:b], y[a:b], None)
            if abs(m - target) <= slope_tol and r2 >= MIN_R2:
                key = (round(x[b - 1] - x[a], 3), -abs(m - target))
                if best_key is None or key > best_key:
                    best, best_key = (a, b), key
    return best


# --------------------------------------------------------------------------
# Branch analysis
# --------------------------------------------------------------------------
def analyze_branch(voltage, current, polarity, area_cm2) -> BranchAnalysis | None:
    """Convert to |J|, build the log-log curve and fit the three regimes."""
    # Keep only physical points: current has the same sign as the voltage.
    keep = (
        (np.sign(voltage) == polarity)
        & (np.sign(current) == polarity)
        & (np.abs(voltage) >= max(MIN_VOLTAGE_V, 1e-12))
    )
    v, i = np.abs(voltage[keep]), np.abs(current[keep])
    if v.size < 20:
        return None

    order = np.argsort(v)
    v, i = v[order], i[order]
    uniq, inv = np.unique(v, return_inverse=True)        # merge duplicate V
    j_si = np.bincount(inv, weights=i) / np.bincount(inv) / (area_cm2 * 1e-4)  # A/m^2
    v = uniq
    if v.size < 20:
        return None

    x, y = _bin_on_log_grid(v, j_si)
    if x.size < 12:
        return None
    s = _local_slopes(x, y)
    n = x.size

    # (a) Ohmic: lowest-voltage window with fitted slope ~ 1 (line forced to 1)
    ohmic = None
    first_up = np.flatnonzero(np.nan_to_num(s) > 1 + OHMIC_LOCAL_BAND + 0.5)
    ohmic_hi = int(first_up[0]) if first_up.size else n
    win = _find_window(x, y, s, 0, ohmic_hi, 1.0, OHMIC_SLOPE_TOL,
                       OHMIC_LOCAL_BAND, OHMIC_MIN_DECADES)
    if win:
        ohmic = _make_region("Ohmic", x, y, v, win[0], win[1], 1.0)

    # (b) Trap-filled: first sustained run with local slope > 2 above ohmic
    trap = None
    lo = ohmic.end if ohmic else 0
    above = np.nan_to_num(s) > TRAP_SLOPE_MIN
    k = lo
    while k < n:
        if above[k]:
            e = k
            while e < n and above[e]:
                e += 1
            if x[e - 1] - x[k] >= TRAP_MIN_DECADES:
                trap = _make_region("Trap-filled", x, y, v, k, e, None)
                break
            k = e
        else:
            k += 1

    # (c) Trap-free SCLC: slope-2 window ABOVE the trap-filled region
    sclc = None
    if trap is not None:
        win = _find_window(x, y, s, trap.end, n, 2.0, SCLC_SLOPE_TOL,
                           SCLC_LOCAL_BAND, SCLC_MIN_DECADES)
        if win:
            sclc = _make_region("SCLC", x, y, v, win[0], win[1], 2.0)

    return BranchAnalysis(
        polarity=polarity, voltage=v, current_density=j_si / 10.0,   # mA/cm^2
        log_v=x, log_j=y, local_slope=s, ohmic=ohmic, trap=trap, sclc=sclc,
    )


def _branch_score(b: BranchAnalysis) -> float:
    score = 0.0
    if b.ohmic:
        score += 1.0 + b.ohmic.r_squared - abs(b.ohmic.free_slope - 1)
    if b.trap:
        score += 1.0
    if b.sclc:
        score += 2.0
        score += b.sclc.r_squared - abs(b.sclc.free_slope - 2)
    return score


# --------------------------------------------------------------------------
# V_TFL
# --------------------------------------------------------------------------
def _intersect(r1: Region, r2: Region):
    """Intersection of two log-log lines. Returns (V, J_log10) or None."""
    dm = r1.slope - r2.slope
    if abs(dm) < 1e-6:
        return None
    lv = (r2.intercept - r1.intercept) / dm
    return float(10 ** lv), float(r1.log_j(lv))


def determine_vtfl(b: BranchAnalysis):
    """V_TFL = trap-filled fit  x  SCLC fit.  Returns (V_TFL, log10 J) or (None, None)."""
    if b.trap is None or b.sclc is None:
        return None, None
    hit = _intersect(b.trap, b.sclc)
    return (None, None) if hit is None else hit


# --------------------------------------------------------------------------
# Report + plot
# --------------------------------------------------------------------------
def _describe(region: Region | None, label: str) -> str:
    if region is None:
        return f"{label}: not identified"
    return (
        f"{label}: {region.v_min:.3g}-{region.v_max:.3g} V | line slope "
        f"{region.slope:.3f} (free fit {region.free_slope:.3f}) | "
        f"R^2 = {region.r_squared:.4f}"
    )


def report_and_plot(b, carrier, filename, area_cm2, thickness_nm, eps_r):
    L = thickness_nm * 1e-9
    eps = EPSILON_0_F_M * eps_r
    side = "positive" if b.polarity > 0 else "negative"

    print(f"\nFile   : {filename}")
    print(f"Carrier: {carrier}-only device, {side}-voltage branch")
    print(f"Device : A = {area_cm2:g} cm^2, L = {thickness_nm:g} nm, eps_r = {eps_r:g}")
    print(_describe(b.ohmic, "1) Ohmic       (slope = 1)"))
    print(_describe(b.trap, "2) Trap-filled (slope > 2)"))
    print(_describe(b.sclc, "3) SCLC        (slope = 2)"))

    mu_cm2 = None
    if b.sclc is not None:
        # J = c V^2 (A/m^2, V) with c = 10^intercept (slope fixed to 2).
        c = 10 ** b.sclc.intercept
        mu_cm2 = 8.0 * c * L**3 / (9.0 * eps) * 1e4
        print(f"Mott-Gurney {carrier} mobility: {mu_cm2:.3e} cm^2/(V s)")
    vtfl, log_j_tfl = determine_vtfl(b)
    nt = None
    if vtfl is not None:
        nt = 2.0 * eps * vtfl / (ELEMENTARY_CHARGE_C * L**2) / 1e6
        print(f"V_TFL = {vtfl:.4g} V  (trap-filled -> SCLC transition)")
        print(f"Trap density N_t = {nt:.3e} cm^-3")
    else:
        if b.trap is None:
            why = "no trap-filled region (slope > 2) was found"
        else:
            why = (f"after the trap-filled region the slope never returns to 2 "
                   f"(no trap-free SCLC up to {b.voltage.max():.3g} V)")
        print(f"V_TFL, mobility and N_t not determined: {why}.")
        print("  -> measure to higher voltage until the log-log slope settles back to 2.")

    # ---------------- plot ----------------
    colors = {"Ohmic": "tab:blue", "Trap-filled": "tab:orange", "SCLC": "tab:red"}
    shade = {"Ohmic": "#dbe6ff", "Trap-filled": "#dcf0dc", "SCLC": "#ffe8cc"}
    fit_label = {
        "Ohmic": "Ohmic fit (slope = 1)",
        "Trap-filled": "Trap-filled fit (slope = m > 2)",
        "SCLC": "Trap-free SCLC fit (slope = 2)",
    }

    fig, (ax, ax_s) = plt.subplots(1, 2, figsize=(14, 5.8),
                                   gridspec_kw={"width_ratios": [1.5, 1]})
    ax.loglog(b.voltage, b.current_density, color="0.25", lw=1.8, label="Dark I-V (data), |J|")

    regions = [r for r in (b.ohmic, b.trap, b.sclc) if r is not None]
    xlo, xhi = np.log10(b.voltage.min()), np.log10(b.voltage.max())
    for n_r, r in enumerate(regions):
        for a_ in (ax, ax_s):
            a_.axvspan(r.v_min, r.v_max, color=shade[r.name], alpha=0.7, lw=0)
        # dashed fit: fitted range, extended to V_TFL when the line meets it
        lo, hi = np.log10(r.v_min), np.log10(r.v_max)
        if vtfl is not None:
            lt = np.log10(vtfl)
            if r.name == "Trap-filled" and lt > hi:
                hi = lt
            if r.name == "SCLC" and lt < lo:
                lo = lt
        xs = np.linspace(max(lo, xlo), min(hi, xhi), 50)
        ax.loglog(10 ** xs, 10 ** r.log_j(xs) / 10.0, "--", lw=2.2,
                  color=colors[r.name],
                  label=fit_label[r.name] if r.name != "Trap-filled"
                  else f"Trap-filled fit (slope = {r.slope:.2f} > 2)")
        ax.text(np.sqrt(r.v_min * r.v_max), 0.97 - 0.06 * n_r, r.name + (" region" if r.name != "SCLC" else " (trap-free)"),
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                color=colors[r.name], fontsize=9)

    if vtfl is not None:
        y_tfl = 10 ** log_j_tfl / 10.0
        ax.plot([vtfl], [y_tfl], "o", color="black", ms=9, zorder=5)
        ax.annotate(f"$V_{{TFL}}$ = {vtfl:.3g} V", xy=(vtfl, y_tfl),
                    xytext=(30, -50), textcoords="offset points", fontsize=12,
                    arrowprops=dict(arrowstyle="->", lw=1.6))
        ax_s.axvline(vtfl, color="black", ls="--", lw=1.2, label="$V_{TFL}$")

    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Current density (mA/cm$^2$)")
    ax.set_title(r"Dark $J$-$V$ curve with $V_{TFL}$")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9)
    if mu_cm2 is not None:
        ax.text(0.03, 0.05, f"$\\mu_{{{carrier[0]}}}$ = {mu_cm2:.2e} cm$^2$/Vs",
                transform=ax.transAxes, fontsize=11,
                bbox=dict(boxstyle="round", fc="white", ec="0.6"))

    ax_s.plot(10 ** b.log_v, b.local_slope, color="tab:blue", lw=1.8)
    ax_s.set_xscale("log")
    for val, col, lab in ((1, "tab:blue", "m = 1 (ohmic)"), (2, "tab:red", "m = 2 (SCLC)"),
                          (TRAP_SLOPE_MIN, "tab:orange", f"m > {TRAP_SLOPE_MIN:g} (trap-filled)")):
        ax_s.axhline(val, color=col, ls=":" if val != 2 else "--", lw=1.4, label=lab)
    ax_s.set_ylim(0, min(max(4.0, np.nanmax(b.local_slope) * 1.05), 8))
    ax_s.set_xlabel("Voltage (V)")
    ax_s.set_ylabel(r"Local slope  $d\log J / d\log V$")
    ax_s.set_title("Local log-log slope")
    ax_s.grid(True, which="both", ls=":", alpha=0.4)
    ax_s.legend(fontsize=8, loc="upper left")

    fig.suptitle(f"{carrier.capitalize()}-only SCLC analysis: {Path(filename).name}")
    fig.tight_layout()
    out = Path(filename).with_name(f"{Path(filename).stem}_SCLC_analysis.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {out}")
    plt.show()


# --------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------
def choose_input():
    """Start-up window: carrier type, branch, device parameters, data file."""
    root = tk.Tk()
    root.title("SCLC Mobility Analysis")
    root.resizable(False, False)

    carrier = tk.StringVar(value="hole")
    branch = tk.StringVar(value="auto")
    filename = tk.StringVar(value="")
    area = tk.StringVar(value=str(DEFAULT_AREA_CM2))
    thick = tk.StringVar(value=str(DEFAULT_THICKNESS_NM))
    eps = tk.StringVar(value=str(DEFAULT_REL_PERMITTIVITY))
    result = [None]

    ttk.Label(root, text="Carrier measurement:").pack(anchor="w", padx=14, pady=(12, 2))
    row = ttk.Frame(root)
    row.pack(anchor="w", padx=14)
    ttk.Radiobutton(row, text="Hole", variable=carrier, value="hole").pack(side="left", padx=(0, 18))
    ttk.Radiobutton(row, text="Electron", variable=carrier, value="electron").pack(side="left")

    ttk.Label(root, text="Voltage branch:").pack(anchor="w", padx=14, pady=(10, 2))
    for text, val in (("Automatic (best-defined regimes)", "auto"),
                      ("Positive voltage", "positive"), ("Negative voltage", "negative")):
        ttk.Radiobutton(root, text=text, variable=branch, value=val).pack(anchor="w", padx=24)

    ttk.Label(root, text="Device parameters:").pack(anchor="w", padx=14, pady=(10, 2))
    grid = ttk.Frame(root)
    grid.pack(anchor="w", padx=24)
    for r, (lab, var) in enumerate((("Area (cm²)", area), ("Thickness (nm)", thick),
                                    ("Relative permittivity", eps))):
        ttk.Label(grid, text=lab).grid(row=r, column=0, sticky="w", pady=1)
        ttk.Entry(grid, textvariable=var, width=12).grid(row=r, column=1, padx=8)

    ttk.Label(root, text="Data file:").pack(anchor="w", padx=14, pady=(10, 2))
    ttk.Label(root, textvariable=filename, wraplength=440).pack(anchor="w", padx=14)

    def browse():
        root.lift()
        root.update_idletasks()
        sel = filedialog.askopenfilename(
            parent=root, title="Select dark I-V data file",
            filetypes=[("Data files", "*.txt *.asc *.csv *.dat"), ("All files", "*.*")])
        if sel:
            filename.set(sel)

    def analyze():
        if not filename.get():
            messagebox.showwarning("Select a data file", "Choose an I-V data file first.", parent=root)
            browse()
            return
        try:
            params = (float(area.get()), float(thick.get()), float(eps.get()))
            if min(params) <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("Device parameters", "Area, thickness and permittivity must be positive numbers.", parent=root)
            return
        result[0] = (filename.get(), carrier.get(), branch.get(), *params)
        root.destroy()

    btn = ttk.Frame(root)
    btn.pack(fill="x", padx=14, pady=14)
    ttk.Button(btn, text="Browse…", command=browse).pack(side="left")
    ttk.Button(btn, text="Analyze", command=analyze).pack(side="right")
    ttk.Button(btn, text="Cancel", command=root.destroy).pack(side="right", padx=(0, 8))

    root.after(250, browse)      # open the file picker straight away
    root.mainloop()
    return result[0]


def main() -> None:
    selection = choose_input()
    if selection is None:
        return
    filename, carrier, requested, area, thickness, eps_r = selection

    try:
        voltage, current = read_iv_file(filename)
        branches = [r for p in (1, -1)
                    if (r := analyze_branch(voltage, current, p, area)) is not None]
        if not branches:
            raise ValueError("No usable positive- or negative-voltage sweep was found.")
        if requested == "auto":
            chosen = max(branches, key=_branch_score)
        else:
            pol = 1 if requested == "positive" else -1
            chosen = next((r for r in branches if r.polarity == pol), None)
            if chosen is None:
                raise ValueError(f"No usable {requested}-voltage sweep was found.")
        report_and_plot(chosen, carrier, filename, area, thickness, eps_r)
    except (ValueError, FloatingPointError) as exc:
        messagebox.showerror("SCLC analysis error", str(exc))
        raise


if __name__ == "__main__":
    main()
