"""Control-aware thermal design windows for MRR WDM CPO links.

This is the source-of-truth simulation for the IPC 2026 paper draft.
It intentionally avoids the abandoned GPU-power mapping and reports
requirements versus photonic-layer temperature rise, residual detuning,
Q factor, WDM spacing, and electrical slicer SNR.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr


C_LIGHT = 299_792_458.0
CENTER_WL_NM = 1550.0
CENTER_WL_M = CENTER_WL_NM * 1e-9
DNDT_SI = 1.86e-4
N_GROUP_SI = 4.0
DLAMBDA_DT_NM_PER_C = CENTER_WL_NM * DNDT_SI / N_GROUP_SI
FEC_HD = 3.8e-3
N_CHANNELS = 8
DEFAULT_Q_VALUES = [3000, 5200, 10000, 15000]
DEFAULT_SNR_VALUES = [20, 25, 30, 35]
DEFAULT_SPACING_VALUES = [50, 100, 200]
PAM4_LEVELS = np.array([0.0, 1.0, 2.0, 3.0])
PAM4_BITS = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.int8)
PAM4_MEAN = float(np.mean(PAM4_LEVELS))
PAM4_VAR = float(np.var(PAM4_LEVELS))
ITIO_TUNABILITY_NM_PER_V = 0.589
ITIO_RANGE_NM = 1.5
THERMAL_TUNING_MW_PER_NM = 4.0
PACKAGE_CASES = [
    # OIF estimates a 3.2T dense optical engine at 32 W in about
    # 20 mm x 20 mm, motivating CPO stress scenarios with tens of
    # degrees of local photonic-layer rise. The exact spatial profile
    # below is a disclosed edge-hotspot scenario, not measured data.
    ("mild edge", 8.0, 4.0, 0.8),
    ("nominal edge", 16.0, 8.0, 1.8),
    ("severe edge", 28.0, 14.0, 3.2),
]


def parse_number_list(raw: str, cast=float) -> list:
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def spacing_nm(spacing_ghz: float) -> float:
    return (CENTER_WL_M**2 / C_LIGHT) * (spacing_ghz * 1e9) * 1e9


def fwhm_nm(q_factor: float) -> float:
    return CENTER_WL_NM / q_factor


def channel_offsets_nm(spacing_ghz: float, n_channels: int = N_CHANNELS) -> np.ndarray:
    offsets = np.arange(n_channels, dtype=float) - (n_channels - 1) / 2
    return offsets * spacing_nm(spacing_ghz)


def lorentzian_drop(delta_nm: np.ndarray | float, linewidth_nm: float) -> np.ndarray | float:
    return 1.0 / (1.0 + (2.0 * np.asarray(delta_nm) / linewidth_nm) ** 2)


def channel_gains(q_factor: float, spacing_ghz: float, channel: int, residual_nm: float) -> tuple[float, np.ndarray]:
    offsets = channel_offsets_nm(spacing_ghz)
    linewidth = fwhm_nm(q_factor)
    ring_resonance = offsets[channel] + residual_nm
    deltas = offsets - ring_resonance
    gains = lorentzian_drop(deltas, linewidth)
    desired = float(gains[channel])
    leakage = np.delete(gains, channel).astype(float)
    return desired, leakage


def hamming_distance(tx_idx: int, dec_idx: int) -> int:
    return int(np.count_nonzero(PAM4_BITS[tx_idx] != PAM4_BITS[dec_idx]))


HAMMING = np.array([[hamming_distance(i, j) for j in range(4)] for i in range(4)], dtype=float)


def interference_distribution(leakage_gains: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.array([0.0])
    probs = np.array([1.0])
    for gain in leakage_gains:
        values = (values[:, None] + gain * PAM4_LEVELS[None, :]).ravel()
        probs = (probs[:, None] * 0.25).ravel()
    return values, probs


def exact_gray_pam4_ber(
    desired_gain: float,
    leakage_gains: np.ndarray,
    snr_db: float,
    track_pedestal: bool = True,
) -> float:
    """Exact IM/DD PAM4 bit error rate with discrete WDM leakage and AWGN.

    SNR is the electrical slicer SNR for an aligned, unit-gain PAM4 channel:
    SNR = Var(x) / sigma^2. Adjacent WDM channels leak as positive
    intensity/current terms weighted by the Lorentzian drop-port response.
    By default, slicer thresholds include the mean leakage pedestal, leaving
    the random data-dependent leakage as crosstalk noise. Setting
    track_pedestal=False gives a conservative fixed-threshold receiver.
    """

    sigma = math.sqrt(PAM4_VAR / (10.0 ** (snr_db / 10.0)))
    interference_values, interference_probs = interference_distribution(leakage_gains)
    leakage_pedestal = PAM4_MEAN * float(np.sum(leakage_gains))
    thresholds = desired_gain * np.array([0.5, 1.5, 2.5])
    if track_pedestal:
        thresholds = thresholds + leakage_pedestal
    total_bit_errors = 0.0

    for tx_idx, tx_level in enumerate(PAM4_LEVELS):
        means = desired_gain * tx_level + interference_values
        z0 = (thresholds[0] - means) / sigma
        z1 = (thresholds[1] - means) / sigma
        z2 = (thresholds[2] - means) / sigma
        p0 = ndtr(z0)
        p1 = ndtr(z1) - ndtr(z0)
        p2 = ndtr(z2) - ndtr(z1)
        p3 = 1.0 - ndtr(z2)
        decision_probs = np.array(
            [
                np.sum(interference_probs * p0),
                np.sum(interference_probs * p1),
                np.sum(interference_probs * p2),
                np.sum(interference_probs * p3),
            ]
        )
        total_bit_errors += 0.25 * float(np.dot(decision_probs, HAMMING[tx_idx]))

    return total_bit_errors / 2.0


def representative_ber(
    q_factor: float,
    snr_db: float,
    spacing_ghz: float,
    residual_linewidths: float,
    channel: int | None = None,
    track_pedestal: bool = True,
) -> float:
    if channel is None:
        channel = N_CHANNELS // 2 - 1
    residual_nm = residual_linewidths * fwhm_nm(q_factor)
    ber_values = []
    for sign in (-1.0, 1.0):
        desired, leakage = channel_gains(q_factor, spacing_ghz, channel, sign * residual_nm)
        ber_values.append(exact_gray_pam4_ber(desired, leakage, snr_db, track_pedestal))
    return max(ber_values)


def detuning_tolerance(
    q_factor: float,
    snr_db: float,
    spacing_ghz: float,
    fec_threshold: float,
    residual_grid: np.ndarray,
    track_pedestal: bool = True,
) -> tuple[float, bool]:
    previous_residual = float(residual_grid[0])
    previous_ber = representative_ber(q_factor, snr_db, spacing_ghz, previous_residual, track_pedestal=track_pedestal)
    if previous_ber > fec_threshold:
        return 0.0, False

    for residual in residual_grid[1:]:
        residual = float(residual)
        ber = representative_ber(q_factor, snr_db, spacing_ghz, residual, track_pedestal=track_pedestal)
        if ber > fec_threshold:
            log_prev = math.log10(max(previous_ber, 1e-300))
            log_next = math.log10(max(ber, 1e-300))
            log_fec = math.log10(fec_threshold)
            if log_next == log_prev:
                return previous_residual, False
            alpha = (log_fec - log_prev) / (log_next - log_prev)
            alpha = min(1.0, max(0.0, alpha))
            return previous_residual + alpha * (residual - previous_residual), False
        previous_residual = residual
        previous_ber = ber

    return float(residual_grid[-1]), True


def monte_carlo_true_ber(
    q_factor: float,
    snr_db: float,
    spacing_ghz: float,
    residual_linewidths: np.ndarray,
    n_symbols: int,
    seed: int,
    channel: int | None = None,
) -> list[dict]:
    if channel is None:
        channel = N_CHANNELS // 2 - 1

    rng = np.random.default_rng(seed)
    tx_idx = rng.integers(0, 4, n_symbols)
    interferer_idx = rng.integers(0, 4, (n_symbols, N_CHANNELS - 1))
    noise_unit = rng.normal(0.0, 1.0, n_symbols)
    sigma = math.sqrt(PAM4_VAR / (10.0 ** (snr_db / 10.0)))
    rows = []

    for residual_lw in residual_linewidths:
        desired, leakage = channel_gains(
            q_factor,
            spacing_ghz,
            channel,
            float(residual_lw) * fwhm_nm(q_factor),
        )
        tx_symbols = PAM4_LEVELS[tx_idx]
        interferers = PAM4_LEVELS[interferer_idx]
        y = desired * tx_symbols + np.dot(interferers, leakage) + sigma * noise_unit
        leakage_pedestal = PAM4_MEAN * float(np.sum(leakage))
        thresholds = desired * np.array([0.5, 1.5, 2.5]) + leakage_pedestal
        dec_idx = np.digitize(y, thresholds)
        bit_errors = np.count_nonzero(PAM4_BITS[tx_idx] != PAM4_BITS[dec_idx])
        mc_ber = bit_errors / (2.0 * n_symbols)
        analytic = exact_gray_pam4_ber(desired, leakage, snr_db)
        zero_error_upper = 0.5 / (2.0 * n_symbols)
        rows.append(
            {
                "residual_linewidths": float(residual_lw),
                "analytic_ber": analytic,
                "mc_true_ber": mc_ber,
                "mc_plot_ber": max(mc_ber, zero_error_upper),
                "zero_error_upper_bound": zero_error_upper if bit_errors == 0 else "",
                "bit_errors": int(bit_errors),
                "n_symbols": n_symbols,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def package_thermal_rows_from_csv(path: str) -> list[dict]:
    rows = []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"case", "channel", "delta_t_degC"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Thermal CSV missing columns: {sorted(missing)}")
        for row in reader:
            rows.append(
                {
                    "case": row["case"],
                    "channel": int(row["channel"]),
                    "delta_t_degC": float(row["delta_t_degC"]),
                    "source": row.get("source", "user_csv"),
                }
            )
    return rows


def default_package_thermal_rows() -> list[dict]:
    """Compact CPO edge-hotspot stress map.

    This is not measured hardware data. It is a package-informed stress pattern
    intended to exercise nonuniform photonic-layer temperature rise: an edge
    baseline plus a localized nearby compute hotspot across an eight-channel
    optical engine. Users can replace it with FEA or IR data via
    --thermal-map-csv.
    """

    channels = np.arange(1, N_CHANNELS + 1, dtype=float)
    center = (N_CHANNELS + 1) / 2
    rows = []
    for name, baseline, hotspot, slope in PACKAGE_CASES:
        normalized = (channels - center) / (N_CHANNELS - 1)
        hotspot_profile = np.exp(-0.5 * ((channels - center) / 1.8) ** 2)
        delta_t = baseline + hotspot * hotspot_profile + slope * normalized
        for channel, temp in zip(channels, delta_t):
            rows.append(
                {
                    "case": name,
                    "channel": int(channel),
                    "delta_t_degC": float(temp),
                    "source": "default_edge_hotspot",
                }
            )
    return rows


def package_thermal_scenario_rows(args: argparse.Namespace, residual_window_nm: float) -> list[dict]:
    rows = package_thermal_rows_from_csv(args.thermal_map_csv) if args.thermal_map_csv else default_package_thermal_rows()
    enriched = []
    for row in rows:
        drift_nm = DLAMBDA_DT_NM_PER_C * row["delta_t_degC"]
        enriched.append(
            {
                **row,
                "drift_nm": drift_nm,
                "drift_linewidths_q5200": drift_nm / fwhm_nm(5200.0),
                "within_itio_range": drift_nm <= ITIO_RANGE_NM,
                "coarse_trim_needed": drift_nm > ITIO_RANGE_NM,
                "residual_window_nm_q5200_snr25": residual_window_nm,
                "lock_accuracy_fraction_of_drift": residual_window_nm / drift_nm if drift_nm else math.inf,
            }
        )
    return enriched


def configure_plots() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.9,
            "lines.linewidth": 1.7,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def readable_text_color(cmap_name: str, value: float, vmin: float, vmax: float) -> str:
    normalized = 0.0 if vmax == vmin else (value - vmin) / (vmax - vmin)
    normalized = min(1.0, max(0.0, normalized))
    red, green, blue, _ = plt.get_cmap(cmap_name)(normalized)
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "black" if luminance > 0.55 else "white"


def run_sanity_checks() -> None:
    assert abs(DLAMBDA_DT_NM_PER_C * 1000.0 - 72.1) < 0.5
    assert abs(spacing_nm(100.0) - 0.801) < 0.01
    assert abs(fwhm_nm(5200.0) * 1000.0 - 298.1) < 0.5
    b20 = representative_ber(5200.0, 20.0, 100.0, 0.0)
    b30 = representative_ber(5200.0, 30.0, 100.0, 0.0)
    assert b30 < b20
    b0 = representative_ber(5200.0, 25.0, 100.0, 0.0)
    b08 = representative_ber(5200.0, 25.0, 100.0, 0.8)
    assert b08 > b0


def generate_outputs(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_plots()

    q_values = parse_number_list(args.q_values, int)
    snr_source = args.snr_db_values or args.osnr_values or ",".join(map(str, DEFAULT_SNR_VALUES))
    snr_values = parse_number_list(snr_source, float)
    spacing_values = parse_number_list(args.spacing_ghz, float)
    fig_residual_grid = np.linspace(0.0, 1.0, 121)
    design_residual_grid = np.linspace(0.0, 2.0, 241)

    run_sanity_checks()

    fig1_rows = []
    fig1_snr = 25.0
    fig1_spacing = 100.0
    for q_factor in q_values:
        for residual in fig_residual_grid:
            fig1_rows.append(
                {
                    "receiver": "adaptive_pedestal",
                    "q": q_factor,
                    "snr_db": fig1_snr,
                    "spacing_ghz": fig1_spacing,
                    "residual_linewidths": float(residual),
                    "ber": representative_ber(q_factor, fig1_snr, fig1_spacing, float(residual)),
                }
            )
    for residual in fig_residual_grid:
        fig1_rows.append(
            {
                "receiver": "fixed_threshold",
                "q": 5200,
                "snr_db": fig1_snr,
                "spacing_ghz": fig1_spacing,
                "residual_linewidths": float(residual),
                "ber": representative_ber(5200, fig1_snr, fig1_spacing, float(residual), track_pedestal=False),
            }
        )

    write_csv(
        out_dir / "fig1_ber_vs_detuning.csv",
        fig1_rows,
        ["receiver", "q", "snr_db", "spacing_ghz", "residual_linewidths", "ber"],
    )

    fig, ax = plt.subplots(figsize=(3.42, 2.25))
    for q_factor in q_values:
        rows = [row for row in fig1_rows if row["q"] == q_factor and row["receiver"] == "adaptive_pedestal"]
        ax.semilogy(
            [row["residual_linewidths"] for row in rows],
            [max(row["ber"], 1e-8) for row in rows],
            label=f"Q={q_factor:g}, adaptive",
        )
    fixed_rows = [row for row in fig1_rows if row["receiver"] == "fixed_threshold"]
    ax.semilogy(
        [row["residual_linewidths"] for row in fixed_rows],
        [max(row["ber"], 1e-8) for row in fixed_rows],
        color="black",
        linestyle=":",
        linewidth=1.3,
        label="Q=5200, fixed slicer",
    )
    ax.axhline(args.fec_threshold, color="crimson", linestyle="--", linewidth=1.2, label="HD-FEC (3.8e-3)")
    ax.set_xlabel(r"Residual detuning $|\Delta\lambda|/\mathrm{FWHM}$")
    ax.set_ylabel("Gray-coded BER")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(1e-8, 1.0)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        fontsize=5.6,
        frameon=False,
        handlelength=1.6,
        columnspacing=0.85,
    )
    fig.subplots_adjust(top=0.78)
    fig.savefig(out_dir / "paper_fig1_ber_vs_detuning.png")
    plt.close(fig)

    design_rows = []
    for spacing in spacing_values:
        for q_factor in q_values:
            for snr in snr_values:
                tol, capped = detuning_tolerance(q_factor, snr, spacing, args.fec_threshold, design_residual_grid)
                design_rows.append(
                    {
                        "q": q_factor,
                        "snr_db": snr,
                        "spacing_ghz": spacing,
                        "tolerance_linewidths": tol,
                        "tolerance_pm": tol * fwhm_nm(q_factor) * 1000.0,
                        "tolerance_degC": tol * fwhm_nm(q_factor) / DLAMBDA_DT_NM_PER_C,
                        "censored_lower_bound": capped,
                    }
                )

    write_csv(
        out_dir / "fig2_design_window.csv",
        design_rows,
        ["q", "snr_db", "spacing_ghz", "tolerance_linewidths", "tolerance_pm", "tolerance_degC", "censored_lower_bound"],
    )

    fig, axes = plt.subplots(1, len(spacing_values), figsize=(6.9, 2.15), sharey=True)
    if len(spacing_values) == 1:
        axes = [axes]
    for ax, spacing in zip(axes, spacing_values):
        heat = np.zeros((len(snr_values), len(q_values)))
        for i, snr in enumerate(snr_values):
            for j, q_factor in enumerate(q_values):
                match = next(
                    row
                    for row in design_rows
                    if row["q"] == q_factor and row["snr_db"] == snr and row["spacing_ghz"] == spacing
                )
                heat[i, j] = match["tolerance_linewidths"]
        cmap_name = "viridis"
        im = ax.imshow(heat, origin="lower", aspect="auto", vmin=0.0, vmax=2.0, cmap=cmap_name)
        ax.set_title(f"{int(spacing)} GHz")
        ax.set_xticks(range(len(q_values)))
        ax.set_xticklabels([str(q) for q in q_values], rotation=35)
        ax.set_yticks(range(len(snr_values)))
        ax.set_yticklabels([f"{int(v)}" for v in snr_values])
        ax.set_xlabel("Q")
        for i, j in itertools.product(range(len(snr_values)), range(len(q_values))):
            val = heat[i, j]
            color = readable_text_color(cmap_name, val, 0.0, 2.0)
            match = next(
                row
                for row in design_rows
                if row["q"] == q_values[j] and row["snr_db"] == snr_values[i] and row["spacing_ghz"] == spacing
            )
            label = f">{val:.1f}" if match["censored_lower_bound"] else f"{val:.2f}"
            ax.text(j, i, label, ha="center", va="center", color=color, fontsize=5.8)
    axes[0].set_ylabel("Electrical SNR (dB)")
    cbar = fig.colorbar(im, ax=axes, fraction=0.030, pad=0.02)
    cbar.set_label("Max detuning / FWHM")
    fig.savefig(out_dir / "paper_fig2_design_window_heatmap.png")
    plt.close(fig)

    q_ref = 5200.0
    snr_ref = 25.0
    spacing_ref = 100.0
    tol_ref, _ = detuning_tolerance(q_ref, snr_ref, spacing_ref, args.fec_threshold, design_residual_grid)
    fixed_tol_ref, _ = detuning_tolerance(
        q_ref, snr_ref, spacing_ref, args.fec_threshold, design_residual_grid, track_pedestal=False
    )
    tol_ref_nm = tol_ref * fwhm_nm(q_ref)
    temp = np.linspace(0.0, 40.0, 201)
    drift_nm = DLAMBDA_DT_NM_PER_C * temp
    fig3_rows = []
    for t, drift in zip(temp, drift_nm):
        fig3_rows.append(
            {
                "delta_t_degC": float(t),
                "uncompensated_drift_nm": float(drift),
                "itio_gate_voltage_v": float(drift / ITIO_TUNABILITY_NM_PER_V),
                "thermal_tuning_power_mw": float(drift * THERMAL_TUNING_MW_PER_NM),
                "residual_tolerance_nm_q5200_snr25": tol_ref_nm,
                "residual_tolerance_degC_q5200_snr25": tol_ref_nm / DLAMBDA_DT_NM_PER_C,
            }
        )
    write_csv(
        out_dir / "fig3_tuning_requirements.csv",
        fig3_rows,
        [
            "delta_t_degC",
            "uncompensated_drift_nm",
            "itio_gate_voltage_v",
            "thermal_tuning_power_mw",
            "residual_tolerance_nm_q5200_snr25",
            "residual_tolerance_degC_q5200_snr25",
        ],
    )

    fig, ax = plt.subplots(figsize=(3.42, 2.15))
    ax.plot(temp, drift_nm, color="#1f77b4", label="Thermo-optic drift")
    ax.axhline(ITIO_RANGE_NM, color="#2ca02c", linestyle="--", label="1.5 nm gate range")
    ax.axhline(spacing_nm(100.0), color="gray", linestyle=":", label="100 GHz spacing")
    ax.axhspan(0, tol_ref_nm, color="crimson", alpha=0.13, label="Residual BER window")
    ax.set_xlabel("Temperature rise (C)")
    ax.set_ylabel("Wavelength correction (nm)")
    ax.set_xlim(0, 40)
    ax.set_ylim(0, max(drift_nm) * 1.04)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=5.8)
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim()[0] / ITIO_TUNABILITY_NM_PER_V, ax.get_ylim()[1] / ITIO_TUNABILITY_NM_PER_V)
    ax2.set_ylabel("ITiO gate bias (V)")
    fig.subplots_adjust(left=0.18, right=0.82, bottom=0.20, top=0.96)
    fig.savefig(out_dir / "paper_fig3_tuning_requirements.png")
    plt.close(fig)

    package_rows = package_thermal_scenario_rows(args, tol_ref_nm)
    write_csv(
        out_dir / "fig5_package_thermal_scenarios.csv",
        package_rows,
        [
            "case",
            "channel",
            "delta_t_degC",
            "source",
            "drift_nm",
            "drift_linewidths_q5200",
            "within_itio_range",
            "coarse_trim_needed",
            "residual_window_nm_q5200_snr25",
            "lock_accuracy_fraction_of_drift",
        ],
    )

    cases = list(dict.fromkeys(row["case"] for row in package_rows))
    thermal = np.zeros((len(cases), N_CHANNELS))
    for i, case in enumerate(cases):
        for row in package_rows:
            if row["case"] == case:
                thermal[i, row["channel"] - 1] = row["delta_t_degC"]

    fig, ax = plt.subplots(figsize=(3.42, 1.88))
    cmap_name = "magma"
    thermal_vmax = max(40, float(np.max(thermal)))
    im = ax.imshow(thermal, origin="lower", aspect="auto", cmap=cmap_name, vmin=0, vmax=thermal_vmax)
    ax.set_xticks(range(N_CHANNELS))
    ax.set_xticklabels([str(i) for i in range(1, N_CHANNELS + 1)])
    ax.set_yticks(range(len(cases)))
    ax.set_yticklabels(cases)
    ax.set_xlabel("WDM channel / ring")
    ax.set_ylabel("Thermal case")
    for i, j in itertools.product(range(len(cases)), range(N_CHANNELS)):
        val = thermal[i, j]
        text_color = readable_text_color(cmap_name, val, 0, thermal_vmax)
        ax.text(j, i, f"{val:.0f}", ha="center", va="center", color=text_color, fontsize=6.2)
        if DLAMBDA_DT_NM_PER_C * val > ITIO_RANGE_NM:
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="cyan", linewidth=1.1))
    cbar = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.02)
    cbar.set_label("Delta T (C)")
    fig.savefig(out_dir / "paper_fig5_package_thermal_scenarios.png")
    plt.close(fig)

    mc_residuals = np.linspace(0.35, 0.70, 15)
    mc_rows = monte_carlo_true_ber(
        q_ref,
        snr_ref,
        spacing_ref,
        mc_residuals,
        args.n_symbols,
        args.seed,
    )
    write_csv(
        out_dir / "fig4_monte_carlo_validation.csv",
        mc_rows,
        [
            "residual_linewidths",
            "analytic_ber",
            "mc_true_ber",
            "mc_plot_ber",
            "zero_error_upper_bound",
            "bit_errors",
            "n_symbols",
        ],
    )

    fig, ax = plt.subplots(figsize=(3.42, 2.15))
    ax.semilogy(
        [row["residual_linewidths"] for row in mc_rows],
        [max(row["analytic_ber"], 1e-8) for row in mc_rows],
        color="#1f77b4",
        label="Semi-analytic",
    )
    ax.semilogy(
        [row["residual_linewidths"] for row in mc_rows],
        [max(row["mc_plot_ber"], 1e-8) for row in mc_rows],
        "o",
        markersize=3.0,
        color="#d62728",
        label="Monte Carlo",
    )
    ax.axhline(args.fec_threshold, color="black", linestyle="--", linewidth=1.1, label="HD-FEC (3.8e-3)")
    ax.set_xlabel(r"Residual detuning $|\Delta\lambda|/\mathrm{FWHM}$")
    ax.set_ylabel("Gray-coded BER")
    ax.set_xlim(0.35, 0.70)
    ax.set_ylim(1e-5, 5e-2)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right")
    fig.savefig(out_dir / "paper_fig4_mc_validation.png")
    plt.close(fig)

    summary_rows = []
    for row in design_rows:
        if row["spacing_ghz"] == 100.0:
            summary_rows.append(row)
    write_csv(
        out_dir / "summary_thresholds.csv",
        summary_rows,
        ["q", "snr_db", "spacing_ghz", "tolerance_linewidths", "tolerance_pm", "tolerance_degC", "censored_lower_bound"],
    )

    comparison_rows = [
        {
            "quantity": "thermo_optic_drift",
            "value": DLAMBDA_DT_NM_PER_C * 1000.0,
            "unit": "pm_per_degC",
        },
        {"quantity": "q5200_fwhm", "value": fwhm_nm(5200.0) * 1000.0, "unit": "pm"},
        {
            "quantity": "adaptive_residual_window_q5200_snr25_100ghz",
            "value": tol_ref * fwhm_nm(5200.0) * 1000.0,
            "unit": "pm",
        },
        {
            "quantity": "fixed_slicer_residual_window_q5200_snr25_100ghz",
            "value": fixed_tol_ref * fwhm_nm(5200.0) * 1000.0,
            "unit": "pm",
        },
        {
            "quantity": "itio_voltage_for_adaptive_window",
            "value": tol_ref * fwhm_nm(5200.0) / ITIO_TUNABILITY_NM_PER_V,
            "unit": "V",
        },
        {"quantity": "itio_temperature_range", "value": ITIO_RANGE_NM / DLAMBDA_DT_NM_PER_C, "unit": "degC"},
        {
            "quantity": "default_package_severe_peak_temperature_rise",
            "value": max(row["delta_t_degC"] for row in package_rows if row["case"] == cases[-1]),
            "unit": "degC",
        },
        {
            "quantity": "default_package_severe_peak_drift",
            "value": max(row["drift_nm"] for row in package_rows if row["case"] == cases[-1]),
            "unit": "nm",
        },
    ]
    write_csv(out_dir / "design_comparison.csv", comparison_rows, ["quantity", "value", "unit"])

    print(f"Generated outputs in {out_dir}")
    print(f"dLambda/dT = {DLAMBDA_DT_NM_PER_C * 1000.0:.1f} pm/C")
    print(f"100 GHz spacing = {spacing_nm(100.0):.3f} nm")
    print(f"Q=5200 FWHM = {fwhm_nm(5200.0) * 1000.0:.1f} pm")
    print(
        "Q=5200, electrical SNR=25 dB tolerance = "
        f"{tol_ref:.2f} FWHM = {tol_ref_nm * 1000.0:.0f} pm = {tol_ref_nm / DLAMBDA_DT_NM_PER_C:.2f} C"
    )
    print(
        "Q=5200 fixed-slicer tolerance = "
        f"{fixed_tol_ref:.2f} FWHM = {fixed_tol_ref * fwhm_nm(q_ref) * 1000.0:.0f} pm"
    )


def build_parser() -> argparse.ArgumentParser:
    default_out = Path(__file__).resolve().parent.parent / "figures" / "design_window"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(default_out), help="Output directory for figures and CSV files.")
    parser.add_argument("--seed", type=int, default=2026, help="Monte Carlo RNG seed.")
    parser.add_argument("--q-values", default=",".join(map(str, DEFAULT_Q_VALUES)), help="Comma-separated Q values.")
    parser.add_argument(
        "--snr-db-values",
        default=None,
        help="Comma-separated electrical slicer SNR values in dB.",
    )
    parser.add_argument("--osnr-values", default=None, help="Deprecated alias for --snr-db-values.")
    parser.add_argument(
        "--spacing-ghz",
        default=",".join(map(str, DEFAULT_SPACING_VALUES)),
        help="Comma-separated WDM channel spacings in GHz.",
    )
    parser.add_argument("--n-symbols", type=int, default=200_000, help="PAM4 symbols per Monte Carlo point.")
    parser.add_argument("--fec-threshold", type=float, default=FEC_HD, help="BER threshold for HD-FEC.")
    parser.add_argument(
        "--thermal-map-csv",
        default=None,
        help="Optional package thermal map CSV with columns case,channel,delta_t_degC.",
    )
    return parser


if __name__ == "__main__":
    generate_outputs(build_parser().parse_args())
