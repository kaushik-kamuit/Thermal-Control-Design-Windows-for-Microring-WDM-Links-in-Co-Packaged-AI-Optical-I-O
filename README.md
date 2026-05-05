# Thermal Control Design Windows for Microring WDM Links in Co-Packaged AI Optical I/O

This repository contains the IPC 2026 paper draft, reproducible simulation code, and generated figures for a control-aware thermal design-window study of microring WDM links in co-packaged AI optical I/O.

## Contents

- `IPC2026_Handoff_Package/handoff/docs/ipc2026_FINAL.tex` - IEEEtran paper source.
- `IPC2026_Handoff_Package/handoff/docs/ipc2026_FINAL.pdf` - compiled two-page paper.
- `IPC2026_Handoff_Package/handoff/code/ipc2026_cpo_design_window.py` - source-of-truth simulation script.
- `IPC2026_Handoff_Package/handoff/figures/design_window/` - generated paper figures and CSV outputs.

## Reproduce Figures

```powershell
python .\IPC2026_Handoff_Package\handoff\code\ipc2026_cpo_design_window.py `
  --out-dir .\IPC2026_Handoff_Package\handoff\figures\design_window
```

The script prints numerical sanity checks including silicon thermo-optic drift, 100 GHz channel spacing near 1550 nm, the `Q=5200` linewidth, and the main adaptive/fixed-slicer residual lock budgets.

## Paper Build

```powershell
cd .\IPC2026_Handoff_Package\handoff\docs
pdflatex -interaction=nonstopmode -halt-on-error ipc2026_FINAL.tex
pdflatex -interaction=nonstopmode -halt-on-error ipc2026_FINAL.tex
```

## Scope

The model is a first-order link-level design study. It uses Lorentzian microring drop response, Gray-coded PAM4 bit counting, WDM leakage, slicer-domain SNR, and illustrative package-informed thermal stress cases. It does not claim measured CPO hardware validation.
