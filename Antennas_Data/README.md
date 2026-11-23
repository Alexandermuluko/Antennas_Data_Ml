# Antenna Project Scaffold

This project scaffold helps reproduce and optimize satellite/navigation antennas using EM simulation and machine learning.

Structure:
- `data/` - exported simulation CSVs and raw outputs
- `cst_projects/` - original CST files (move your `*.cst` files here)
- `scripts/` - helper scripts for data export and processing
- `notebooks/` - Jupyter notebooks for exploration
- `models/` - saved ML models
- `docs/` - thesis/report outlines and notes

Getting started (Windows + WSL):
1. Install Python 3.10+ in WSL or Windows and create a virtual environment.
2. Install requirements: `pip install -r requirements.txt`.
3. Put your CST files into `cst_projects/`.
4. Use `scripts/data_export.py` to convert simulator outputs (or use synthetic data if CST automation is not available).

See `docs/THESIS-OUTLINE.md` for suggested thesis chapters and `scripts/` for starter code.

If you want, I can wire up CST automation instructions next (requires CST Studio Suite and its Python API).
