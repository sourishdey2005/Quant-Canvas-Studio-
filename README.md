# Qanvas Studio — Quantum Circuit Visualizer

Browser-based quantum circuit playground built with **Streamlit + Qiskit**.

## Features

- Circuit editor (gate palette + per-gate configuration)
- Live simulation (statevector + probabilities + Bloch)
- Noise models: ideal, depolarizing, amplitude damping, phase damping, readout error
- Timeline mode: step-by-step evolution (per-gate playback)
- Circuit optimization: Qiskit `transpile()` with optimization level 0–3 + optimized QASM export
- Custom gates: define your own **1-qubit / 2-qubit unitary** matrices and use them like normal gates
- Circuit composer: save **blocks** and chain them **end-to-end** using a pipeline to form a complete circuit
- Import/Export:
  - Project JSON (includes gates, custom gates, blocks, pipeline)
  - Import OpenQASM (best-effort into the editor)
  - Export OpenQASM 2/3 + circuit PNG + results JSON

## Requirements

- Python **3.10+**

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

## How to Use

### 1) Build a Circuit

- Use **Gate Palette** to add gates.
- Or use **Add Gate (Drop Menu)** in the Circuit Builder to pick a gate from a dropdown and insert it at any position.
- Use **Gate Configuration** to edit:
  - `cx/cz/swap`: set `Control Qubit`
  - `rx/ry/rz`: set `Angle (radians)`
- Use **Undo/Redo** when iterating.

### 2) Simulate

In **Settings**:

- Choose qubit count, shots, noise model.
- Optional:
  - `Enable timeline (step-by-step)`
  - `Optimize circuit (transpile)` + optimization level
  - `Use pipeline (end-to-end blocks)` (if you built a pipeline)

Click **Run Simulation**.

### 3) Visualizations

The right panel contains tabs:

- **Bloch**: Bloch sphere for up to 3 qubits
- **Probabilities**: basis-state probability bars (+ measurement pie if enabled)
- **Amplitudes**: complex-plane scatter + amplitude table (Re/Im/phase/prob)
- **Density**: |ρᵢⱼ| heatmap + metrics (depth/size/width + gate counts)
- **Timeline**: step slider for state evolution (when enabled)
- **Code**: OpenQASM 2/3 and optimized variants (when enabled)

### 4) Custom Gates

Sidebar → **Custom Gates**:

- Choose arity (1-qubit or 2-qubit)
- Paste a JSON matrix:
  - Real numbers: `[[1,0],[0,1]]`
  - Complex strings: `[["0.5+0.5j","0"],["0","0.5-0.5j"]]`
  - Or `[re,im]` pairs
- Save → “Add … (custom …)” appears in the palette

Notes:
- The matrix must be **unitary** (U†U = I)
- 1-qubit = 2×2, 2-qubit = 4×4

### 5) Circuit Composer (Blocks + Pipeline)

Use this to build a **complete end-to-end circuit** from multiple parts.

- Save the current editor circuit as a **Block**
- Add blocks to the **Pipeline**
- Reorder pipeline with Up/Down
- Either:
  - **Load Composed Circuit into Editor** (to edit the full chain), or
  - Enable `Use pipeline (end-to-end blocks)` and run simulation directly

All blocks in a pipeline must use the same number of qubits.

### 6) Import / Export

- **Project JSON** (recommended): round-trip everything (gates + custom gates + blocks + pipeline)
- **Import OpenQASM**: best-effort conversion into the editor gate list (not all gate types can be mapped)
- Export OpenQASM, circuit PNG, results JSON from the sidebar when a simulation exists

## Known Limitations

- Pipeline composition currently requires all blocks to have the same `num_qubits`.
- OpenQASM import is best-effort; unsupported operations may be skipped.
- Timeline stops at measurements (no mid-circuit measurement playback yet).

## Credits

Made By Sourish Dey
