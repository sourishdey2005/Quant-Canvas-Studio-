"""
Quantum Circuit Designer
Streamlit-based interactive quantum circuit builder and simulator

License: MIT
Python Version: 3.10+
"""

import json
import time
from typing import Optional, List, Dict, Any
from enum import Enum
from dataclasses import dataclass, field, asdict

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info import Statevector, Operator, DensityMatrix, partial_trace, entropy
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error, phase_damping_error, pauli_error

# ============================================================================
# Constants and Enums
# ============================================================================

class GateType(str, Enum):
    """Supported quantum gate types"""
    # Single-qubit gates
    H = "h"
    X = "x"
    Y = "y"
    Z = "z"
    S = "s"
    SDG = "sdg"
    T = "t"
    TDG = "tdg"
    ID = "id"
    SX = "sx"
    RX = "rx"
    RY = "ry"
    RZ = "rz"
    P = "p"
    U = "u"
    
    # Multi-qubit gates
    CX = "cx"
    CY = "cy"
    CZ = "cz"
    SWAP = "swap"
    ISWAP = "iswap"
    CCX = "ccx"
    CRX = "crx"
    RXX = "rxx"
    RYY = "ryy"
    RZZ = "rzz"
    
    # Other operations
    MEASURE = "measure"
    RESET = "reset"
    BARRIER = "barrier"


class NoiseModelType(str, Enum):
    """Supported noise models"""
    IDEAL = "ideal"
    DEPOLARIZING = "depolarizing"
    AMPLITUDE_DAMPING = "amplitude_damping"
    PHASE_DAMPING = "phase_damping"
    BIT_FLIP = "bit_flip"
    PHASE_FLIP = "phase_flip"


@dataclass
class Gate:
    """Quantum gate data structure"""
    name: str
    label: str
    qubit: int
    position: int
    control: Optional[int] = None
    control2: Optional[int] = None
    param: Optional[float] = None
    param2: Optional[float] = None
    param3: Optional[float] = None
    description: str = ""
    id: str = field(default_factory=lambda: str(time.time()))


# Gate definitions
GATE_DEFINITIONS = {
    'single_qubit': [
        {'name': 'id', 'label': 'I', 'description': 'Identity (No-op)'},
        {'name': 'h', 'label': 'H', 'description': 'Hadamard gate'},
        {'name': 'x', 'label': 'X', 'description': 'Pauli-X gate'},
        {'name': 'y', 'label': 'Y', 'description': 'Pauli-Y gate'},
        {'name': 'z', 'label': 'Z', 'description': 'Pauli-Z gate'},
        {'name': 's', 'label': 'S', 'description': 'Phase gate'},
        {'name': 't', 'label': 'T', 'description': 'T gate'},
        {'name': 'sx', 'label': '√X', 'description': 'Square Root of X'},
        {'name': 'p', 'label': 'P', 'description': 'Phase rotation'},
        {'name': 'rx', 'label': 'Rx', 'description': 'X rotation'},
        {'name': 'ry', 'label': 'Ry', 'description': 'Y rotation'},
        {'name': 'rz', 'label': 'Rz', 'description': 'Z rotation'},
        {'name': 'u', 'label': 'U3', 'description': 'Universal 3-param'},
    ],
    'multi_qubit': [
        {'name': 'cx', 'label': 'CNOT', 'description': 'Controlled-NOT'},
        {'name': 'cy', 'label': 'CY', 'description': 'Controlled-Y'},
        {'name': 'cz', 'label': 'CZ', 'description': 'Controlled-Z'},
        {'name': 'swap', 'label': 'SWAP', 'description': 'Swap qubits'},
        {'name': 'iswap', 'label': 'iSWAP', 'description': 'Imaginary Swap'},
        {'name': 'crx', 'label': 'CRX', 'description': 'Controlled Rx'},
        {'name': 'ccx', 'label': 'CCX', 'description': 'Toffoli (CCNOT)'},
        {'name': 'rxx', 'label': 'XX', 'description': 'Ising XX'},
        {'name': 'ryy', 'label': 'YY', 'description': 'Ising YY'},
        {'name': 'rzz', 'label': 'ZZ', 'description': 'Ising ZZ'},
    ],
    'other': [
        {'name': 'measure', 'label': 'M', 'description': 'Measurement'},
        {'name': 'reset', 'label': 'R', 'description': 'Reset'},
        {'name': 'barrier', 'label': '||', 'description': 'Barrier'},
    ]
}

# ============================================================================
# Quantum Simulation Engine
# ============================================================================

class QuantumEngine:
    """Quantum circuit simulation engine"""
    
    @staticmethod
    def build_circuit(num_qubits: int, gates: List[Gate]) -> QuantumCircuit:
        """Build Qiskit circuit from gate list"""
        qr = QuantumRegister(num_qubits, 'q')
        cr = ClassicalRegister(num_qubits, 'c')
        qc = QuantumCircuit(qr, cr)
        
        sorted_gates = sorted(gates, key=lambda g: g.position)
        
        for gate in sorted_gates:
            try:
                QuantumEngine._apply_gate(qc, gate, num_qubits)
            except Exception as e:
                st.warning(f"Could not apply gate {gate.name}: {e}")
                continue
        
        return qc
    
    @staticmethod
    def _apply_gate(qc: QuantumCircuit, gate: Gate, num_qubits: int):
        """Apply a single gate to circuit"""
        name = gate.name.lower()
        q = gate.qubit
        
        # Single Qubit
        if name == 'id': qc.id(q)
        elif name == 'h': qc.h(q)
        elif name == 'x': qc.x(q)
        elif name == 'y': qc.y(q)
        elif name == 'z': qc.z(q)
        elif name == 's': qc.s(q)
        elif name == 'sdg': qc.sdg(q)
        elif name == 't': qc.t(q)
        elif name == 'tdg': qc.tdg(q)
        elif name == 'sx': qc.sx(q)
        
        # Single Qubit Parametrized
        elif name == 'p' and gate.param is not None: qc.p(gate.param, q)
        elif name == 'rx' and gate.param is not None: qc.rx(gate.param, q)
        elif name == 'ry' and gate.param is not None: qc.ry(gate.param, q)
        elif name == 'rz' and gate.param is not None: qc.rz(gate.param, q)
        elif name == 'u':
            p1 = gate.param if gate.param is not None else 0.0
            p2 = gate.param2 if gate.param2 is not None else 0.0
            p3 = gate.param3 if gate.param3 is not None else 0.0
            qc.u(p1, p2, p3, q)
            
        # Two Qubit
        elif name == 'cx' and gate.control is not None and gate.control != q:
            qc.cx(gate.control, q)
        elif name == 'cy' and gate.control is not None and gate.control != q:
            qc.cy(gate.control, q)
        elif name == 'cz' and gate.control is not None and gate.control != q:
            qc.cz(gate.control, q)
        elif name == 'swap' and gate.control is not None and gate.control != q:
            qc.swap(gate.control, q)
        elif name == 'iswap' and gate.control is not None and gate.control != q:
            qc.iswap(gate.control, q)
            
        # Two Qubit Parametrized
        elif name == 'crx' and gate.control is not None and gate.param is not None and gate.control != q:
            qc.crx(gate.param, gate.control, q)
        elif name == 'rxx' and gate.control is not None and gate.param is not None and gate.control != q:
            qc.rxx(gate.param, gate.control, q)
        elif name == 'ryy' and gate.control is not None and gate.param is not None and gate.control != q:
            qc.ryy(gate.param, gate.control, q)
        elif name == 'rzz' and gate.control is not None and gate.param is not None and gate.control != q:
            qc.rzz(gate.param, gate.control, q)
            
        # Three Qubit
        elif name == 'ccx' and gate.control is not None and gate.control2 is not None:
            if len({gate.control, gate.control2, q}) == 3:
                qc.ccx(gate.control, gate.control2, q)
                
        # Other
        elif name == 'measure':
            qc.measure(q, q)
        elif name == 'reset':
            qc.reset(q)
        elif name == 'barrier':
            qc.barrier(q)
    
    @staticmethod
    def simulate_statevector(qc: QuantumCircuit) -> np.ndarray:
        """Run statevector simulation"""
        qc_sim = qc.copy()
        if 'measure' in qc_sim.count_ops():
            try:
                qc_sim.remove_final_measurements(inplace=True)
            except AttributeError:
                qc_sim.data = [inst for inst in qc_sim.data if inst.operation.name != 'measure']
        
        simulator = AerSimulator(method='statevector')
        qc_sim.save_statevector()
        result = simulator.run(qc_sim).result()
        return result.get_statevector()
    
    @staticmethod
    def simulate_with_noise(qc: QuantumCircuit, noise_type: str, shots: int) -> Dict:
        """Run noisy simulation"""
        simulator = AerSimulator()
        
        try:
            if noise_type != 'ideal':
                noise_model = NoiseModel()
                basis_1q = ['id', 'h', 'x', 'y', 'z', 's', 't', 'sx', 'rx', 'ry', 'rz', 'p', 'u']
                basis_2q = ['cx', 'cy', 'cz', 'swap', 'iswap', 'crx', 'rxx', 'ryy', 'rzz']
                
                if noise_type == 'depolarizing':
                    noise_model.add_all_qubit_quantum_error(depolarizing_error(0.01, 1), basis_1q)
                    noise_model.add_all_qubit_quantum_error(depolarizing_error(0.05, 2), basis_2q)
                elif noise_type == 'amplitude_damping':
                    noise_model.add_all_qubit_quantum_error(amplitude_damping_error(0.05), basis_1q)
                elif noise_type == 'phase_damping':
                    noise_model.add_all_qubit_quantum_error(phase_damping_error(0.05), basis_1q)
                elif noise_type == 'bit_flip':
                    noise_model.add_all_qubit_quantum_error(pauli_error([('X', 0.05), ('I', 0.95)]), basis_1q)
                elif noise_type == 'phase_flip':
                    noise_model.add_all_qubit_quantum_error(pauli_error([('Z', 0.05), ('I', 0.95)]), basis_1q)
                    
                simulator.set_options(noise_model=noise_model)
        except Exception as e:
            st.warning(f"Failed to configure noise model: {e}")
        
        qc_sim = qc.copy()
        if 'measure' not in qc_sim.count_ops():
            qc_sim.measure_all()
        
        result = simulator.run(qc_sim, shots=shots).result()
        return {'counts': result.get_counts()}
    
    @staticmethod
    def calculate_probabilities(statevector: np.ndarray, num_qubits: int) -> Dict[str, float]:
        """Calculate probability distribution"""
        probs = np.abs(statevector) ** 2
        return {
            format(i, f'0{num_qubits}b'): float(p)
            for i, p in enumerate(probs) if p > 1e-10
        }
    
    @staticmethod
    def calculate_density_matrix(statevector: np.ndarray) -> Optional[np.ndarray]:
        """Calculate density matrix (limit up to 6 qubits)"""
        num_qubits = int(np.log2(len(statevector)))
        if num_qubits <= 6:
            return DensityMatrix(statevector).data
        return None

    @staticmethod
    def generate_bloch_data(statevector: np.ndarray, num_qubits: int) -> List[Dict]:
        """Generate Bloch sphere data"""
        bloch_data = []
        sv = Statevector(statevector)
        
        for qubit_idx in range(min(num_qubits, 3)):
            if num_qubits == 1:
                rho = sv.to_operator().data
                x = 2 * np.real(rho[0, 1])
                y = 2 * np.imag(rho[0, 1])
                z = np.real(rho[0, 0] - rho[1, 1])
            else:
                x = np.real(sv.expectation_value(Operator.from_label('X'), qargs=[qubit_idx]))
                y = np.real(sv.expectation_value(Operator.from_label('Y'), qargs=[qubit_idx]))
                z = np.real(sv.expectation_value(Operator.from_label('Z'), qargs=[qubit_idx]))
            
            bloch_data.append({
                'qubit': qubit_idx,
                'x': float(x),
                'y': float(y),
                'z': float(z),
                'purity': float(min(1.0, x**2 + y**2 + z**2))
            })
        
        return bloch_data

    @staticmethod
    def generate_timeline(qc: QuantumCircuit) -> List[Dict]:
        """Generate statevector at each step of the circuit"""
        timeline = []
        qc_sim = qc.copy()
        if 'measure' in qc_sim.count_ops():
            try:
                qc_sim.remove_final_measurements(inplace=True)
            except AttributeError:
                qc_sim.data = [inst for inst in qc_sim.data if inst.operation.name != 'measure']

        current_qc = qc_sim.copy_empty_like()
        try:
            timeline.append({'step': 0, 'gate': 'Initial', 'statevector': Statevector.from_int(0, 2**qc_sim.num_qubits).data})
        except Exception: pass

        for i, inst in enumerate(qc_sim.data):
            current_qc.append(inst)
            try:
                sv = QuantumEngine.simulate_statevector(current_qc)
                timeline.append({'step': i+1, 'gate': inst.operation.name.upper(), 'statevector': sv})
            except Exception: pass
        return timeline

    @staticmethod
    def calculate_entropies(statevector: np.ndarray) -> List[float]:
        """Calculate single qubit Von Neumann entropies"""
        try:
            n = int(np.log2(len(statevector)))
            sv = Statevector(statevector)
            return [float(entropy(partial_trace(sv, [j for j in range(n) if j != i]))) for i in range(n)]
        except Exception:
            return []


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_bloch_sphere(bloch_data: List[Dict]) -> go.Figure:
    """Create Bloch sphere visualization"""
    if not bloch_data:
        fig = go.Figure()
        fig.add_annotation(text="Add gates to see Bloch sphere", 
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    phi = np.linspace(0, 2*np.pi, 50)
    theta = np.linspace(0, np.pi, 25)
    phi, theta = np.meshgrid(phi, theta)
    
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    fig = go.Figure()
    
    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        colorscale='Blues',
        opacity=0.1,
        showscale=False,
        hoverinfo='skip'
    ))
    
    axis_range = 1.2
    fig.add_trace(go.Scatter3d(
        x=[-axis_range, axis_range], y=[0, 0], z=[0, 0],
        mode='lines', line=dict(color='red', width=3),
        name='X', hoverinfo='name'
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[-axis_range, axis_range], z=[0, 0],
        mode='lines', line=dict(color='green', width=3),
        name='Y', hoverinfo='name'
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[-axis_range, axis_range],
        mode='lines', line=dict(color='blue', width=3),
        name='Z', hoverinfo='name'
    ))
    
    colors = ['purple', 'orange', 'cyan']
    for i, data in enumerate(bloch_data):
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter3d(
            x=[0, data['x']],
            y=[0, data['y']],
            z=[0, data['z']],
            mode='markers+lines',
            name=f'Qubit {data["qubit"]}',
            marker=dict(size=6, color=color),
            line=dict(color=color, width=5)
        ))
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X', range=[-axis_range, axis_range], showbackground=False),
            yaxis=dict(title='Y', range=[-axis_range, axis_range], showbackground=False),
            zaxis=dict(title='Z', range=[-axis_range, axis_range], showbackground=False),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def plot_statevector(probabilities: Dict[str, float]) -> go.Figure:
    """Create statevector probability chart"""
    if not probabilities:
        fig = go.Figure()
        fig.add_annotation(text="Run simulation to see statevector",
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    labels = list(probabilities.keys())
    values = list(probabilities.values())
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=values,
            marker_color=['#8b5cf6' if v > 0.5 else '#06b6d4' for v in values],
            text=[f'{v:.3f}' for v in values],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='Statevector Probabilities',
        xaxis_title='Basis State',
        yaxis_title='Probability',
        yaxis=dict(range=[0, 1]),
        height=300,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def plot_measurements(counts: Dict[str, int]) -> go.Figure:
    """Create measurement results pie chart"""
    if not counts:
        fig = go.Figure()
        fig.add_annotation(text="Run simulation to see measurements",
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    labels = list(counts.keys())
    values = list(counts.values())
    total = sum(values)
    
    fig = go.Figure(data=[
        go.Pie(
            labels=labels,
            values=values,
            hole=0.3,
            marker=dict(colors=['#8b5cf6', '#06b6d4', '#22c55e', '#f59e0b', '#ef4444']),
            textinfo='label+percent',
            hoverinfo='label+value+percent'
        )
    ])
    
    fig.update_layout(
        title=f'Measurement Results ({total} shots)',
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def plot_complex_amplitudes(statevector: np.ndarray, num_qubits: int) -> go.Figure:
    """Scatter of amplitudes in the complex plane (Re vs Im)"""
    amps = np.asarray(statevector, dtype=complex).flatten()
    labels = [format(i, f"0{num_qubits}b") for i in range(len(amps))]
    mag = np.abs(amps)
    
    mask = mag > 1e-10
    
    fig = go.Figure()
    
    max_val = max(np.max(mag) * 1.1, 0.1) if len(mag[mask]) > 0 else 1.0
    fig.add_shape(type="line", x0=-max_val, x1=max_val, y0=0, y1=0, line=dict(color="gray", width=1, dash="dash"))
    fig.add_shape(type="line", x0=0, x1=0, y0=-max_val, y1=max_val, line=dict(color="gray", width=1, dash="dash"))
    
    if any(mask):
        fig.add_trace(go.Scatter(
            x=np.real(amps[mask]),
            y=np.imag(amps[mask]),
            mode='markers+text',
            marker=dict(
                size=mag[mask] * 50,
                color=mag[mask],
                colorscale='Plasma',
                showscale=True,
                colorbar=dict(title="|Amp|")
            ),
            text=np.array(labels)[mask],
            textposition="top center",
            hovertemplate="State: %{text}<br>Re: %{x:.4f}<br>Im: %{y:.4f}<br>|Amp|: %{marker.color:.4f}<extra></extra>"
        ))

    fig.update_layout(
        title='Complex Amplitudes',
        xaxis_title='Real',
        yaxis_title='Imaginary',
        xaxis=dict(range=[-max_val, max_val], showgrid=False),
        yaxis=dict(range=[-max_val, max_val], showgrid=False, scaleanchor="x", scaleratio=1),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def plot_phases(statevector: np.ndarray, num_qubits: int) -> go.Figure:
    """Plot phase angles of states with non-zero probability"""
    amps = np.asarray(statevector, dtype=complex).flatten()
    labels = [format(i, f"0{num_qubits}b") for i in range(len(amps))]
    mag = np.abs(amps)
    phases = np.angle(amps)
    
    mask = mag > 1e-10
    
    fig = go.Figure()
    
    if any(mask):
        fig.add_trace(go.Barpolar(
            r=mag[mask]**2,
            theta=np.degrees(phases[mask]),
            text=np.array(labels)[mask],
            marker_color=np.degrees(phases[mask]),
            marker_colorscale='hsv',
            marker_line_color="black",
            marker_line_width=1,
            opacity=0.8,
            hovertemplate="State: %{text}<br>Phase: %{theta:.1f}°<br>Prob: %{r:.4f}<extra></extra>"
        ))
        
    fig.update_layout(
        title='State Phases (Polar)',
        polar=dict(
            radialaxis=dict(title='Probability', range=[0, 1.05], angle=45),
            angularaxis=dict(direction="counterclockwise")
        ),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def plot_density_matrix_heatmap(density: np.ndarray) -> go.Figure:
    """Create density matrix absolute value heatmap"""
    mag = np.abs(density)
    
    hover_text = []
    for i in range(mag.shape[0]):
        row_text = []
        for j in range(mag.shape[1]):
            row_text.append(
                f"|ρ|: {mag[i,j]:.4f}<br>Re: {np.real(density[i,j]):.4f}<br>Im: {np.imag(density[i,j]):.4f}"
            )
        hover_text.append(row_text)
        
    fig = go.Figure(data=go.Heatmap(
        z=mag,
        colorscale='Viridis',
        hoverinfo='text',
        text=hover_text
    ))
    
    fig.update_layout(
        title='Density Matrix Magnitude |ρ|',
        xaxis_title='Column Index',
        yaxis_title='Row Index',
        height=400,
        yaxis=dict(autorange='reversed'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def render_gate_reference():
    """Renders the static gate reference guide."""
    st.subheader("Complete Quantum Gate Reference (With Equations)")

    with st.expander("🔷 1. SINGLE-QUBIT PAULI GATES", expanded=True):
        st.markdown("##### 🟦 I (Identity Gate)")
        st.latex(r"I = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}")
        st.markdown("**Action**: $I|\\psi\\rangle = |\\psi\\rangle$")
        st.caption("👉 No change to state\n👉 Used for padding / alignment")

        st.markdown("---")
        st.markdown("##### 🟦 X (Pauli-X / NOT Gate)")
        st.latex(r"X = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}")
        st.markdown(r"**Action**: $X|0\rangle = |1\rangle, \quad X|1\rangle = |0\rangle$")
        st.caption("👉 Bit flip operation")

        st.markdown("---")
        st.markdown("##### 🟦 Y (Pauli-Y Gate)")
        st.latex(r"Y = \begin{bmatrix} 0 & -i \\ i & 0 \end{bmatrix}")
        st.markdown(r"**Action**: $Y|0\rangle = i|1\rangle, \quad Y|1\rangle = -i|0\rangle$")
        st.caption("👉 Bit flip + phase rotation (imaginary plane rotation)")

        st.markdown("---")
        st.markdown("##### 🟦 Z (Pauli-Z Gate)")
        st.latex(r"Z = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}")
        st.markdown(r"**Action**: $Z|0\rangle = |0\rangle, \quad Z|1\rangle = -|1\rangle$")
        st.caption("👉 Phase flip gate (interference control)")

    with st.expander("🔷 2. HADAMARD GATE (SUPERPOSITION)"):
        st.markdown("##### 🟦 H Gate")
        st.latex(r"H = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}")
        st.markdown(r"**Action**: $H|0\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}}, \quad H|1\rangle = \frac{|0\rangle - |1\rangle}{\sqrt{2}}$")
        st.caption("👉 Creates superposition\n👉 Enables quantum parallelism + interference")

    with st.expander("🔷 3. PHASE GATES"):
        st.markdown("##### 🟦 S Gate (Phase Gate)")
        st.latex(r"S = \begin{bmatrix} 1 & 0 \\ 0 & i \end{bmatrix}")
        st.markdown(r"**Action**: $S|1\rangle = i|1\rangle$")

        st.markdown("---")
        st.markdown("##### 🟦 T Gate (π/8 Gate)")
        st.latex(r"T = \begin{bmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{bmatrix}")
        
        st.markdown("---")
        st.markdown("##### 🟦 General Phase Gate P(λ)")
        st.latex(r"P(\lambda) = \begin{bmatrix} 1 & 0 \\ 0 & e^{i\lambda} \end{bmatrix}")
        st.caption("👉 Controls interference directly via phase")

    with st.expander("🔷 4. ROTATION GATES (VERY IMPORTANT)"):
        st.markdown("##### 🟦 RX(θ)")
        st.latex(r"R_X(\theta) = e^{-i\theta X/2} = \begin{bmatrix} \cos(\theta/2) & -i\sin(\theta/2) \\ -i\sin(\theta/2) & \cos(\theta/2) \end{bmatrix}")

        st.markdown("---")
        st.markdown("##### 🟦 RY(θ)")
        st.latex(r"R_Y(\theta) = e^{-i\theta Y/2} = \begin{bmatrix} \cos(\theta/2) & -\sin(\theta/2) \\ \sin(\theta/2) & \cos(\theta/2) \end{bmatrix}")

        st.markdown("---")
        st.markdown("##### 🟦 RZ(θ)")
        st.latex(r"R_Z(\theta) = e^{-i\theta Z/2} = \begin{bmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{bmatrix}")
        st.caption("👉 RZ controls phase, RX/RY control probability amplitudes")

    with st.expander("🔷 5. SQUARE ROOT GATES"):
        st.markdown("##### 🟦 SX (√X)")
        st.markdown(r"$SX = \sqrt{X}, \quad SX \cdot SX = X$")
        st.latex(r"SX = \frac{1}{2}\begin{bmatrix} 1+i & 1-i \\ 1-i & 1+i \end{bmatrix}")
        st.caption("👉 Hardware-native gate (IBM standard basis)")

    with st.expander("🔷 6. UNIVERSAL GATES"):
        st.markdown("##### 🟦 U3 Gate (MOST IMPORTANT)")
        st.latex(r"U_3(\theta, \phi, \lambda) = \begin{bmatrix} \cos(\theta/2) & -e^{i\lambda}\sin(\theta/2) \\ e^{i\phi}\sin(\theta/2) & e^{i(\phi+\lambda)}\cos(\theta/2) \end{bmatrix}")
        st.caption("👉 Can represent ANY single-qubit operation")

    with st.expander("🔷 7. TWO-QUBIT ENTANGLEMENT GATES"):
        st.markdown("##### 🟦 CNOT (CX)")
        st.markdown(r"$CX|c,t\rangle = |c, t \oplus c\rangle$")
        st.latex(r"CX = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{bmatrix}")
        st.caption("👉 Creates entanglement")

        st.markdown("---")
        st.markdown("##### 🟦 CZ Gate")
        st.markdown(r"$CZ|11\rangle = -|11\rangle$")
        st.latex(r"CZ = \text{diag}(1, 1, 1, -1)")

    with st.expander("🔷 8. CONTROLLED ROTATIONS"):
        st.markdown("##### 🟦 CRX(θ)")
        st.latex(r"CRX(\theta) = |0\rangle\langle0| \otimes I + |1\rangle\langle1| \otimes R_X(\theta)")
        st.caption("👉 Conditional rotation")

    with st.expander("🔷 9. TOFFOLI / MULTI-CONTROL"):
        st.markdown("##### 🟦 CCX (Toffoli)")
        st.markdown(r"$|a,b,c\rangle \rightarrow |a,b, c \oplus (a \cdot b)\rangle$")
        st.caption("👉 Quantum AND gate")

    with st.expander("🔷 10. SWAP FAMILY"):
        st.markdown("##### 🟦 SWAP")
        st.markdown(r"$SWAP|a,b\rangle = |b,a\rangle$")
        st.latex(r"SWAP = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}")

        st.markdown("---")
        st.markdown("##### 🟦 iSWAP")
        st.markdown(r"$|01\rangle \rightarrow i|10\rangle$")

    with st.expander("🔷 11. PHYSICS INTERACTION GATES"):
        st.markdown(r"**XX(θ)**: $e^{-i\theta X \otimes X / 2}$")
        st.markdown(r"**YY(θ)**: $e^{-i\theta Y \otimes Y / 2}$")
        st.markdown(r"**ZZ(θ)**: $e^{-i\theta Z \otimes Z / 2}$")
        st.caption("👉 Used in QAOA + quantum chemistry")

    with st.expander("🔷 12. MEASUREMENT"):
        st.markdown(r"$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$")
        st.markdown("**Measurement probabilities:**")
        st.markdown(r"$P(0) = |\alpha|^2, \quad P(1) = |\beta|^2$")
        st.caption("👉 Collapses quantum state")

    with st.expander("🔷 13. NOISE CHANNELS (REALISTIC SIMULATION)"):
        st.markdown(r"**Bit Flip**: $\rho \rightarrow (1-p)\rho + p X\rho X$")
        st.markdown(r"**Phase Flip**: $\rho \rightarrow (1-p)\rho + p Z\rho Z$")
        st.markdown(r"**Depolarizing**: $\rho \rightarrow (1-p)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z)$")
        st.markdown(r"**Amplitude Damping**: $|1\rangle \rightarrow |0\rangle$ with probability $\gamma$")


# ============================================================================
# Streamlit UI
# ============================================================================

def render_circuit_visualization(gates: List[Gate], num_qubits: int):
    """Render circuit diagram using Qiskit matplotlib drawer"""
    if not gates:
        st.info("👉 Add gates from the sidebar to build your circuit")
        return
    
    st.subheader("Circuit Diagram")
    
    try:
        from qiskit.visualization import circuit_drawer
        
        qc = QuantumEngine.build_circuit(num_qubits, gates)
        
        fig = circuit_drawer(
            qc,
            output='mpl',
            style={'backgroundcolor': '#FFFFFF'}
        )
        
        fig.patch.set_facecolor('white')
        st.pyplot(fig, clear_figure=True, bbox_inches='tight')
        
    except Exception as e:
        st.warning(f"Graphical rendering failed: {e}. Falling back to text.")
        qc = QuantumEngine.build_circuit(num_qubits, gates)
        st.code(str(qc.draw(output='text')), language='text')


def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Quantum Circuit Designer",
        page_icon="⚛️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Base Typography */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Modern Dark Theme Backgrounds with Subtle Gradient */
        .stApp { 
            background: radial-gradient(circle at top right, #1e293b 0%, #0f172a 40%, #020617 100%); 
        }
        
        /* Glassmorphism Sidebar */
        [data-testid="stSidebar"] { 
            background-color: rgba(15, 23, 42, 0.6) !important; 
            backdrop-filter: blur(12px);
            border-right: 1px solid rgba(255, 255, 255, 0.05); 
        }

        /* Gradient Typography for Main Headers */
        h1 { 
            background: linear-gradient(to right, #38bdf8, #8b5cf6); 
            -webkit-background-clip: text; 
            -webkit-text-fill-color: transparent; 
            font-weight: 700 !important;
            letter-spacing: -0.5px;
        }
        h2, h3 { color: #f8fafc !important; font-weight: 600 !important; }
        
        /* Elegant Alerts */
        .stAlert { 
            background-color: rgba(30, 41, 59, 0.5); 
            backdrop-filter: blur(5px);
            border: 1px solid rgba(56, 189, 248, 0.3); 
            color: #e2e8f0; 
            border-radius: 8px;
        }
        
        /* Premium Metrics Cards with Hover Lift */
        div[data-testid="stMetric"] { 
            background: linear-gradient(145deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.7)); 
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.05); 
            padding: 1.2rem; 
            border-radius: 12px; 
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-3px);
            border-color: rgba(56, 189, 248, 0.5);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
        }
        div[data-testid="stMetricValue"] { 
            font-size: 2rem !important; 
            font-weight: 700; 
            color: #38bdf8; 
        }
        
        /* Interactive Buttons */
        .stButton > button {
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            background: linear-gradient(to bottom, #1e293b, #0f172a);
            color: #e2e8f0;
            font-weight: 500;
            transition: all 0.2s ease-in-out;
        }
        .stButton > button:hover {
            border-color: #38bdf8;
            color: #38bdf8;
            transform: scale(1.02);
            box-shadow: 0 0 15px rgba(56, 189, 248, 0.2);
        }
        .stButton > button:active {
            transform: scale(0.98);
        }

        /* Primary Button (Run Simulation) Glow */
        .stButton > button[kind="primary"] {
            background: linear-gradient(to right, #0ea5e9, #6366f1);
            border: none;
            color: white;
            font-weight: 600;
        }
        .stButton > button[kind="primary"]:hover {
            box-shadow: 0 0 20px rgba(99, 102, 241, 0.5);
            color: white;
        }
        
        /* Modern Tabs Styling */
        button[data-baseweb="tab"] {
            background-color: transparent !important;
            border-radius: 6px 6px 0 0 !important;
            padding: 0.5rem 1rem !important;
            margin-right: 0.2rem;
            color: #94a3b8 !important;
            border-bottom: 2px solid transparent !important;
            transition: all 0.2s;
        }
        button[data-baseweb="tab"]:hover {
            color: #e2e8f0 !important;
            background-color: rgba(255, 255, 255, 0.05) !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #38bdf8 !important;
            border-bottom: 2px solid #38bdf8 !important;
            background-color: rgba(56, 189, 248, 0.1) !important;
        }
        
        /* Expanders & Code */
        div[data-testid="stExpander"] { 
            background-color: rgba(30, 41, 59, 0.4); 
            border: 1px solid rgba(255, 255, 255, 0.05); 
            border-radius: 8px; 
            backdrop-filter: blur(4px);
        }
        .stDataFrame { 
            border-radius: 8px; 
            overflow: hidden; 
            border: 1px solid rgba(255, 255, 255, 0.05); 
        }
        code { 
            color: #38bdf8 !important; 
            background-color: rgba(15, 23, 42, 0.6) !important;
            border-radius: 4px;
            padding: 0.2em 0.4em;
        }

        /* Input Fields */
        .stTextInput > div > div > input, 
        .stNumberInput > div > div > input {
            background-color: rgba(15, 23, 42, 0.6);
            color: white;
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 6px;
        }
        .stTextInput > div > div > input:focus, 
        .stNumberInput > div > div > input:focus {
            border-color: #38bdf8;
            box-shadow: 0 0 0 1px #38bdf8;
        }
        
        /* Subheader aesthetic */
        h3 {
            margin-top: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("⚛️ Quantum Circuit Designer")
    st.markdown("Interactive Quantum Circuit Visualizer")
    
    if 'gates' not in st.session_state:
        st.session_state.gates = []
    if 'num_qubits' not in st.session_state:
        st.session_state.num_qubits = 2
    if 'saved_blocks' not in st.session_state:
        st.session_state.saved_blocks = {}
    if 'simulation_result' not in st.session_state:
        st.session_state.simulation_result = None
    
    with st.sidebar:
        st.header("🔧 Quantum Gates")
        
        st.subheader("Single-Qubit Gates")
        for gate in GATE_DEFINITIONS['single_qubit']:
            if st.button(f"{gate['label']} - {gate['description']}", key=f"gate_{gate['name']}"):
                new_gate = Gate(
                    name=gate['name'], label=gate['label'], qubit=0, position=len(st.session_state.gates),
                    param=3.14159 if gate['name'] in ['rx', 'ry', 'rz', 'p', 'u'] else None,
                    param2=0.0 if gate['name'] == 'u' else None,
                    param3=0.0 if gate['name'] == 'u' else None, description=gate['description']
                )
                st.session_state.gates.append(new_gate)
                st.rerun()
        
        st.subheader("Multi-Qubit Gates")
        for gate in GATE_DEFINITIONS['multi_qubit']:
            if st.button(f"{gate['label']} - {gate['description']}", key=f"gate_{gate['name']}"):
                new_gate = Gate(
                    name=gate['name'], label=gate['label'], qubit=1, control=0,
                    control2=2 if gate['name'] == 'ccx' else None, position=len(st.session_state.gates),
                    param=3.14159 if gate['name'] in ['crx', 'rxx', 'ryy', 'rzz'] else None, description=gate['description']
                )
                st.session_state.gates.append(new_gate)
                st.rerun()
        
        st.subheader("Other Operations")
        for gate in GATE_DEFINITIONS['other']:
            if st.button(f"{gate['label']} - {gate['description']}", key=f"gate_{gate['name']}"):
                new_gate = Gate(
                    name=gate['name'], label=gate['label'], qubit=0,
                    position=len(st.session_state.gates), description=gate['description']
                )
                st.session_state.gates.append(new_gate)
                st.rerun()
        
        st.divider()
        st.header("⚙️ Settings")
        num_qubits = st.slider("Number of Qubits", 1, 10, st.session_state.num_qubits)
        st.session_state.num_qubits = num_qubits
        
        shots = st.slider("Shots", 1, 8192, 1024)
        noise_model = st.selectbox("Noise Model", options=[e.value for e in NoiseModelType], index=0)
        
        st.divider()
        st.header("🎯 Actions")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎮 Run Simulation", use_container_width=True, type="primary"):
                with st.spinner("Running quantum simulation..."):
                    start_time = time.time()
                    
                    qc = QuantumEngine.build_circuit(num_qubits, st.session_state.gates)
                    statevector = QuantumEngine.simulate_statevector(qc)
                    probabilities = QuantumEngine.calculate_probabilities(statevector, num_qubits)
                    bloch_data = QuantumEngine.generate_bloch_data(statevector, num_qubits)
                    density_matrix = QuantumEngine.calculate_density_matrix(statevector)
                    
                    counts = None
                    if noise_model != 'ideal' or shots > 0:
                        counts_result = QuantumEngine.simulate_with_noise(qc, noise_model, shots)
                        counts = counts_result['counts']
                    
                    qasm_str = qc.qasm() if hasattr(qc, 'qasm') else str(qc)
                    timeline = QuantumEngine.generate_timeline(qc)
                    entropies = QuantumEngine.calculate_entropies(statevector)
                    
                    try:
                        qc_opt = transpile(qc, optimization_level=3)
                        depth_opt, size_opt = qc_opt.depth(), qc_opt.size()
                        qasm_opt = qc_opt.qasm() if hasattr(qc_opt, 'qasm') else str(qc_opt)
                    except Exception:
                        depth_opt, size_opt, qasm_opt = qc.depth(), qc.size(), qasm_str

                    elapsed = time.time() - start_time
                    
                    st.session_state.simulation_result = {
                        'statevector': statevector, 'probabilities': probabilities, 'bloch_data': bloch_data,
                        'density_matrix': density_matrix, 'counts': counts, 'qasm': qasm_str, 'time': elapsed,
                        'num_gates': len(st.session_state.gates), 'depth': qc.depth(), 'width': qc.width(),
                        'size': qc.size(), 'timeline': timeline, 'entropies': entropies,
                        'depth_opt': depth_opt, 'size_opt': size_opt, 'qasm_opt': qasm_opt
                    }
                    
                    st.success(f"Simulation completed in {elapsed:.3f}s")
                    st.rerun()
        
        with col2:
            if st.button("🗑️ Clear Circuit", use_container_width=True):
                st.session_state.gates = []
                st.session_state.simulation_result = None
                st.rerun()
        
        st.divider()
        st.subheader("📚 Presets")
        
        if st.button("Create Bell State", use_container_width=True):
            st.session_state.gates = [
                Gate(name='h', label='H', qubit=0, position=0, description='Hadamard'),
                Gate(name='cx', label='CNOT', qubit=1, control=0, position=1, description='CNOT'),
            ]
            st.session_state.num_qubits = 2
            st.session_state.simulation_result = None
            st.rerun()
        
        if st.button("Create GHZ State", use_container_width=True):
            st.session_state.gates = [
                Gate(name='h', label='H', qubit=0, position=0, description='Hadamard'),
                Gate(name='cx', label='CNOT', qubit=1, control=0, position=1, description='CNOT'),
                Gate(name='cx', label='CNOT', qubit=2, control=1, position=2, description='CNOT'),
            ]
            st.session_state.num_qubits = 3
            st.session_state.simulation_result = None
            st.rerun()
        
        st.divider()
        if st.session_state.simulation_result:
            st.download_button(
                label="📥 Export QASM",
                data=st.session_state.simulation_result['qasm'],
                file_name=f"circuit_{int(time.time())}.qasm",
                mime="text/plain",
                use_container_width=True
            )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Circuit Builder")
        if st.session_state.gates:
            st.write(f"**{len(st.session_state.gates)} gates added**")
            st.subheader("Gate Configuration")
            
            edited_gates = []
            for i, gate in enumerate(st.session_state.gates):
                with st.expander(f"Step {i+1}: {gate.label} on Qubit {gate.qubit}", expanded=False):
                    col_a, col_b, col_c = st.columns([1.5, 2.5, 2])
                    with col_a:
                        safe_qubit = min(gate.qubit, num_qubits - 1)
                        new_qubit = st.number_input("Target Q", min_value=0, max_value=num_qubits-1, value=safe_qubit, key=f"qubit_{gate.id}")
                    
                    new_control = getattr(gate, 'control', None)
                    new_control2 = getattr(gate, 'control2', None)
                    new_param = getattr(gate, 'param', None)
                    new_param2 = getattr(gate, 'param2', None)
                    new_param3 = getattr(gate, 'param3', None)
                    
                    with col_b:
                        if gate.name in ['cx', 'cy', 'cz', 'swap', 'iswap', 'ccx', 'crx', 'rxx', 'ryy', 'rzz']:
                            safe_control = min(new_control if new_control is not None else 0, num_qubits - 1)
                            new_control = st.number_input("Ctrl 1", min_value=0, max_value=num_qubits-1, value=safe_control, key=f"ctrl_{gate.id}")
                        if gate.name in ['ccx']:
                            safe_control2 = min(new_control2 if new_control2 is not None else (1 if num_qubits > 1 else 0), num_qubits - 1)
                            new_control2 = st.number_input("Ctrl 2", min_value=0, max_value=num_qubits-1, value=safe_control2, key=f"ctrl2_{gate.id}")
                        if gate.name in ['rx', 'ry', 'rz', 'p', 'u', 'crx', 'rxx', 'ryy', 'rzz']:
                            new_param = st.number_input("Angle θ", value=float(new_param) if new_param is not None else 3.14159, key=f"param_{gate.id}")
                        if gate.name in ['u']:
                            new_param2 = st.number_input("Angle ϕ", value=float(new_param2) if new_param2 is not None else 0.0, key=f"param2_{gate.id}")
                            new_param3 = st.number_input("Angle λ", value=float(new_param3) if new_param3 is not None else 0.0, key=f"param3_{gate.id}")
                            
                    with col_c:
                        st.markdown("<div style='margin-top: 1.75rem;'></div>", unsafe_allow_html=True)
                        btn1, btn2, btn3 = st.columns(3)
                        with btn1:
                            if st.button("⬆️", key=f"up_{gate.id}", disabled=(i == 0), help="Move gate up"):
                                st.session_state.gates[i], st.session_state.gates[i-1] = st.session_state.gates[i-1], st.session_state.gates[i]
                                st.rerun()
                        with btn2:
                            if st.button("⬇️", key=f"dn_{gate.id}", disabled=(i == len(st.session_state.gates)-1), help="Move gate down"):
                                st.session_state.gates[i], st.session_state.gates[i+1] = st.session_state.gates[i+1], st.session_state.gates[i]
                                st.rerun()
                        with btn3:
                            if st.button("🗑️", key=f"remove_{gate.id}", help="Remove gate"):
                                st.session_state.gates.pop(i)
                                st.rerun()
                    
                    edited_gates.append(Gate(
                        name=gate.name, label=gate.label, qubit=new_qubit, position=i,
                        control=new_control, control2=new_control2, param=new_param, param2=new_param2,
                        param3=new_param3, description=gate.description, id=gate.id
                    ))
            
            st.session_state.gates = edited_gates
            render_circuit_visualization(st.session_state.gates, num_qubits)
        else:
            st.info("👉 Add gates from the sidebar to build your circuit")
            
        # Circuit Composer Feature
        st.divider()
        st.subheader("🧱 Circuit Composer (Blocks)")
        st.write("Save your current circuit as a custom block to quickly append and build larger circuits end-to-end.")
        
        c_name, c_btn = st.columns([3, 2])
        with c_name:
            block_name = st.text_input("Block Name", placeholder="e.g., Bell State Prep", label_visibility="collapsed")
        with c_btn:
            if st.button("💾 Save as Block", use_container_width=True):
                if not st.session_state.gates:
                    st.warning("Cannot save an empty circuit.")
                elif not block_name:
                    st.warning("Please provide a block name.")
                else:
                    st.session_state.saved_blocks[block_name] = [Gate(**asdict(g)) for g in st.session_state.gates]
                    st.success(f"Saved '{block_name}'!")
                    
        if st.session_state.saved_blocks:
            st.markdown("#### Block Library")
            for b_name, b_gates in list(st.session_state.saved_blocks.items()):
                with st.container():
                    col_b1, col_b2, col_b3 = st.columns([3, 2, 1])
                    col_b1.markdown(f"**{b_name}** ({len(b_gates)} ops)")
                    if col_b2.button("➕ Append", key=f"app_{b_name}", help="Append block to end of current circuit", use_container_width=True):
                        for g in b_gates:
                            new_g = Gate(**asdict(g))
                            new_g.id = str(time.time() + np.random.random())
                            new_g.position = len(st.session_state.gates)
                            st.session_state.gates.append(new_g)
                        st.rerun()
                    if col_b3.button("🗑️", key=f"del_b_{b_name}", help="Delete block"):
                        del st.session_state.saved_blocks[b_name]
                        st.rerun()
    
    with col2:
        st.header("Analysis & Reference")
        main_tabs = st.tabs(["Visualizations", "Gate Reference"])

        with main_tabs[0]:
            if st.session_state.simulation_result:
                result = st.session_state.simulation_result

                vis_tabs = st.tabs([
                    "1. Bloch", "2. Probs", "3. Amps", "4. Density", "5. Timeline", 
                    "6. Code", "7. Entangle", "8. Interfere", "9. Noise", "10. Struct",
                    "11. Fidelity", "12. Space", "13. Optim", "14. Algo", "15. Measure"
                ])
                
                with vis_tabs[0]:
                    st.subheader("Bloch Sphere")
                    st.plotly_chart(plot_bloch_sphere(result['bloch_data']), use_container_width=True, key="main_bloch_fig")
                    st.caption("Step-wise rotation animation requires timeline feature to be enabled.")
                    
                with vis_tabs[1]:
                    st.subheader("Statevector")
                    st.plotly_chart(plot_statevector(result['probabilities']), use_container_width=True, key="main_sv_fig")
                    if result['counts']:
                        st.subheader("Measurements")
                        st.plotly_chart(plot_measurements(result['counts']), use_container_width=True, key="main_meas_fig")
                    else:
                        st.info("Run simulation with a noise model or shots > 0 to see measurement counts.")
                
                with vis_tabs[2]:
                    st.subheader("Complex Amplitudes")
                    st.plotly_chart(plot_complex_amplitudes(result['statevector'], num_qubits), use_container_width=True, key="main_amp_fig")
                    st.subheader("State Phases")
                    st.plotly_chart(plot_phases(result['statevector'], num_qubits), use_container_width=True, key="main_phase_fig")
                    st.subheader("Amplitude Table")
                    amps = np.asarray(result['statevector']).flatten()
                    df_amps = [
                        { "State": format(i, f"0{num_qubits}b"), "Real": float(np.real(amp)), "Imag": float(np.imag(amp)),
                          "Mag": float(np.abs(amp)), "Phase (rad)": float(np.angle(amp)), "Prob": float(np.abs(amp)**2) }
                        for i, amp in enumerate(amps) if np.abs(amp)**2 > 1e-10
                    ]
                    if df_amps: st.dataframe(df_amps, use_container_width=True)
                
                with vis_tabs[3]:
                    if result.get('density_matrix') is not None:
                        st.subheader("Density Matrix")
                        st.plotly_chart(plot_density_matrix_heatmap(result['density_matrix']), use_container_width=True, key="main_density_fig")
                    elif num_qubits > 6:
                        st.info("Density Matrix visualization disabled for > 6 qubits to conserve memory.")
                    
                    st.subheader("Circuit Metrics")
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    col_m1.metric("Qubits", num_qubits)
                    col_m2.metric("Gates", result['num_gates'])
                    col_m3.metric("Depth", result.get('depth', 'N/A'))
                    col_m4.metric("Size", result.get('size', 'N/A'))
                    
                    col_w1, col_w2, col_w3, col_w4 = st.columns(4)
                    col_w1.metric("Width", result.get('width', 'N/A'))
                    col_w2.metric("Time", f"{result['time']:.3f}s")
                
                with vis_tabs[4]:
                    st.subheader("Timeline Explorer")
                    if result.get('timeline'):
                        max_step = len(result['timeline']) - 1
                        if 'tl_step' not in st.session_state: st.session_state.tl_step = 0
                        if st.session_state.tl_step > max_step: st.session_state.tl_step = max_step
                            
                        col_p1, col_p2, col_p3, col_p4 = st.columns([1, 1, 1, 3])
                        with col_p1:
                            if st.button("⏮️ Prev", use_container_width=True):
                                st.session_state.tl_step = max(0, st.session_state.tl_step - 1)
                                st.session_state.tl_playing = False
                        with col_p2:
                            if st.button("▶️ Play / Pause", use_container_width=True):
                                st.session_state.tl_playing = not getattr(st.session_state, 'tl_playing', False)
                        with col_p3:
                            if st.button("⏭️ Next", use_container_width=True):
                                st.session_state.tl_step = min(max_step, st.session_state.tl_step + 1)
                                st.session_state.tl_playing = False
                                
                        current_step = st.session_state.tl_step
                        selected_step = st.slider("Circuit Step", 0, max_step, value=current_step, key="tl_step_slider")
                        
                        if selected_step != current_step:
                            st.session_state.tl_step = selected_step
                            st.session_state.tl_playing = False
                            st.rerun()
                            
                        step_data = result['timeline'][st.session_state.tl_step]
                        st.write(f"**Gate Applied:** `{step_data['gate']}` (Step {step_data['step']} of {max_step})")
                        
                        col_t1, col_t2 = st.columns(2)
                        with col_t1:
                            step_probs = QuantumEngine.calculate_probabilities(step_data['statevector'], num_qubits)
                            st.plotly_chart(plot_statevector(step_probs), use_container_width=True, key="tl_sv_fig")
                        with col_t2:
                            step_bloch = QuantumEngine.generate_bloch_data(step_data['statevector'], num_qubits)
                            st.plotly_chart(plot_bloch_sphere(step_bloch), use_container_width=True, key="tl_bloch_fig")
                            
                        if getattr(st.session_state, 'tl_playing', False):
                            if st.session_state.tl_step < max_step:
                                time.sleep(0.75)
                                st.session_state.tl_step += 1
                                st.rerun()
                            else:
                                st.session_state.tl_playing = False
                    else:
                        st.info("Timeline data not available.")
                
                with vis_tabs[5]:
                    st.subheader("OpenQASM 2")
                    st.code(result['qasm'], language='qasm')
                    st.subheader("OpenQASM 3")
                    try:
                        from qiskit import qasm3
                        qc_out = QuantumEngine.build_circuit(num_qubits, st.session_state.gates)
                        st.code(qasm3.dumps(qc_out), language='qasm')
                    except Exception as e:
                        st.info(f"OpenQASM 3 not available: {e}")
                
                with vis_tabs[6]:
                    st.subheader("Entanglement Analytics")
                    if result.get('entropies'):
                        fig = go.Figure(data=[go.Bar(
                            x=[f"Q{i}" for i in range(num_qubits)], y=result['entropies'], marker_color='#22c55e',
                            text=[f"{e:.3f}" for e in result['entropies']], textposition='auto'
                        )])
                        fig.update_layout(title="Single-Qubit Von Neumann Entropy", yaxis_title="Entropy (S)", height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig, use_container_width=True, key="entangle_bar_fig")
                        
                        if max(result['entropies']) > 0.1: st.success(f"High entanglement detected! Max S: {max(result['entropies']):.3f}")
                        else: st.info("Low or no entanglement detected.")
                    else: st.info("Metrics not available.")
                    
                with vis_tabs[7]:
                    st.subheader("Quantum Interference")
                    amps = np.asarray(result['statevector']).flatten()
                    indices = np.argsort(np.abs(amps))[-32:] if num_qubits > 6 else np.arange(len(amps))
                    labels = [format(i, f"0{num_qubits}b") for i in indices]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=labels, y=np.real(amps[indices]), name='Real', marker_color='#3b82f6'))
                    fig.add_trace(go.Bar(x=labels, y=np.imag(amps[indices]), name='Imaginary', marker_color='#f59e0b'))
                    fig.update_layout(title="State Amplitudes (Interference)", barmode='group', height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True, key="interfere_bar_fig")
                
                with vis_tabs[8]:
                    st.subheader("Noise & Decoherence")
                    if result.get('density_matrix') is not None:
                        purity = float(np.real(np.trace(np.dot(result['density_matrix'], result['density_matrix']))))
                        col_n1, col_n2 = st.columns(2)
                        col_n1.metric("State Purity Tr(ρ²)", f"{purity:.4f}")
                        if purity > 0.999: col_n2.success("Pure State (Ideal)")
                        else: col_n2.warning("Mixed State (Decohered)")
                    else: st.info("Run with ≤6 qubits to view purity metrics.")
                
                with vis_tabs[9]:
                    st.subheader("Circuit Structure")
                    if st.session_state.gates:
                        gate_counts = [0] * num_qubits
                        for g in st.session_state.gates:
                            gate_counts[g.qubit] += 1
                            if g.control is not None: gate_counts[g.control] += 1
                        fig = go.Figure(data=[go.Bar(x=[f"Q{i}" for i in range(num_qubits)], y=gate_counts, marker_color='#8b5cf6')])
                        fig.update_layout(title="Operations per Qubit", yaxis_title="Gate Count", height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig, use_container_width=True, key="struct_bar_fig")
                    else: st.info("Add gates to analyze structure.")
                
                with vis_tabs[10]:
                    st.subheader("Fidelity Analytics")
                    if noise_model != 'ideal' and result.get('counts'):
                        total_shots = sum(result['counts'].values())
                        theoretical, actual = result['probabilities'], {k: v/total_shots for k, v in result['counts'].items()}
                        fidelity = sum(np.sqrt(theoretical.get(k, 0) * actual.get(k, 0)) for k in set(theoretical) | set(actual)) ** 2
                        
                        col_f1, col_f2 = st.columns(2)
                        col_f1.metric("Bhattacharyya Fidelity", f"{fidelity:.4f}")
                        if fidelity > 0.95: col_f2.success("High fidelity")
                        elif fidelity > 0.7: col_f2.warning("Moderate reduction")
                        else: col_f2.error("Significant noise impact")
                        
                        st.info("The ideal expected distribution comparison:")
                        st.plotly_chart(plot_statevector(result['probabilities']), use_container_width=True, key="fid_ideal_sv_fig")
                    else: st.success("Simulation ran in IDEAL mode. Maximum Theoretical Fidelity achieved.")
                
                with vis_tabs[11]:
                    st.subheader("State Space Explorer")
                    st.write("Top highly probable basis states (Projection Collapse):")
                    for state, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)[:5]:
                        st.progress(prob, text=f"|{state}⟩ : {prob*100:.2f}%")
                
                with vis_tabs[12]:
                    st.subheader("Circuit Optimization")
                    col_o1, col_o2 = st.columns(2)
                    col_o1.metric("Optimized Depth", result.get('depth_opt', 0), delta=result.get('depth_opt', 0) - result.get('depth', 0), delta_color="inverse")
                    col_o2.metric("Optimized Size", result.get('size_opt', 0), delta=result.get('size_opt', 0) - result.get('size', 0), delta_color="inverse")
                    with st.expander("View Optimized OpenQASM"): st.code(result.get('qasm_opt', ''), language='qasm')
                
                with vis_tabs[13]:
                    st.subheader("Algorithm Insight")
                    gates_used = set(g.name for g in st.session_state.gates)
                    if 'h' in gates_used and ('cx' in gates_used or 'cz' in gates_used):
                        st.write("💡 **Analysis:** Superposition and entanglement generation detected.")
                    elif 'h' in gates_used:
                        st.write("💡 **Analysis:** Parallel basis state superposition utilized.")
                    else: st.write("💡 **Analysis:** Purely classical-like Z-basis operations.")
                
                with vis_tabs[14]:
                    st.subheader("Measurement Analytics")
                    if result.get('counts'):
                        total_shots = sum(result['counts'].values())
                        all_keys = sorted(list(set(result['probabilities']) | set(result['counts'])))
                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=all_keys, y=[result['probabilities'].get(k, 0) for k in all_keys], name='Theoretical'))
                        fig.add_trace(go.Bar(x=all_keys, y=[result['counts'].get(k, 0)/total_shots for k in all_keys], name=f'Actual ({total_shots})'))
                        fig.update_layout(barmode='group', title="Theoretical vs Actual Sampling", height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig, use_container_width=True, key="meas_comp_fig")
                    else: st.info("Run with noise model or shots > 0 to see statistical variance.")

            else:
                st.info("Run simulation to see visualizations")

        with main_tabs[1]:
            render_gate_reference()
    
    st.markdown("""
        <div style='text-align: center; color: #94a3b8; padding: 2rem 1rem; margin-top: 3rem; border-top: 1px solid #334155;'>
            <p style='font-size: 1.1em; margin-bottom: 0.5rem;'>Quantum Circuit Designer | Built with Streamlit & Qiskit</p>
            <p style='font-size: 1.05em;'>
                Made By <a href='https://sourishdeyportfolio.vercel.app/' target='_blank' style='color: #38bdf8; text-decoration: none; font-weight: 600; transition: color 0.2s;'>Sourish Dey</a>
            </p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()