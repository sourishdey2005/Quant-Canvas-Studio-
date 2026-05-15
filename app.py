"""
Qircuit Studio - Quantum Circuit Visualizer
Streamlit-based interactive quantum circuit builder and simulator

Author: Qircuit Studio Development Team
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
        
        # Sort gates by position
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
                # Qiskit 1.0+ fallback
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
    
    # Create sphere
    phi = np.linspace(0, 2*np.pi, 50)
    theta = np.linspace(0, np.pi, 25)
    phi, theta = np.meshgrid(phi, theta)
    
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    fig = go.Figure()
    
    # Add sphere surface
    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        colorscale='Blues',
        opacity=0.1,
        showscale=False,
        hoverinfo='skip'
    ))
    
    # Add axes
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
    
    # Add Bloch vectors
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
    percentages = [v/total*100 for v in values]
    
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
    
    # Only show states with non-zero probability
    mask = mag > 1e-10
    
    fig = go.Figure()
    
    # Add origin and axes
    max_val = max(np.max(mag) * 1.1, 0.1) if len(mag[mask]) > 0 else 1.0
    fig.add_shape(type="line", x0=-max_val, x1=max_val, y0=0, y1=0, line=dict(color="gray", width=1, dash="dash"))
    fig.add_shape(type="line", x0=0, x1=0, y0=-max_val, y1=max_val, line=dict(color="gray", width=1, dash="dash"))
    
    if any(mask):
        fig.add_trace(go.Scatter(
            x=np.real(amps[mask]),
            y=np.imag(amps[mask]),
            mode='markers+text',
            marker=dict(
                size=mag[mask] * 50,  # Size by magnitude
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
    
    # Only show states with non-zero probability
    mask = mag > 1e-10
    
    fig = go.Figure()
    
    if any(mask):
        fig.add_trace(go.Barpolar(
            r=mag[mask]**2,  # Radius represents probability
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
        
        # Create the matplotlib figure with a white background for high visibility
        fig = circuit_drawer(
            qc,
            output='mpl',
            style={'backgroundcolor': '#FFFFFF'}
        )
        
        # Force the outer figure background to white
        fig.patch.set_facecolor('white')
        
        st.pyplot(fig, clear_figure=True, bbox_inches='tight')
        
    except Exception as e:
        st.warning(f"Graphical rendering failed: {e}. Falling back to text.")
        qc = QuantumEngine.build_circuit(num_qubits, gates)
        st.code(str(qc.draw(output='text')), language='text')


def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Qircuit Studio",
        page_icon="⚛️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main {
            background-color: #0f172a;
        }
        .stAlert {
            background-color: #1e293b;
            border: 1px solid #475569;
        }
        div[data-testid="stMetricValue"] {
            font-size: 24px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("⚛️ Qircuit Studio")
    st.markdown("Interactive Quantum Circuit Visualizer")
    
    # Initialize session state
    if 'gates' not in st.session_state:
        st.session_state.gates = []
    if 'num_qubits' not in st.session_state:
        st.session_state.num_qubits = 2
    if 'simulation_result' not in st.session_state:
        st.session_state.simulation_result = None
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Quantum Gates")
        
        # Gate selection
        st.subheader("Single-Qubit Gates")
        for gate in GATE_DEFINITIONS['single_qubit']:
            if st.button(f"{gate['label']} - {gate['description']}", key=f"gate_{gate['name']}"):
                new_gate = Gate(
                    name=gate['name'],
                    label=gate['label'],
                    qubit=0,
                    position=len(st.session_state.gates),
                    param=3.14159 if gate['name'] in ['rx', 'ry', 'rz', 'p', 'u'] else None,
                    param2=0.0 if gate['name'] == 'u' else None,
                    param3=0.0 if gate['name'] == 'u' else None,
                    description=gate['description']
                )
                st.session_state.gates.append(new_gate)
                st.rerun()
        
        st.subheader("Multi-Qubit Gates")
        for gate in GATE_DEFINITIONS['multi_qubit']:
            if st.button(f"{gate['label']} - {gate['description']}", key=f"gate_{gate['name']}"):
                new_gate = Gate(
                    name=gate['name'],
                    label=gate['label'],
                    qubit=1,
                    control=0,
                    control2=2 if gate['name'] == 'ccx' else None,
                    position=len(st.session_state.gates),
                    param=3.14159 if gate['name'] in ['crx', 'rxx', 'ryy', 'rzz'] else None,
                    description=gate['description']
                )
                st.session_state.gates.append(new_gate)
                st.rerun()
        
        st.subheader("Other Operations")
        for gate in GATE_DEFINITIONS['other']:
            if st.button(f"{gate['label']} - {gate['description']}", key=f"gate_{gate['name']}"):
                new_gate = Gate(
                    name=gate['name'],
                    label=gate['label'],
                    qubit=0,
                    position=len(st.session_state.gates),
                    description=gate['description']
                )
                st.session_state.gates.append(new_gate)
                st.rerun()
        
        st.divider()
        
        # Circuit settings
        st.header("⚙️ Settings")
        num_qubits = st.slider("Number of Qubits", 1, 10, st.session_state.num_qubits)
        st.session_state.num_qubits = num_qubits
        
        shots = st.slider("Shots", 1, 8192, 1024)
        noise_model = st.selectbox(
            "Noise Model",
            options=[e.value for e in NoiseModelType],
            index=0
        )
        
        st.divider()
        
        # Actions
        st.header("🎯 Actions")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎮 Run Simulation", use_container_width=True, type="primary"):
                with st.spinner("Running quantum simulation..."):
                    start_time = time.time()
                    
                    # Build and simulate circuit
                    qc = QuantumEngine.build_circuit(num_qubits, st.session_state.gates)
                    statevector = QuantumEngine.simulate_statevector(qc)
                    probabilities = QuantumEngine.calculate_probabilities(statevector, num_qubits)
                    bloch_data = QuantumEngine.generate_bloch_data(statevector, num_qubits)
                    
                    # Calculate density matrix
                    density_matrix = QuantumEngine.calculate_density_matrix(statevector)
                    
                    # Run noisy simulation if needed
                    counts = None
                    if noise_model != 'ideal' or shots > 0:
                        counts_result = QuantumEngine.simulate_with_noise(qc, noise_model, shots)
                        counts = counts_result['counts']
                    
                    # Get QASM
                    qasm_str = qc.qasm() if hasattr(qc, 'qasm') else str(qc)
                    
                    # Generate timeline and metrics
                    timeline = QuantumEngine.generate_timeline(qc)
                    entropies = QuantumEngine.calculate_entropies(statevector)
                    
                    # Optimize circuit (Level 3)
                    try:
                        qc_opt = transpile(qc, optimization_level=3)
                        depth_opt, size_opt = qc_opt.depth(), qc_opt.size()
                        qasm_opt = qc_opt.qasm() if hasattr(qc_opt, 'qasm') else str(qc_opt)
                    except Exception:
                        depth_opt, size_opt, qasm_opt = qc.depth(), qc.size(), qasm_str

                    elapsed = time.time() - start_time
                    
                    st.session_state.simulation_result = {
                        'statevector': statevector,
                        'probabilities': probabilities,
                        'bloch_data': bloch_data,
                        'density_matrix': density_matrix,
                        'counts': counts,
                        'qasm': qasm_str,
                        'time': elapsed,
                        'num_gates': len(st.session_state.gates),
                        'depth': qc.depth(),
                        'width': qc.width(),
                        'size': qc.size(),
                        'timeline': timeline,
                        'entropies': entropies,
                        'depth_opt': depth_opt,
                        'size_opt': size_opt,
                        'qasm_opt': qasm_opt
                    }
                    
                    st.success(f"Simulation completed in {elapsed:.3f}s")
                    st.rerun()
        
        with col2:
            if st.button("🗑️ Clear Circuit", use_container_width=True):
                st.session_state.gates = []
                st.session_state.simulation_result = None
                st.rerun()
        
        # Preset circuits
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
        
        # Export
        st.divider()
        if st.session_state.simulation_result:
            st.download_button(
                label="📥 Export QASM",
                data=st.session_state.simulation_result['qasm'],
                file_name=f"circuit_{int(time.time())}.qasm",
                mime="text/plain",
                use_container_width=True
            )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Circuit Builder")
        
        # Display current gates
        if st.session_state.gates:
            st.write(f"**{len(st.session_state.gates)} gates added**")
            
            # Gate editor
            st.subheader("Gate Configuration")
            edited_gates = []
            for i, gate in enumerate(st.session_state.gates):
                with st.expander(f"Gate {i+1}: {gate.label} on qubit {gate.qubit}", expanded=False):
                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a:
                        safe_qubit = min(gate.qubit, num_qubits - 1)
                        new_qubit = st.number_input(
                            "Target Q",
                            min_value=0,
                            max_value=num_qubits-1,
                            value=safe_qubit,
                            key=f"qubit_{i}"
                        )
                    with col_b:
                        new_position = st.number_input(
                            "Position",
                            min_value=0,
                            value=gate.position,
                            key=f"pos_{i}"
                        )
                    
                    new_control = getattr(gate, 'control', None)
                    new_control2 = getattr(gate, 'control2', None)
                    new_param = getattr(gate, 'param', None)
                    new_param2 = getattr(gate, 'param2', None)
                    new_param3 = getattr(gate, 'param3', None)
                    
                    with col_c:
                        if gate.name in ['cx', 'cy', 'cz', 'swap', 'iswap', 'ccx', 'crx', 'rxx', 'ryy', 'rzz']:
                            safe_control = min(new_control if new_control is not None else 0, num_qubits - 1)
                            new_control = st.number_input("Ctrl 1", min_value=0, max_value=num_qubits-1, value=safe_control, key=f"ctrl_{i}")
                        if gate.name in ['ccx']:
                            safe_control2 = min(new_control2 if new_control2 is not None else (1 if num_qubits > 1 else 0), num_qubits - 1)
                            new_control2 = st.number_input("Ctrl 2", min_value=0, max_value=num_qubits-1, value=safe_control2, key=f"ctrl2_{i}")
                            
                        if gate.name in ['rx', 'ry', 'rz', 'p', 'u', 'crx', 'rxx', 'ryy', 'rzz']:
                            new_param = st.number_input("Angle θ", value=float(new_param) if new_param is not None else 3.14159, key=f"param_{i}")
                        if gate.name in ['u']:
                            new_param2 = st.number_input("Angle ϕ", value=float(new_param2) if new_param2 is not None else 0.0, key=f"param2_{i}")
                            new_param3 = st.number_input("Angle λ", value=float(new_param3) if new_param3 is not None else 0.0, key=f"param3_{i}")
                            
                    with col_d:
                        st.markdown("<div style='margin-top: 1.75rem;'></div>", unsafe_allow_html=True)
                        if st.button("🗑️ Remove", key=f"remove_{i}"):
                            st.session_state.gates.pop(i)
                            st.rerun()
                    
                    edited_gate = Gate(
                        name=gate.name,
                        label=gate.label,
                        qubit=new_qubit,
                        position=new_position,
                        control=new_control,
                        control2=new_control2,
                        param=new_param,
                        param2=new_param2,
                        param3=new_param3,
                        description=gate.description,
                        id=gate.id
                    )
                    edited_gates.append(edited_gate)
            
            st.session_state.gates = edited_gates
            
            # Visual representation
            render_circuit_visualization(st.session_state.gates, num_qubits)
        else:
            st.info("👉 Add gates from the sidebar to build your circuit")
    
    with col2:
        st.header("Visualizations")
        
        if st.session_state.simulation_result:
            result = st.session_state.simulation_result

            tabs = st.tabs([
                "1. Bloch", "2. Probs", "3. Amps", "4. Density", "5. Timeline", 
                "6. Code", "7. Entangle", "8. Interfere", "9. Noise", "10. Struct",
                "11. Fidelity", "12. Space", "13. Optim", "14. Algo", "15. Measure"
            ])
            
            with tabs[0]: # 1. BLOCH SPHERE TAB
                st.subheader(" Bloch Sphere")
                bloch_fig = plot_bloch_sphere(result['bloch_data'])
                st.plotly_chart(bloch_fig, use_container_width=True, key="main_bloch_fig")
                st.caption("Step-wise rotation animation requires timeline feature to be enabled.")
                
            with tabs[1]: # 2. PROBABILITIES TAB
                st.subheader(" Statevector")
                sv_fig = plot_statevector(result['probabilities'])
                st.plotly_chart(sv_fig, use_container_width=True, key="main_sv_fig")
                
                if result['counts']:
                    st.subheader(" Measurements")
                    meas_fig = plot_measurements(result['counts'])
                    st.plotly_chart(meas_fig, use_container_width=True, key="main_meas_fig")
                else:
                    st.info("Run simulation with a noise model or shots > 0 to see measurement counts.")
            
            with tabs[2]: # 3. AMPLITUDES TAB
                st.subheader(" Complex Amplitudes")
                amp_fig = plot_complex_amplitudes(result['statevector'], num_qubits)
                st.plotly_chart(amp_fig, use_container_width=True, key="main_amp_fig")
                
                st.subheader(" State Phases")
                phase_fig = plot_phases(result['statevector'], num_qubits)
                st.plotly_chart(phase_fig, use_container_width=True, key="main_phase_fig")
                
                st.subheader(" Amplitude Table")
                amps = np.asarray(result['statevector']).flatten()
                df_amps = []
                for i, amp in enumerate(amps):
                    p = np.abs(amp)**2
                    if p > 1e-10:
                        df_amps.append({
                            "State": format(i, f"0{num_qubits}b"),
                            "Real": float(np.real(amp)),
                            "Imag": float(np.imag(amp)),
                            "Mag": float(np.abs(amp)),
                            "Phase (rad)": float(np.angle(amp)),
                            "Prob": float(p)
                        })
                if df_amps:
                    st.dataframe(df_amps, use_container_width=True)
            
            with tabs[3]: # 4. DENSITY MATRIX TAB
                if result.get('density_matrix') is not None:
                    st.subheader(" Density Matrix")
                    density_fig = plot_density_matrix_heatmap(result['density_matrix'])
                    st.plotly_chart(density_fig, use_container_width=True, key="main_density_fig")
                elif num_qubits > 6:
                    st.info("Density Matrix visualization disabled for > 6 qubits to conserve memory.")
                
                st.subheader(" Circuit Metrics")
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                col_m1.metric("Qubits", num_qubits)
                col_m2.metric("Gates", result['num_gates'])
                col_m3.metric("Depth", result.get('depth', 'N/A'))
                col_m4.metric("Size", result.get('size', 'N/A'))
                
                col_w1, col_w2, col_w3, col_w4 = st.columns(4)
                col_w1.metric("Width", result.get('width', 'N/A'))
                col_w2.metric("Time", f"{result['time']:.3f}s")
                st.caption("Single-qubit entropies require advanced density matrix analysis feature.")
            
            with tabs[4]: # 5. TIMELINE TAB
                st.subheader(" Timeline Explorer")
                if result.get('timeline'):
                    max_step = len(result['timeline']) - 1
                    
                    if 'tl_step' not in st.session_state:
                        st.session_state.tl_step = 0
                    if st.session_state.tl_step > max_step:
                        st.session_state.tl_step = max_step
                        
                    # Playback controls
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
                        sv_step = step_data['statevector']
                        step_probs = QuantumEngine.calculate_probabilities(sv_step, num_qubits)
                        st.plotly_chart(plot_statevector(step_probs), use_container_width=True, key="tl_sv_fig")
                    with col_t2:
                        step_bloch = QuantumEngine.generate_bloch_data(sv_step, num_qubits)
                        st.plotly_chart(plot_bloch_sphere(step_bloch), use_container_width=True, key="tl_bloch_fig")
                        
                    # Auto-play logic loop
                    if getattr(st.session_state, 'tl_playing', False):
                        if st.session_state.tl_step < max_step:
                            time.sleep(0.75)
                            st.session_state.tl_step += 1
                            st.rerun()
                        else:
                            st.session_state.tl_playing = False
                else:
                    st.info("Timeline data not available.")
            
            with tabs[5]: # 6. CODE TAB
                st.subheader(" OpenQASM 2")
                st.code(result['qasm'], language='qasm')
                st.subheader("📜 OpenQASM 3")
                try:
                    from qiskit import qasm3
                    qc_out = QuantumEngine.build_circuit(num_qubits, st.session_state.gates)
                    st.code(qasm3.dumps(qc_out), language='qasm')
                except ImportError:
                    st.info("OpenQASM 3 output is not available (Qiskit qasm3 module required).")
                except Exception as e:
                    st.warning(f"Could not generate OpenQASM 3: {e}")
            
            with tabs[6]: # 7. ENTANGLEMENT TAB
                st.subheader(" Entanglement Analytics")
                if result.get('entropies'):
                    fig = go.Figure(data=[go.Bar(
                        x=[f"Q{i}" for i in range(num_qubits)],
                        y=result['entropies'],
                        marker_color='#22c55e',
                        text=[f"{e:.3f}" for e in result['entropies']],
                        textposition='auto'
                    )])
                    fig.update_layout(
                        title="Single-Qubit Von Neumann Entropy",
                        xaxis_title="Qubit",
                        yaxis_title="Entropy (S)",
                        height=300,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig, use_container_width=True, key="entangle_bar_fig")
                    
                    max_entropy = max(result['entropies'])
                    if max_entropy > 0.1:
                        st.success(f"High entanglement detected! Maximum single-qubit entropy: {max_entropy:.3f}")
                    else:
                        st.info("Low or no entanglement detected in the current state.")
                else:
                    st.info("Entanglement metrics not available.")
                
            with tabs[7]: # 8. INTERFERENCE TAB
                st.subheader(" Quantum Interference")
                amps = np.asarray(result['statevector']).flatten()
                
                # Only show top states if too many qubits to avoid clutter
                if num_qubits > 6:
                    st.caption("Showing top 32 basis states due to large state space.")
                    indices = np.argsort(np.abs(amps))[-32:]
                    indices = np.sort(indices)
                else:
                    indices = np.arange(len(amps))
                    
                labels = [format(i, f"0{num_qubits}b") for i in indices]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(x=labels, y=np.real(amps[indices]), name='Real', marker_color='#3b82f6'))
                fig.add_trace(go.Bar(x=labels, y=np.imag(amps[indices]), name='Imaginary', marker_color='#f59e0b'))
                fig.update_layout(
                    title="State Amplitudes (Real vs Imaginary Interference)",
                    barmode='group',
                    height=350,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True, key="interfere_bar_fig")
            
            with tabs[8]: # 9. NOISE & DECOHERENCE TAB
                st.subheader(" Noise & Decoherence")
                if result.get('density_matrix') is not None:
                    dm = result['density_matrix']
                    purity = float(np.real(np.trace(np.dot(dm, dm))))
                    col_n1, col_n2 = st.columns(2)
                    col_n1.metric("State Purity Tr(ρ²)", f"{purity:.4f}")
                    if purity > 0.999:
                        col_n2.success("Pure State (Ideal)")
                    else:
                        col_n2.warning("Mixed State (Decohered)")
                else:
                    st.info("Run with a small number of qubits (≤6) to view purity metrics.")
            
            with tabs[9]: # 10. CIRCUIT STRUCTURE TAB
                st.subheader(" Circuit Structure")
                if st.session_state.gates:
                    gate_counts = [0] * num_qubits
                    for g in st.session_state.gates:
                        gate_counts[g.qubit] += 1
                        if g.control is not None:
                            gate_counts[g.control] += 1
                    fig = go.Figure(data=[go.Bar(x=[f"Q{i}" for i in range(num_qubits)], y=gate_counts, marker_color='#8b5cf6')])
                    fig.update_layout(title="Operations per Qubit", yaxis_title="Gate Count", height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True, key="struct_bar_fig")
                else:
                    st.info("Add gates to analyze structure.")
                
            with tabs[10]: # 11. FIDELITY & ERROR ANALYTICS TAB
                st.subheader(" Fidelity Analytics")
                if noise_model != 'ideal' and result.get('counts'):
                    total_shots = sum(result['counts'].values())
                    theoretical = result['probabilities']
                    actual = {k: v/total_shots for k, v in result['counts'].items()}
                    
                    all_keys = set(theoretical.keys()) | set(actual.keys())
                    # Classical Bhattacharyya fidelity
                    fidelity = sum(np.sqrt(theoretical.get(k, 0) * actual.get(k, 0)) for k in all_keys) ** 2
                    
                    col_f1, col_f2 = st.columns(2)
                    col_f1.metric("Classical Measurement Fidelity", f"{fidelity:.4f}")
                    if fidelity > 0.95:
                        col_f2.success("High fidelity")
                    elif fidelity > 0.7:
                        col_f2.warning("Moderate fidelity reduction")
                    else:
                        col_f2.error("Significant noise impact")
                        
                    st.info("The ideal expected distribution comparison:")
                    st.plotly_chart(plot_statevector(result['probabilities']), use_container_width=True, key="fid_ideal_sv_fig")
                else:
                    st.success("Simulation ran in IDEAL mode. Maximum Theoretical Fidelity achieved.")
                
            with tabs[11]: # 12. STATE SPACE EXPLORER TAB
                st.subheader(" State Space Explorer")
                st.write("Top highly probable basis states (Projection Collapse):")
                sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
                for state, prob in sorted_probs[:5]:
                    st.progress(prob, text=f"|{state}⟩ : {prob*100:.2f}%")
            
            with tabs[12]: # 13. CIRCUIT OPTIMIZATION TAB
                st.subheader(" Circuit Optimization")
                col_o1, col_o2 = st.columns(2)
                orig_depth, opt_depth = result.get('depth', 0), result.get('depth_opt', 0)
                orig_size, opt_size = result.get('size', 0), result.get('size_opt', 0)
                
                col_o1.metric("Optimized Depth", opt_depth, delta=opt_depth - orig_depth, delta_color="inverse")
                col_o2.metric("Optimized Size (Gates)", opt_size, delta=opt_size - orig_size, delta_color="inverse")
                
                with st.expander("View Optimized OpenQASM"):
                    st.code(result.get('qasm_opt', ''), language='qasm')
            
            with tabs[13]: # 14. QUANTUM ALGORITHM INSIGHT TAB
                st.subheader(" Algorithm Insight")
                gates_used = set(g.name for g in st.session_state.gates)
                if 'h' in gates_used and ('cx' in gates_used or 'cz' in gates_used):
                    st.write("💡 **Analysis:** Circuit contains superposition and entanglement generation (e.g., Bell or GHZ state architecture).")
                elif 'h' in gates_used:
                    st.write("💡 **Analysis:** Circuit utilizes superposition across parallel basis states.")
                else:
                    st.write("💡 **Analysis:** Circuit is executing classical-like operations exclusively in the Z-basis.")
            
            with tabs[14]: # 15. MEASUREMENT ANALYTICS TAB
                st.subheader(" Measurement Analytics")
                if result.get('counts'):
                    total_shots = sum(result['counts'].values())
                    st.write(f"**Total Shots Simulated:** {total_shots}")
                    
                    theoretical = result['probabilities']
                    actual = {k: v/total_shots for k, v in result['counts'].items()}
                    
                    all_keys = sorted(list(set(theoretical.keys()) | set(actual.keys())))
                    theo_vals = [theoretical.get(k, 0.0) for k in all_keys]
                    act_vals = [actual.get(k, 0.0) for k in all_keys]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=all_keys, y=theo_vals, name='Theoretical (Ideal)'))
                    fig.add_trace(go.Bar(x=all_keys, y=act_vals, name=f'Actual ({total_shots} shots)'))
                    fig.update_layout(barmode='group', title="Theoretical vs Actual Sampling", height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True, key="meas_comp_fig")
                else:
                    st.info("Run simulation with a noise model or shots > 0 to see statistical variance and sampling convergence.")

        else:
            st.info("Run simulation to see visualizations")
    
    # Footer
    st.divider()
    st.markdown("""
        <div style='text-align: center; color: #94a3b8; padding: 1rem;'>
            <p>Qircuit Studio - Quantum Circuit Visualizer | Built with Streamlit & Qiskit</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()