"""
Qanvas Studio - Quantum Circuit Visualizer (Streamlit MVP)
Python 3.10+

Run:
  streamlit run app.py
"""

import json
import time
import io
from typing import Optional, List, Dict, Any
from enum import Enum
from dataclasses import dataclass, field
from uuid import uuid4

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import transpile
from qiskit.quantum_info import Statevector, Operator, partial_trace
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    amplitude_damping_error,
    phase_damping_error,
    ReadoutError,
)
from qiskit.visualization import circuit_drawer

# ============================================================================
# Constants and Enums
# ============================================================================

class GateType(str, Enum):
    """Supported quantum gate types"""
    H = "h"
    X = "x"
    Y = "y"
    Z = "z"
    S = "s"
    SDG = "sdg"
    T = "t"
    TDG = "tdg"
    ID = "id"
    RX = "rx"
    RY = "ry"
    RZ = "rz"
    P = "p"
    CX = "cx"
    CY = "cy"
    CZ = "cz"
    SWAP = "swap"
    CCX = "ccx"
    MEASURE = "measure"
    RESET = "reset"
    BARRIER = "barrier"


class NoiseModelType(str, Enum):
    """Supported noise models"""
    IDEAL = "ideal"
    DEPOLARIZING = "depolarizing"
    AMPLITUDE_DAMPING = "amplitude_damping"
    PHASE_DAMPING = "phase_damping"
    READOUT = "readout"


@dataclass
class Gate:
    """Quantum gate data structure"""
    name: str
    label: str
    qubit: int
    position: int
    control: Optional[int] = None
    controls: Optional[List[int]] = None
    param: Optional[float] = None
    params: Optional[List[float]] = None
    description: str = ""
    id: str = field(default_factory=lambda: uuid4().hex)


# ============================================================================
# Undo/Redo helpers (session state)
# ============================================================================

def _gate_to_dict(g: Gate) -> Dict[str, Any]:
    return {
        "name": g.name,
        "label": g.label,
        "qubit": g.qubit,
        "position": g.position,
        "control": g.control,
        "controls": getattr(g, "controls", None),
        "param": g.param,
        "params": getattr(g, "params", None),
        "description": g.description,
        "id": g.id,
    }


def _gate_from_dict(d: Dict[str, Any]) -> Gate:
    controls = d.get("controls")
    if controls is not None:
        controls = [int(x) for x in controls]
    params = d.get("params")
    if params is not None:
        params = [float(x) for x in params]
    return Gate(
        name=str(d["name"]),
        label=str(d.get("label") or str(d["name"]).upper()),
        qubit=int(d["qubit"]),
        position=int(d["position"]),
        control=(None if d.get("control") is None else int(d["control"])),
        controls=controls,
        param=(None if d.get("param") is None else float(d["param"])),
        params=params,
        description=str(d.get("description") or ""),
        id=str(d.get("id") or uuid4().hex),
    )


def _snapshot_gates() -> List[Dict[str, Any]]:
    return [_gate_to_dict(g) for g in st.session_state.gates]


def _push_history():
    st.session_state.undo_stack.append(_snapshot_gates())
    st.session_state.redo_stack.clear()


def _apply_snapshot(snapshot: List[Dict[str, Any]]):
    st.session_state.gates = [_gate_from_dict(d) for d in snapshot]
    st.session_state.simulation_result = None


def _undo():
    if len(st.session_state.undo_stack) < 2:
        return
    current = st.session_state.undo_stack.pop()
    st.session_state.redo_stack.append(current)
    _apply_snapshot(st.session_state.undo_stack[-1])


def _redo():
    if not st.session_state.redo_stack:
        return
    snapshot = st.session_state.redo_stack.pop()
    st.session_state.undo_stack.append(snapshot)
    _apply_snapshot(snapshot)


# ============================================================================
# Custom gates (user-defined unitaries)
# ============================================================================

def _parse_complex(x: Any) -> complex:
    if isinstance(x, complex):
        return x
    if isinstance(x, (int, float, np.number)):
        return complex(float(x), 0.0)
    if isinstance(x, (list, tuple)) and len(x) == 2:
        return complex(float(x[0]), float(x[1]))
    if isinstance(x, str):
        s = x.strip().lower().replace(" ", "")
        s = s.replace("i", "j")
        return complex(s)
    raise ValueError(f"Unsupported complex format: {type(x)}")


def _matrix_from_text(text: str) -> np.ndarray:
    """Parse a matrix from JSON-ish text. Supports complex strings like '0.5+0.5j' or [re,im]."""
    raw = json.loads(text)
    arr = np.array([[ _parse_complex(v) for v in row ] for row in raw], dtype=complex)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("Matrix must be square")
    return arr


def _is_unitary(u: np.ndarray, tol: float = 1e-6) -> bool:
    if u.ndim != 2 or u.shape[0] != u.shape[1]:
        return False
    ident = np.eye(u.shape[0], dtype=complex)
    return np.allclose(u.conj().T @ u, ident, atol=tol, rtol=0)


def _export_project_json() -> str:
    return json.dumps(
        {
            "version": 1,
            "num_qubits": int(st.session_state.num_qubits),
            "gates": [_gate_to_dict(g) for g in st.session_state.gates],
            "custom_gates": st.session_state.custom_gates,
            "circuit_blocks": st.session_state.get("circuit_blocks", {}),
            "pipeline": st.session_state.get("pipeline", []),
        },
        indent=2,
        sort_keys=True,
    )


def _import_project_json(text: str):
    data = json.loads(text)
    num_qubits = int(data.get("num_qubits", 2))
    gates = data.get("gates", [])
    custom_gates = data.get("custom_gates", {})
    circuit_blocks = data.get("circuit_blocks", {})
    pipeline = data.get("pipeline", [])

    st.session_state.num_qubits = max(1, min(20, num_qubits))
    st.session_state.custom_gates = dict(custom_gates)
    st.session_state.gates = [_gate_from_dict(d) for d in gates]
    st.session_state.circuit_blocks = dict(circuit_blocks)
    st.session_state.pipeline = list(pipeline)
    st.session_state.simulation_result = None
    st.session_state.undo_stack = [_snapshot_gates()]
    st.session_state.redo_stack = []


def _qiskit_circuit_to_gates(qc: QuantumCircuit) -> List[Gate]:
    gates: List[Gate] = []
    pos = 0
    for instr in qc.data:
        name = instr.operation.name.lower()
        qubits = [qc.find_bit(q).index for q in instr.qubits]

        if name in {"h", "x", "y", "z", "s", "t", "id"}:
            gates.append(Gate(name=name, label=name.upper(), qubit=qubits[0], position=pos, description="Imported"))
            pos += 1
            continue
        if name in {"rx", "ry", "rz", "p"}:
            theta = None
            try:
                theta = float(instr.operation.params[0])
            except Exception:
                theta = None
            gates.append(
                Gate(
                    name=name,
                    label=name.upper(),
                    qubit=qubits[0],
                    position=pos,
                    param=theta,
                    description="Imported",
                )
            )
            pos += 1
            continue
        if name in {"cx", "cz", "swap"} and len(qubits) == 2:
            gates.append(
                Gate(
                    name=name,
                    label=("CNOT" if name == "cx" else name.upper()),
                    qubit=qubits[1],
                    control=qubits[0],
                    position=pos,
                    description="Imported",
                )
            )
            pos += 1
            continue
        if name == "measure" and len(qubits) == 1:
            gates.append(Gate(name="measure", label="M", qubit=qubits[0], position=pos, description="Imported"))
            pos += 1
            continue
        if name == "reset" and len(qubits) == 1:
            gates.append(Gate(name="reset", label="R", qubit=qubits[0], position=pos, description="Imported"))
            pos += 1
            continue
        if name == "barrier":
            # Skip barriers to keep editor simple
            continue

    return gates


def _normalize_gate_positions(gates: List[Gate], start_at: int = 0) -> List[Gate]:
    """Return a copy of gates with positions re-based starting at start_at (preserves relative order)."""
    if not gates:
        return []
    sorted_g = sorted(gates, key=lambda g: (int(g.position), g.id))
    rebased: List[Gate] = []
    for i, g in enumerate(sorted_g):
        rebased.append(
            Gate(
                name=g.name,
                label=g.label,
                qubit=int(g.qubit),
                position=int(start_at + i),
                control=g.control,
                controls=(g.controls[:] if g.controls else None),
                param=g.param,
                params=(g.params[:] if g.params else None),
                description=g.description,
                id=g.id,
            )
        )
    return rebased


def _insert_gate_at(new_gate: Gate, position: int):
    """Insert a gate at a specific position by shifting later gates to the right."""
    position = int(max(0, position))
    shifted: List[Gate] = []
    for g in st.session_state.gates:
        if int(g.position) >= position:
            shifted.append(
                Gate(
                    name=g.name,
                    label=g.label,
                    qubit=int(g.qubit),
                    position=int(g.position) + 1,
                    control=g.control,
                    controls=(g.controls[:] if g.controls else None),
                    param=g.param,
                    params=(g.params[:] if g.params else None),
                    description=g.description,
                    id=g.id,
                )
            )
        else:
            shifted.append(g)
    new_gate.position = position
    shifted.append(new_gate)
    st.session_state.gates = sorted(shifted, key=lambda gg: (int(gg.position), gg.id))


def _compose_blocks(block_ids: List[str]) -> Dict[str, Any]:
    """
    Compose saved blocks end-to-end. Returns:
      { num_qubits, gates, custom_gates }
    Enforces consistent num_qubits across blocks.
    """
    blocks = st.session_state.get("circuit_blocks", {})
    if not block_ids:
        raise ValueError("No blocks selected")

    selected = []
    for bid in block_ids:
        b = blocks.get(bid)
        if not b:
            raise ValueError(f"Unknown block id: {bid}")
        selected.append(b)

    num_qubits = int(selected[0]["num_qubits"])
    for b in selected:
        if int(b["num_qubits"]) != num_qubits:
            raise ValueError("All blocks must have the same number of qubits to compose end-to-end.")

    composed_gates: List[Gate] = []
    composed_custom: Dict[str, Any] = {}

    pos = 0
    for b in selected:
        # Merge custom gates used by blocks (later definitions overwrite)
        composed_custom.update(dict(b.get("custom_gates", {})))
        block_gates = [_gate_from_dict(d) for d in b.get("gates", [])]
        block_gates = _normalize_gate_positions(block_gates, start_at=pos)
        composed_gates.extend(block_gates)
        pos = (max([g.position for g in composed_gates]) + 1) if composed_gates else 0

    return {"num_qubits": num_qubits, "gates": composed_gates, "custom_gates": composed_custom}


# Gate definitions
GATE_DEFINITIONS = {
    'single_qubit': [
        {'name': 'id', 'label': 'I', 'description': 'Identity gate'},
        {'name': 'h', 'label': 'H', 'description': 'Hadamard gate'},
        {'name': 'x', 'label': 'X', 'description': 'Pauli-X gate'},
        {'name': 'y', 'label': 'Y', 'description': 'Pauli-Y gate'},
        {'name': 'z', 'label': 'Z', 'description': 'Pauli-Z gate'},
        {'name': 's', 'label': 'S', 'description': 'Phase gate'},
        {'name': 'sdg', 'label': 'Sdg', 'description': 'Inverse S gate'},
        {'name': 't', 'label': 'T', 'description': 'T gate'},
        {'name': 'tdg', 'label': 'Tdg', 'description': 'Inverse T gate'},
        {'name': 'sx', 'label': 'SX', 'description': 'Sqrt(X) gate'},
        {'name': 'sxdg', 'label': 'SXdg', 'description': 'Inverse sqrt(X) gate'},
        {'name': 'sy', 'label': 'SY', 'description': 'Sqrt(Y) gate (unitary)'},
        {'name': 'p', 'label': 'P', 'description': 'Phase shift P(lambda)'},
        {'name': 'rx', 'label': 'Rx', 'description': 'X rotation'},
        {'name': 'ry', 'label': 'Ry', 'description': 'Y rotation'},
        {'name': 'rz', 'label': 'Rz', 'description': 'Z rotation'},
        {'name': 'u1', 'label': 'U1', 'description': 'Phase-only rotation U1(lambda)'},
        {'name': 'u2', 'label': 'U2', 'description': 'Universal U2(phi,lambda)'},
        {'name': 'u3', 'label': 'U3', 'description': 'Universal U3(theta,phi,lambda)'},
        {'name': 'global_phase', 'label': 'GP', 'description': 'Global phase (UI utility)'},
    ],
    'multi_qubit': [
        {'name': 'cx', 'label': 'CNOT', 'description': 'Controlled-NOT'},
        {'name': 'cz', 'label': 'CZ', 'description': 'Controlled-Z'},
        {'name': 'cy', 'label': 'CY', 'description': 'Controlled-Y'},
        {'name': 'ch', 'label': 'CH', 'description': 'Controlled-H'},
        {'name': 'cs', 'label': 'CS', 'description': 'Controlled-S'},
        {'name': 'ct', 'label': 'CT', 'description': 'Controlled-T'},
        {'name': 'cp', 'label': 'CP', 'description': 'Controlled phase CP(lambda)'},
        {'name': 'crx', 'label': 'CRx', 'description': 'Controlled Rx(theta)'},
        {'name': 'cry', 'label': 'CRy', 'description': 'Controlled Ry(theta)'},
        {'name': 'crz', 'label': 'CRz', 'description': 'Controlled Rz(theta)'},
        {'name': 'xx', 'label': 'XX', 'description': 'XX interaction (RXX(theta))'},
        {'name': 'yy', 'label': 'YY', 'description': 'YY interaction (RYY(theta))'},
        {'name': 'zz', 'label': 'ZZ', 'description': 'ZZ interaction (RZZ(theta))'},
        {'name': 'rxx', 'label': 'RXX', 'description': 'RXX(theta)'},
        {'name': 'ryy', 'label': 'RYY', 'description': 'RYY(theta)'},
        {'name': 'rzz', 'label': 'RZZ', 'description': 'RZZ(theta)'},
        {'name': 'swap', 'label': 'SWAP', 'description': 'Swap qubits'},
        {'name': 'iswap', 'label': 'iSWAP', 'description': 'iSWAP gate'},
        {'name': 'sqrt_swap', 'label': 'SqrtSWAP', 'description': 'sqrt(SWAP) gate'},
        {'name': 'ecr', 'label': 'ECR', 'description': 'IBM echoed cross resonance (if available)'},
        {'name': 'ccx', 'label': 'CCX', 'description': 'Toffoli (2 controls)'},
        {'name': 'ccz', 'label': 'CCZ', 'description': 'CCZ (2 controls)'},
        {'name': 'mcx', 'label': 'MCX', 'description': 'Multi-controlled X'},
        {'name': 'mcz', 'label': 'MCZ', 'description': 'Multi-controlled Z'},
    ],
    'other': [
        {'name': 'measure', 'label': 'M', 'description': 'Measurement'},
        {'name': 'measure_x', 'label': 'Mx', 'description': 'Measure in X basis (H then measure)'},
        {'name': 'measure_z', 'label': 'Mz', 'description': 'Measure in Z basis'},
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
        
        if q >= num_qubits:
            return
            
        # Helper: resolve controls list
        controls_list: List[int] = []
        if gate.controls is not None:
            controls_list = [int(x) for x in gate.controls]
        elif gate.control is not None:
            controls_list = [int(gate.control)]

        def _validate_qubits(indices: List[int]):
            for idx in indices:
                if idx < 0 or idx >= num_qubits:
                    raise ValueError(f"Qubit out of range: {idx}")

        if name in {"id", "i"}:
            qc.id(q)
        elif name == 'h':
            qc.h(q)
        elif name == 'x':
            qc.x(q)
        elif name == 'y':
            qc.y(q)
        elif name == 'z':
            qc.z(q)
        elif name == 's':
            qc.s(q)
        elif name == 'sdg':
            qc.sdg(q)
        elif name == 't':
            qc.t(q)
        elif name == 'tdg':
            qc.tdg(q)
        elif name == 'sx':
            qc.sx(q)
        elif name == 'sxdg':
            qc.sxdg(q)
        elif name == 'sy':
            # sqrt(Y) is equivalent to Ry(pi/2) up to a global phase (global phase is physically irrelevant)
            qc.ry(np.pi / 2, q)
        elif name == 'rx' and gate.param is not None:
            qc.rx(gate.param, q)
        elif name == 'ry' and gate.param is not None:
            qc.ry(gate.param, q)
        elif name == 'rz' and gate.param is not None:
            qc.rz(gate.param, q)
        elif name in {"p", "u1"}:
            # U1(lambda) == P(lambda)
            lam = gate.param if gate.param is not None else (gate.params[0] if gate.params else None)
            if lam is None:
                raise ValueError("P/U1 requires lambda")
            qc.p(float(lam), q)
        elif name == "u2":
            if not gate.params or len(gate.params) < 2:
                raise ValueError("U2 requires [phi, lambda] in params")
            phi, lam = float(gate.params[0]), float(gate.params[1])
            qc.u(np.pi / 2, phi, lam, q)
        elif name == "u3":
            if not gate.params or len(gate.params) < 3:
                raise ValueError("U3 requires [theta, phi, lambda] in params")
            theta, phi, lam = float(gate.params[0]), float(gate.params[1]), float(gate.params[2])
            qc.u(theta, phi, lam, q)
        elif name == "global_phase":
            ph = gate.param if gate.param is not None else (gate.params[0] if gate.params else None)
            if ph is None:
                raise ValueError("Global phase requires a value")
            qc.global_phase = (qc.global_phase or 0) + float(ph)
        elif name in {"cx", "cz", "cy", "ch", "cs", "ct", "swap", "iswap", "sqrt_swap", "cp", "crx", "cry", "crz", "rxx", "ryy", "rzz", "xx", "yy", "zz", "ecr"}:
            if not controls_list:
                raise ValueError(f"{name.upper()} requires a control/other qubit")
            c = int(controls_list[0])
            _validate_qubits([c, q])
            if c == q:
                raise ValueError("duplicate bit arguments (control == target)")
            if name == "cx":
                qc.cx(c, q)
            elif name == "cz":
                qc.cz(c, q)
            elif name == "cy":
                qc.cy(c, q)
            elif name == "ch":
                qc.ch(c, q)
            elif name == "cs":
                qc.cs(c, q)
            elif name == "ct":
                # Controlled-T == controlled phase of pi/4 on |11>
                qc.cp(np.pi / 4, c, q)
            elif name == "swap":
                qc.swap(c, q)
            elif name == "iswap":
                qc.iswap(c, q)
            elif name == "sqrt_swap":
                if hasattr(qc, "sqrt_swap"):
                    qc.sqrt_swap(c, q)  # type: ignore[attr-defined]
                else:
                    # fallback unitary
                    u = np.array(
                        [
                            [1, 0, 0, 0],
                            [0, 0.5 + 0.5j, 0.5 - 0.5j, 0],
                            [0, 0.5 - 0.5j, 0.5 + 0.5j, 0],
                            [0, 0, 0, 1],
                        ],
                        dtype=complex,
                    )
                    qc.append(UnitaryGate(u, label="SqrtSWAP"), [c, q])
            elif name == "cp":
                lam = gate.param if gate.param is not None else (gate.params[0] if gate.params else None)
                if lam is None:
                    raise ValueError("CP requires lambda")
                qc.cp(float(lam), c, q)
            elif name == "crx":
                th = gate.param if gate.param is not None else (gate.params[0] if gate.params else None)
                if th is None:
                    raise ValueError("CRX requires theta")
                qc.crx(float(th), c, q)
            elif name == "cry":
                th = gate.param if gate.param is not None else (gate.params[0] if gate.params else None)
                if th is None:
                    raise ValueError("CRY requires theta")
                qc.cry(float(th), c, q)
            elif name == "crz":
                th = gate.param if gate.param is not None else (gate.params[0] if gate.params else None)
                if th is None:
                    raise ValueError("CRZ requires theta")
                qc.crz(float(th), c, q)
            elif name in {"rxx", "xx"}:
                th = gate.param if gate.param is not None else (gate.params[0] if gate.params else None)
                if th is None:
                    raise ValueError("RXX/XX requires theta")
                qc.rxx(float(th), c, q)
            elif name in {"ryy", "yy"}:
                th = gate.param if gate.param is not None else (gate.params[0] if gate.params else None)
                if th is None:
                    raise ValueError("RYY/YY requires theta")
                qc.ryy(float(th), c, q)
            elif name in {"rzz", "zz"}:
                th = gate.param if gate.param is not None else (gate.params[0] if gate.params else None)
                if th is None:
                    raise ValueError("RZZ/ZZ requires theta")
                qc.rzz(float(th), c, q)
            elif name == "ecr":
                if hasattr(qc, "ecr"):
                    qc.ecr(c, q)  # type: ignore[attr-defined]
                else:
                    raise ValueError("ECR not available in this Qiskit version")
        elif name in {"ccx", "ccz", "mcx", "mcz"}:
            # Multi-control family uses gate.controls (preferred). gate.qubit is the target.
            ctrls = controls_list
            if name in {"ccx", "ccz"}:
                if len(ctrls) < 2:
                    raise ValueError(f"{name.upper()} requires 2 controls")
                ctrls = ctrls[:2]
            else:
                if len(ctrls) < 2:
                    raise ValueError(f"{name.upper()} requires at least 2 controls")
            _validate_qubits(ctrls + [q])
            if q in ctrls:
                raise ValueError("duplicate bit arguments (a control equals target)")
            if name == "ccx":
                qc.ccx(ctrls[0], ctrls[1], q)
            elif name == "ccz":
                # CCZ via H-target + CCX + H-target
                qc.h(q)
                qc.ccx(ctrls[0], ctrls[1], q)
                qc.h(q)
            elif name == "mcx":
                qc.mcx(ctrls, q)
            else:
                # MCZ via H-target + MCX + H-target
                qc.h(q)
                qc.mcx(ctrls, q)
                qc.h(q)
        elif name.startswith("custom:"):
            gid = name.split("custom:", 1)[1]
            custom = getattr(st.session_state, "custom_gates", {}).get(gid)
            if not custom:
                raise ValueError(f"Unknown custom gate id: {gid}")
            arity = int(custom.get("arity", 1))
            matrix_text = str(custom.get("matrix_text", ""))
            u = _matrix_from_text(matrix_text)
            if not _is_unitary(u):
                raise ValueError("Custom gate matrix is not unitary")
            ug = UnitaryGate(u, label=str(custom.get("label", gate.label or "U")))
            if arity == 1:
                qc.append(ug, [q])
            else:
                if not controls_list:
                    raise ValueError("Custom 2-qubit gate requires Control Qubit")
                c = int(controls_list[0])
                if c == q:
                    raise ValueError("duplicate bit arguments (control == target)")
                qc.append(ug, [c, q])
        elif name in {"measure", "measure_z"}:
            qc.measure(q, q)
        elif name == "measure_x":
            qc.h(q)
            qc.measure(q, q)
        elif name == 'reset':
            qc.reset(q)
        elif name == 'barrier':
            qc.barrier(q)
    
    @staticmethod
    def simulate_statevector(qc: QuantumCircuit) -> np.ndarray:
        """Run statevector simulation (deployment-safe).

        Prefer Qiskit quantum_info Statevector evolution (no Aer dependency, no 'unknown instruction' issues).
        Fall back to Aer if needed.
        """
        qc_sim = qc.copy()
        
        # Check for measurements (compatible with different Qiskit versions)
        has_measurements = False
        try:
            has_measurements = qc_sim.has_measurements()
        except AttributeError:
            has_measurements = any(instr.operation.name == 'measure' for instr in qc_sim.data)
        
        if has_measurements:
            qc_sim.remove_final_measurements(inplace=True)

        # Best-effort normalize to common basis gates so Statevector evolution works broadly
        try:
            qc_sim = transpile(qc_sim, basis_gates=["u", "cx"], optimization_level=0)
        except Exception:
            pass

        # Primary path: no Aer required
        try:
            sv = Statevector.from_instruction(qc_sim)
            return np.asarray(sv.data, dtype=complex)
        except Exception:
            pass

        # Fallback: Aer statevector
        simulator = AerSimulator(method="statevector")
        try:
            qc_aer = transpile(qc_sim, simulator, optimization_level=0)
        except Exception:
            qc_aer = qc_sim
        qc_aer.save_statevector()
        result = simulator.run(qc_aer).result()
        return result.get_statevector()
    
    @staticmethod
    def simulate_with_noise(qc: QuantumCircuit, noise_type: str, shots: int) -> Dict:
        """Run noisy simulation"""
        simulator = AerSimulator()
        
        if noise_type == 'depolarizing':
            noise_model = NoiseModel()
            error_1q = depolarizing_error(0.01, 1)
            error_2q = depolarizing_error(0.05, 2)
            noise_model.add_all_qubit_quantum_error(error_1q, ['h', 'x', 'y', 'z'])
            noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cz'])
            simulator.set_options(noise_model=noise_model)
        elif noise_type == 'amplitude_damping':
            noise_model = NoiseModel()
            error = amplitude_damping_error(0.05)
            noise_model.add_all_qubit_quantum_error(error, ['h', 'x', 'y', 'z'])
            simulator.set_options(noise_model=noise_model)
        elif noise_type == 'phase_damping':
            noise_model = NoiseModel()
            error = phase_damping_error(0.03)
            noise_model.add_all_qubit_quantum_error(error, ['h', 'x', 'y', 'z', 's', 't', 'rx', 'ry', 'rz'])
            simulator.set_options(noise_model=noise_model)
        elif noise_type == 'readout':
            noise_model = NoiseModel()
            ro = ReadoutError([[0.98, 0.02], [0.02, 0.98]])
            noise_model.add_all_qubit_readout_error(ro)
            simulator.set_options(noise_model=noise_model)
        
        qc_sim = qc.copy()
        
        has_measurements = False
        try:
            has_measurements = qc_sim.has_measurements()
        except AttributeError:
            has_measurements = any(instr.operation.name == 'measure' for instr in qc_sim.data)
        
        if not has_measurements:
            qc_sim.measure_all()

        # Transpile to Aer-supported instructions to avoid "unknown instruction" on cloud deployments
        try:
            qc_sim = transpile(qc_sim, simulator, optimization_level=0)
        except Exception:
            pass

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
    def calculate_density_matrix(statevector: np.ndarray) -> np.ndarray:
        """Calculate density matrix from statevector"""
        return np.outer(statevector, np.conj(statevector))

    @staticmethod
    def single_qubit_entropies(statevector: np.ndarray, num_qubits: int) -> List[float]:
        """Von Neumann entropy of each 1-qubit reduced state (0..1 for pure->maximally mixed)."""
        sv = Statevector(statevector)
        entropies: List[float] = []
        for q in range(num_qubits):
            try:
                rho = partial_trace(sv, [i for i in range(num_qubits) if i != q]).data
                # eigenvalues are real (up to numeric noise)
                evals = np.real(np.linalg.eigvals(rho))
                evals = np.clip(evals, 0.0, 1.0)
                evals = evals / max(1e-12, float(np.sum(evals)))
                ent = float(-np.sum([p * np.log2(p) for p in evals if p > 1e-12]))
                # For a single qubit, max entropy is 1
                entropies.append(min(1.0, max(0.0, ent)))
            except Exception:
                entropies.append(0.0)
        return entropies

    @staticmethod
    def simulate_timeline(qc: QuantumCircuit, num_qubits: int, max_steps: int = 200) -> Dict[str, Any]:
        """
        Step-by-step state evolution for visualization.
        Stops at first measurement. Returns:
          { "ops": [ {name, qubits, label}... ], "states": [statevector0, statevector1, ...] }
        where states[i] is after applying ops[i-1] (states[0] is initial).
        """
        ops: List[Dict[str, Any]] = []
        states: List[np.ndarray] = []

        sv = Statevector.from_int(0, 2**num_qubits)
        states.append(np.asarray(sv.data, dtype=complex))

        for instr in qc.data[:max_steps]:
            name = instr.operation.name.lower()
            if name == "measure":
                break
            try:
                qidx = [qc.find_bit(q).index for q in instr.qubits]
            except Exception:
                qidx = []
            ops.append(
                {
                    "name": name,
                    "qubits": qidx,
                    "label": getattr(instr.operation, "label", None) or name.upper(),
                }
            )
            try:
                sv = sv.evolve(instr.operation, qargs=qidx if qidx else None)
            except Exception:
                # If something can't be evolved (rare), stop the timeline rather than crashing UI
                break
            states.append(np.asarray(sv.data, dtype=complex))

        return {"ops": ops, "states": states}
    
    @staticmethod
    def generate_bloch_data(statevector: np.ndarray, num_qubits: int) -> List[Dict]:
        """Generate Bloch sphere data"""
        bloch_data = []
        sv = Statevector(statevector)
        
        for qubit_idx in range(min(num_qubits, 3)):
            try:
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
            except Exception:
                bloch_data.append({
                    'qubit': qubit_idx,
                    'x': 0.0, 'y': 0.0, 'z': 1.0,
                    'purity': 1.0
                })
        
        return bloch_data


# ============================================================================
# Circuit Visualization Functions
# ============================================================================

def render_circuit_diagram(qc: QuantumCircuit, num_qubits: int):
    """Render circuit diagram using Qiskit matplotlib backend"""
    try:
        # Draw circuit using matplotlib backend
        fig = circuit_drawer(
            qc,
            output='mpl',
            style={'name': 'bw'},
            fold=-1  # No folding
        )

        # Best-effort sizing for readability (API differs across Qiskit versions)
        if hasattr(fig, "set_size_inches"):
            fig.set_size_inches(12, 2 + num_qubits * 0.8)
        
        # Convert matplotlib figure to Streamlit-compatible format
        st.pyplot(fig, bbox_inches='tight')
        
    except Exception as e:
        # Fallback to text-based rendering if matplotlib fails
        st.warning(f"Circuit diagram rendering issue: {e}")
        render_circuit_text(qc, num_qubits)


def circuit_png_bytes(qc: QuantumCircuit, num_qubits: int) -> Optional[bytes]:
    """Return a PNG bytes render of the circuit, or None if rendering fails."""
    try:
        fig = circuit_drawer(
            qc,
            output="mpl",
            style={"name": "bw"},
            fold=-1,
        )
        if hasattr(fig, "set_size_inches"):
            fig.set_size_inches(12, 2 + num_qubits * 0.8)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=200)
        return buf.getvalue()
    except Exception:
        return None


def render_circuit_text(qc: QuantumCircuit, num_qubits: int):
    """Text-based circuit renderer (no matplotlib required)."""
    st.subheader("Circuit (Text View)")
    try:
        st.code(qc.draw(output="text").single_string(), language="text")
    except Exception:
        st.code(str(qc.draw(output="text")), language="text")


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
        template="plotly_dark",
        scene=dict(
            xaxis=dict(title='X', range=[-axis_range, axis_range], showbackground=False),
            yaxis=dict(title='Y', range=[-axis_range, axis_range], showbackground=False),
            zaxis=dict(title='Z', range=[-axis_range, axis_range], showbackground=False),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220"
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
        template="plotly_dark",
        title='Statevector Probabilities',
        xaxis_title='Basis State',
        yaxis_title='Probability',
        yaxis=dict(range=[0, 1]),
        height=300,
        showlegend=False,
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220"
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


def plot_entanglement_entropy(entanglement_entropies: List[float], num_qubits: int) -> go.Figure:
    """Create entanglement entropy bar chart"""
    if not entanglement_entropies or all(e == 0 for e in entanglement_entropies):
        fig = go.Figure()
        fig.add_annotation(text="Run simulation to see entanglement entropy",
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    qubit_labels = [f'Qubit {i}' for i in range(num_qubits)]
    
    fig = go.Figure(data=[
        go.Bar(
            x=qubit_labels,
            y=entanglement_entropies,
            marker_color='#8b5cf6',
            text=[f'{e:.3f}' for e in entanglement_entropies],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='Entanglement Entropy per Qubit',
        xaxis_title='Qubit',
        yaxis_title='Entanglement Entropy (bits)',
        yaxis=dict(range=[0, 1]),  # Max entanglement for a qubit is 1 bit
        height=300,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig
    
    labels = list(counts.keys())
    values = list(counts.values())
    
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
        template="plotly_dark",
        title=f'Measurement Results ({sum(values)} shots)',
        height=300,
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220"
    )
    
    return fig


def plot_density_matrix_heatmap(density: np.ndarray) -> go.Figure:
    """Heatmap of |rho_ij| for quick decoherence/entanglement intuition."""
    mag = np.abs(density)
    fig = go.Figure(
        data=go.Heatmap(
            z=mag,
            colorscale="Viridis",
            colorbar=dict(title="|rho|"),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title="Density Matrix Magnitude |rho_ij|",
        height=320,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220",
    )
    return fig


def plot_statevector_complex(statevector: np.ndarray, num_qubits: int) -> go.Figure:
    """Scatter of amplitudes in the complex plane (Re vs Im), sized by |amp|."""
    amps = np.asarray(statevector, dtype=complex).flatten()
    labels = [format(i, f"0{num_qubits}b") for i in range(len(amps))]
    mag = np.abs(amps)
    phase = np.angle(amps)

    fig = go.Figure(
        data=go.Scatter(
            x=np.real(amps),
            y=np.imag(amps),
            mode="markers+text",
            text=labels,
            textposition="top center",
            marker=dict(
                size=np.clip(6 + 30 * mag, 6, 32),
                color=phase,
                colorscale="HSV",
                showscale=True,
                colorbar=dict(title="phase (rad)"),
                line=dict(width=1, color="rgba(255,255,255,0.15)"),
            ),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title="Statevector Amplitudes (Complex Plane)",
        xaxis_title="Re(amp)",
        yaxis_title="Im(amp)",
        height=340,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220",
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def statevector_table(statevector: np.ndarray, num_qubits: int) -> List[Dict[str, Any]]:
    amps = np.asarray(statevector, dtype=complex).flatten()
    rows: List[Dict[str, Any]] = []
    for i, a in enumerate(amps):
        p = float(np.abs(a) ** 2)
        if p < 1e-12:
            continue
        rows.append(
            {
                "basis": format(i, f"0{num_qubits}b"),
                "re": float(np.real(a)),
                "im": float(np.imag(a)),
                "abs": float(np.abs(a)),
                "phase_rad": float(np.angle(a)),
                "prob": p,
            }
        )
    rows.sort(key=lambda r: r["prob"], reverse=True)
    return rows[: min(len(rows), 64)]


# ============================================================================
# Streamlit UI
# ============================================================================

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Qanvas Studio",
        page_icon="Q",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS (dark app, white circuit surface)
    st.markdown("""
        <style>
        .main { background-color: #0b1220; color: #e2e8f0; }
        div[data-testid="stAppViewContainer"] { background-color: #0b1220; }
        div[data-testid="stHeader"] { background-color: #0b1220; }
        div[data-testid="stSidebar"] { background-color: #0f172a; }
        .stAlert { background-color: #0f172a; border: 1px solid #334155; }
        div[data-testid="stMetricValue"] { font-size: 24px; }
        .stButton>button { width: 100%; }

        .circuit-surface {
            background: #ffffff;
            color: #0f172a;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 12px 12px 4px 12px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.25);
        }
        .circuit-surface pre {
            background: #ffffff !important;
            color: #0f172a !important;
            border: 1px solid #e2e8f0 !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("Qanvas Studio")
    st.markdown("Interactive quantum circuit builder + live simulation (statevector + Bloch sphere).")
    
    # Initialize session state
    if 'gates' not in st.session_state:
        st.session_state.gates = []
    if 'num_qubits' not in st.session_state:
        st.session_state.num_qubits = 2
    if 'simulation_result' not in st.session_state:
        st.session_state.simulation_result = None
    if 'circuit_display_mode' not in st.session_state:
        st.session_state.circuit_display_mode = 'diagram'
    if 'undo_stack' not in st.session_state:
        st.session_state.undo_stack = [_snapshot_gates()]
    if 'redo_stack' not in st.session_state:
        st.session_state.redo_stack = []
    if not st.session_state.undo_stack:
        st.session_state.undo_stack = [_snapshot_gates()]
    if "custom_gates" not in st.session_state:
        # id -> {id,label,arity,matrix_text}
        st.session_state.custom_gates = {}
    if "circuit_blocks" not in st.session_state:
        # id -> {id,name,num_qubits,gates,custom_gates}
        st.session_state.circuit_blocks = {}
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = []

    # Migrate older Gate objects stored in Streamlit session_state (from previous runs)
    migrated: List[Gate] = []
    changed = False
    for g in list(st.session_state.gates):
        try:
            if isinstance(g, Gate) and hasattr(g, "controls") and hasattr(g, "params"):
                migrated.append(g)
            else:
                migrated.append(
                    Gate(
                        name=getattr(g, "name", ""),
                        label=getattr(g, "label", str(getattr(g, "name", "")).upper()),
                        qubit=int(getattr(g, "qubit", 0)),
                        position=int(getattr(g, "position", 0)),
                        control=(None if getattr(g, "control", None) is None else int(getattr(g, "control"))),
                        controls=getattr(g, "controls", None),
                        param=(None if getattr(g, "param", None) is None else float(getattr(g, "param"))),
                        params=getattr(g, "params", None),
                        description=str(getattr(g, "description", "")),
                        id=str(getattr(g, "id", uuid4().hex)),
                    )
                )
                changed = True
        except Exception:
            # If something is badly formed, drop it rather than crashing the whole app.
            changed = True

    if changed:
        st.session_state.gates = migrated
        st.session_state.simulation_result = None
        st.session_state.undo_stack = [_snapshot_gates()]
        st.session_state.redo_stack = []
    
    # Sidebar
    with st.sidebar:
        st.header("Gate Palette")

        u1, u2 = st.columns(2)
        with u1:
            st.button("Undo", on_click=_undo, disabled=(len(st.session_state.undo_stack) < 2), use_container_width=True)
        with u2:
            st.button("Redo", on_click=_redo, disabled=(len(st.session_state.redo_stack) == 0), use_container_width=True)
        
        # Gate selection
        st.subheader("Single-Qubit Gates")
        for gate in GATE_DEFINITIONS['single_qubit']:
            if st.button(f"{gate['label']} - {gate['description']}", key=f"gate_{gate['name']}"):
                _push_history()
                default_param = None
                default_params = None
                if gate["name"] in {"rx", "ry", "rz"}:
                    default_param = float(np.pi / 2)
                elif gate["name"] in {"p", "u1"}:
                    default_param = float(np.pi / 4)
                elif gate["name"] == "u2":
                    default_params = [float(0.0), float(np.pi)]
                elif gate["name"] == "u3":
                    default_params = [float(np.pi / 2), float(0.0), float(np.pi)]
                elif gate["name"] == "global_phase":
                    default_param = float(0.0)
                new_gate = Gate(
                    name=gate['name'],
                    label=gate['label'],
                    qubit=0,
                    position=len(st.session_state.gates),
                    param=default_param,
                    params=default_params,
                    description=gate['description']
                )
                st.session_state.gates.append(new_gate)
                _push_history()
                st.rerun()
        
        st.subheader("Multi-Qubit Gates")
        for gate in GATE_DEFINITIONS['multi_qubit']:
            if st.button(f"{gate['label']} - {gate['description']}", key=f"gate_{gate['name']}"):
                _push_history()
                default_param = None
                if gate["name"] in {"crx", "cry", "crz", "rxx", "ryy", "rzz", "xx", "yy", "zz"}:
                    default_param = float(np.pi / 2)
                elif gate["name"] == "cp":
                    default_param = float(np.pi / 4)
                new_gate = Gate(
                    name=gate['name'],
                    label=gate['label'],
                    qubit=1,
                    control=0,
                    position=len(st.session_state.gates),
                    param=default_param,
                    description=gate['description']
                )
                if gate["name"] in {"ccx", "ccz"}:
                    new_gate.controls = [0, min(1, st.session_state.num_qubits - 1)]
                if gate["name"] in {"mcx", "mcz"}:
                    new_gate.controls = [0, min(1, st.session_state.num_qubits - 1)]
                st.session_state.gates.append(new_gate)
                _push_history()
                st.rerun()
        
        st.subheader("Other Operations")
        for gate in GATE_DEFINITIONS['other']:
            if st.button(f"{gate['label']} - {gate['description']}", key=f"gate_{gate['name']}"):
                _push_history()
                new_gate = Gate(
                    name=gate['name'],
                    label=gate['label'],
                    qubit=0,
                    position=len(st.session_state.gates),
                    description=gate['description']
                )
                st.session_state.gates.append(new_gate)
                _push_history()
                st.rerun()

        # Custom gates
        st.divider()
        st.subheader("Custom Gates")

        with st.expander("Create / Edit Custom Gate", expanded=False):
            cg_label = st.text_input("Gate label", value="U")
            cg_arity = st.selectbox("Arity", options=[1, 2], index=0)
            default_matrix = (
                "[[1,0],[0,1]]" if cg_arity == 1 else "[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]"
            )
            cg_matrix_text = st.text_area(
                "Unitary matrix (JSON). Complex allowed: \"0.5+0.5j\" or [re,im].",
                value=default_matrix,
                height=140,
            )

            col_cg1, col_cg2 = st.columns(2)
            with col_cg1:
                if st.button("Save Custom Gate", use_container_width=True):
                    try:
                        u = _matrix_from_text(cg_matrix_text)
                        expected = 2 ** int(cg_arity)
                        if u.shape != (expected, expected):
                            raise ValueError(f"Expected a {expected}x{expected} matrix for arity {cg_arity}")
                        if not _is_unitary(u):
                            raise ValueError("Matrix is not unitary (U†U != I)")
                        gate_id = uuid4().hex[:10]
                        st.session_state.custom_gates[gate_id] = {
                            "id": gate_id,
                            "label": cg_label.strip() or "U",
                            "arity": int(cg_arity),
                            "matrix_text": cg_matrix_text,
                        }
                        st.success("Saved.")
                    except Exception as e:
                        st.error(f"Could not save: {e}")
            with col_cg2:
                if st.button("Validate Matrix", use_container_width=True):
                    try:
                        u = _matrix_from_text(cg_matrix_text)
                        ok = _is_unitary(u)
                        st.info(f"Shape: {u.shape[0]}x{u.shape[1]} | Unitary: {ok}")
                    except Exception as e:
                        st.error(f"Invalid: {e}")

        if st.session_state.custom_gates:
            for gid, gdef in list(st.session_state.custom_gates.items()):
                label = gdef.get("label", "U")
                arity = int(gdef.get("arity", 1))
                row1, row2 = st.columns([3, 1])
                with row1:
                    if st.button(f"Add {label} (custom, {arity}q)", key=f"add_custom_{gid}", use_container_width=True):
                        _push_history()
                        if arity == 1:
                            st.session_state.gates.append(
                                Gate(
                                    name=f"custom:{gid}",
                                    label=label,
                                    qubit=0,
                                    position=len(st.session_state.gates),
                                    description="Custom unitary",
                                )
                            )
                        else:
                            st.session_state.gates.append(
                                Gate(
                                    name=f"custom:{gid}",
                                    label=label,
                                    qubit=1,
                                    control=0,
                                    position=len(st.session_state.gates),
                                    description="Custom 2-qubit unitary",
                                )
                            )
                        _push_history()
                        st.rerun()
                with row2:
                    if st.button("Delete", key=f"del_custom_{gid}", use_container_width=True):
                        del st.session_state.custom_gates[gid]
                        st.rerun()
        else:
            st.caption("No custom gates yet.")

        # Circuit composer (blocks/pipeline)
        st.divider()
        st.subheader("Circuit Composer")

        with st.expander("Save Current Circuit as Block", expanded=False):
            block_name = st.text_input("Block name", value="My Block")
            if st.button("Save Block", use_container_width=True):
                bid = uuid4().hex[:10]
                st.session_state.circuit_blocks[bid] = {
                    "id": bid,
                    "name": block_name.strip() or "Block",
                    "num_qubits": int(st.session_state.num_qubits),
                    "gates": [_gate_to_dict(g) for g in st.session_state.gates],
                    "custom_gates": dict(st.session_state.custom_gates),
                }
                st.success("Block saved.")

        if st.session_state.circuit_blocks:
            for bid, b in list(st.session_state.circuit_blocks.items()):
                title = f"{b.get('name','Block')} ({b.get('num_qubits')}q, {len(b.get('gates',[]))} gates)"
                with st.expander(title, expanded=False):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        if st.button("Load", key=f"blk_load_{bid}", use_container_width=True):
                            _push_history()
                            st.session_state.num_qubits = int(b.get("num_qubits", st.session_state.num_qubits))
                            st.session_state.custom_gates = dict(b.get("custom_gates", {}))
                            st.session_state.gates = [_gate_from_dict(d) for d in b.get("gates", [])]
                            st.session_state.simulation_result = None
                            st.session_state.undo_stack = [_snapshot_gates()]
                            st.session_state.redo_stack = []
                            _push_history()
                            st.rerun()
                    with c2:
                        if st.button("Add to Pipeline", key=f"blk_pipe_{bid}", use_container_width=True):
                            st.session_state.pipeline.append(bid)
                            st.rerun()
                    with c3:
                        if st.button("Delete", key=f"blk_del_{bid}", use_container_width=True):
                            st.session_state.circuit_blocks.pop(bid, None)
                            st.session_state.pipeline = [x for x in st.session_state.pipeline if x != bid]
                            st.rerun()
        else:
            st.caption("No blocks saved yet.")

        st.subheader("Pipeline (End-to-End)")
        if st.session_state.pipeline:
            for idx, bid in enumerate(list(st.session_state.pipeline)):
                b = st.session_state.circuit_blocks.get(bid, {})
                label = b.get("name", bid)
                r1, r2, r3, r4 = st.columns([6, 1, 1, 1])
                with r1:
                    st.write(f"{idx+1}. {label}")
                with r2:
                    if st.button("Up", key=f"pipe_up_{idx}", use_container_width=True, disabled=(idx == 0)):
                        st.session_state.pipeline[idx - 1], st.session_state.pipeline[idx] = (
                            st.session_state.pipeline[idx],
                            st.session_state.pipeline[idx - 1],
                        )
                        st.rerun()
                with r3:
                    if st.button("Down", key=f"pipe_dn_{idx}", use_container_width=True, disabled=(idx == len(st.session_state.pipeline) - 1)):
                        st.session_state.pipeline[idx + 1], st.session_state.pipeline[idx] = (
                            st.session_state.pipeline[idx],
                            st.session_state.pipeline[idx + 1],
                        )
                        st.rerun()
                with r4:
                    if st.button("X", key=f"pipe_rm_{idx}", use_container_width=True):
                        st.session_state.pipeline.pop(idx)
                        st.rerun()

            if st.button("Load Composed Circuit into Editor", use_container_width=True):
                try:
                    composed = _compose_blocks(st.session_state.pipeline)
                    _push_history()
                    st.session_state.num_qubits = int(composed["num_qubits"])
                    st.session_state.custom_gates = dict(composed["custom_gates"])
                    st.session_state.gates = composed["gates"]
                    st.session_state.simulation_result = None
                    st.session_state.undo_stack = [_snapshot_gates()]
                    st.session_state.redo_stack = []
                    _push_history()
                    st.success("Composed circuit loaded into editor.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Compose failed: {e}")

            if st.button("Clear Pipeline", use_container_width=True):
                st.session_state.pipeline = []
                st.rerun()
        else:
            st.caption("Add blocks to pipeline to form an end-to-end circuit.")
        
        st.divider()

        # Circuit settings
        st.header("Settings")
        project_name = st.text_input("Circuit name", value=st.session_state.get("project_name", "Untitled Circuit"))
        st.session_state.project_name = project_name
        num_qubits = st.slider("Number of Qubits", 1, 10, st.session_state.num_qubits)
        st.session_state.num_qubits = num_qubits
        
        shots = st.slider("Shots", 1, 8192, 1024)
        sample_measurements = st.checkbox("Sample measurements", value=True)
        enable_timeline = st.checkbox("Enable timeline (step-by-step)", value=False)
        optimize_circuit = st.checkbox("Optimize circuit (transpile)", value=False)
        opt_level = st.select_slider("Optimization level", options=[0, 1, 2, 3], value=1, disabled=(not optimize_circuit))
        use_pipeline = st.checkbox("Use pipeline (end-to-end blocks)", value=False, disabled=(len(st.session_state.pipeline) == 0))
        noise_model = st.selectbox(
            "Noise Model",
            options=[e.value for e in NoiseModelType],
            index=0,
            key="noise_select"
        )
        
        st.divider()

        st.header("Import / Export")
        col_ie1, col_ie2 = st.columns(2)
        with col_ie1:
            st.download_button(
                "Download Project JSON",
                data=_export_project_json(),
                file_name=f"{st.session_state.project_name.replace(' ', '_')}.qanvas.json",
                mime="application/json",
                use_container_width=True,
            )
        with col_ie2:
            uploaded = st.file_uploader("Upload Project JSON", type=["json"], label_visibility="collapsed")
            if uploaded is not None:
                try:
                    _push_history()
                    _import_project_json(uploaded.getvalue().decode("utf-8"))
                    _push_history()
                    st.success("Project loaded.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not load project: {e}")

        with st.expander("Import OpenQASM", expanded=False):
            qasm_text = st.text_area("Paste OpenQASM 2/3", value="", height=140, placeholder="OPENQASM 2.0; ...")
            if st.button("Import QASM into Editor", use_container_width=True):
                try:
                    qc = None
                    try:
                        from qiskit import qasm2 as _qasm2  # type: ignore

                        qc = _qasm2.loads(qasm_text)
                    except Exception:
                        qc = None
                    if qc is None:
                        try:
                            from qiskit import qasm3 as _qasm3  # type: ignore

                            qc = _qasm3.loads(qasm_text)
                        except Exception:
                            qc = None
                    if qc is None:
                        raise ValueError("QASM import not supported by this Qiskit install, or invalid QASM.")

                    _push_history()
                    st.session_state.num_qubits = int(qc.num_qubits)
                    st.session_state.gates = _qiskit_circuit_to_gates(qc)
                    st.session_state.simulation_result = None
                    st.session_state.undo_stack = [_snapshot_gates()]
                    st.session_state.redo_stack = []
                    _push_history()
                    st.success("Imported into editor (best-effort).")
                    st.rerun()
                except Exception as e:
                    st.error(f"Import failed: {e}")
        
        # Actions
        st.header("Actions")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Run Simulation", use_container_width=True, type="primary"):
                with st.spinner("Running quantum simulation..."):
                    try:
                        start_time = time.time()

                        # Build circuit
                        qc_custom_backup = dict(st.session_state.custom_gates)
                        if use_pipeline and st.session_state.pipeline:
                            composed = _compose_blocks(st.session_state.pipeline)
                            num_qubits = int(composed["num_qubits"])
                            st.session_state.custom_gates = dict(composed["custom_gates"])
                            qc = QuantumEngine.build_circuit(num_qubits, composed["gates"])
                        else:
                            qc = QuantumEngine.build_circuit(num_qubits, st.session_state.gates)

                        # Simulate
                        statevector = QuantumEngine.simulate_statevector(qc)
                        probabilities = QuantumEngine.calculate_probabilities(statevector, num_qubits)
                        bloch_data = QuantumEngine.generate_bloch_data(statevector, num_qubits)
                        density = QuantumEngine.calculate_density_matrix(statevector)
                        entropies = QuantumEngine.single_qubit_entropies(statevector, num_qubits)

                        timeline = None
                        if enable_timeline:
                            timeline = QuantumEngine.simulate_timeline(qc, num_qubits=num_qubits, max_steps=200)

                        qc_opt = None
                        qasm_opt = None
                        qasm3_opt = None
                        try:
                            if optimize_circuit:
                                qc_opt = transpile(qc, optimization_level=int(opt_level))
                                qasm_opt = qc_opt.qasm() if hasattr(qc_opt, "qasm") else str(qc_opt)
                                try:
                                    from qiskit import qasm3 as _qasm3  # type: ignore

                                    qasm3_opt = _qasm3.dumps(qc_opt)
                                except Exception:
                                    qasm3_opt = None
                        except Exception as e:
                            st.warning(f"Optimization skipped: {e}")

                        # Noisy simulation
                        counts = None
                        if noise_model != 'ideal' or sample_measurements:
                            counts_result = QuantumEngine.simulate_with_noise(qc, noise_model, shots)
                            counts = counts_result['counts']

                        # Export QASM
                        qasm_str = qc.qasm() if hasattr(qc, 'qasm') else str(qc)
                        qasm3_str = None
                        try:
                            from qiskit import qasm3 as _qasm3  # type: ignore

                            qasm3_str = _qasm3.dumps(qc)
                        except Exception:
                            qasm3_str = None

                        elapsed = time.time() - start_time

                        sim_gate_count = len(st.session_state.gates)
                        if use_pipeline and st.session_state.pipeline:
                            try:
                                sim_gate_count = sum(
                                    len(st.session_state.circuit_blocks[bid].get("gates", []))
                                    for bid in st.session_state.pipeline
                                )
                            except Exception:
                                sim_gate_count = sim_gate_count

                        st.session_state.simulation_result = {
                            'circuit': qc,
                            'circuit_optimized': qc_opt,
                            'statevector': statevector,
                            'probabilities': probabilities,
                            'bloch_data': bloch_data,
                            'density': density,
                            'entropies': entropies,
                            'timeline': timeline,
                            'counts': counts,
                            'qasm': qasm_str,
                            'qasm3': qasm3_str,
                            'qasm_opt': qasm_opt,
                            'qasm3_opt': qasm3_opt,
                            'time': elapsed,
                            'num_gates': sim_gate_count,
                            'used_pipeline': bool(use_pipeline and st.session_state.pipeline),
                        }

                        # restore editor custom gates (simulation may have temporarily swapped them for pipeline)
                        st.session_state.custom_gates = qc_custom_backup

                        st.success(f"Simulation completed in {elapsed:.3f}s")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Simulation failed: {type(e).__name__}: {e}")
        
        with col2:
            if st.button("Clear Circuit", use_container_width=True):
                _push_history()
                st.session_state.gates = []
                st.session_state.simulation_result = None
                _push_history()
                st.rerun()
        
        # Preset circuits
        st.divider()
        st.subheader("Presets")
        
        if st.button("Create Bell State", use_container_width=True):
            _push_history()
            st.session_state.gates = [
                Gate(name='h', label='H', qubit=0, position=0, description='Hadamard'),
                Gate(name='cx', label='CNOT', qubit=1, control=0, position=1, description='CNOT'),
            ]
            st.session_state.num_qubits = 2
            st.session_state.simulation_result = None
            _push_history()
            st.rerun()
        
        if st.button("Create GHZ State", use_container_width=True):
            _push_history()
            st.session_state.gates = [
                Gate(name='h', label='H', qubit=0, position=0, description='Hadamard'),
                Gate(name='cx', label='CNOT', qubit=1, control=0, position=1, description='CNOT'),
                Gate(name='cx', label='CNOT', qubit=2, control=1, position=2, description='CNOT'),
            ]
            st.session_state.num_qubits = 3
            st.session_state.simulation_result = None
            _push_history()
            st.rerun()
        
        # Export
        st.divider()
        if st.session_state.simulation_result:
            qc_export = st.session_state.simulation_result["circuit"]
            st.download_button(
                label="Export QASM",
                data=st.session_state.simulation_result['qasm'],
                file_name=f"circuit_{int(time.time())}.qasm",
                mime="text/plain",
                use_container_width=True
            )

            png = circuit_png_bytes(qc_export, st.session_state.num_qubits)
            if png is not None:
                st.download_button(
                    label="Export Circuit PNG",
                    data=png,
                    file_name=f"circuit_{int(time.time())}.png",
                    mime="image/png",
                    use_container_width=True,
                )

            results_json = json.dumps(
                {
                    "num_qubits": st.session_state.num_qubits,
                    "probabilities": st.session_state.simulation_result.get("probabilities", {}),
                    "counts": st.session_state.simulation_result.get("counts"),
                    "time_s": st.session_state.simulation_result.get("time"),
                },
                indent=2,
                sort_keys=True,
            )
            st.download_button(
                label="Export Results JSON",
                data=results_json,
                file_name=f"results_{int(time.time())}.json",
                mime="application/json",
                use_container_width=True,
            )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Circuit Builder")

        with st.expander("Add Gate (Drop Menu)", expanded=(len(st.session_state.gates) == 0)):
            # Build a unified gate menu including custom gates.
            menu_items: List[Dict[str, Any]] = []
            for g in GATE_DEFINITIONS["single_qubit"]:
                menu_items.append({"key": g["name"], "label": f"{g['label']} (1q)", "arity": 1})
            for g in GATE_DEFINITIONS["multi_qubit"]:
                menu_items.append({"key": g["name"], "label": f"{g['label']} (2q)", "arity": 2})
            for g in GATE_DEFINITIONS["other"]:
                menu_items.append({"key": g["name"], "label": f"{g['label']} (op)", "arity": 1})
            for gid, cg in st.session_state.custom_gates.items():
                arity = int(cg.get("arity", 1))
                label = str(cg.get("label", "U"))
                menu_items.append({"key": f"custom:{gid}", "label": f"{label} (custom {arity}q)", "arity": arity})

            menu_labels = [m["label"] for m in menu_items]
            chosen = st.selectbox("Gate", options=list(range(len(menu_items))), format_func=lambda i: menu_labels[i])
            chosen_item = menu_items[int(chosen)]

            add_col1, add_col2, add_col3 = st.columns(3)
            with add_col1:
                tgt = st.number_input("Target qubit", min_value=0, max_value=num_qubits - 1, value=0)
            with add_col2:
                pos = st.number_input("Insert position", min_value=0, value=len(st.session_state.gates))
            with add_col3:
                ctrl = None
                controls: Optional[List[int]] = None
                if chosen_item["key"] in {"ccx", "ccz"}:
                    c1 = st.number_input("Control 1", min_value=0, max_value=num_qubits - 1, value=0)
                    c2 = st.number_input("Control 2", min_value=0, max_value=num_qubits - 1, value=min(1, num_qubits - 1))
                    controls = [int(c1), int(c2)]
                elif chosen_item["key"] in {"mcx", "mcz"}:
                    opts = [i for i in range(num_qubits) if i != int(tgt)]
                    picked = st.multiselect("Controls (2+)", options=opts, default=opts[:2])
                    controls = [int(x) for x in picked]
                elif int(chosen_item["arity"]) == 2:
                    ctrl = st.number_input("Control/Other qubit", min_value=0, max_value=num_qubits - 1, value=0)

            theta = None
            params: Optional[List[float]] = None
            k = str(chosen_item["key"])
            if k in {"rx", "ry", "rz", "crx", "cry", "crz", "rxx", "ryy", "rzz", "xx", "yy", "zz"}:
                theta = st.number_input("Angle (radians)", value=float(np.pi / 2), format="%.6f")
            elif k in {"p", "u1", "cp"}:
                theta = st.number_input("Lambda (radians)", value=float(np.pi / 4), format="%.6f")
            elif k == "u2":
                phi = st.number_input("Phi (radians)", value=float(0.0), format="%.6f")
                lam = st.number_input("Lambda (radians)", value=float(np.pi), format="%.6f")
                params = [float(phi), float(lam)]
            elif k == "u3":
                th = st.number_input("Theta (radians)", value=float(np.pi / 2), format="%.6f")
                phi = st.number_input("Phi (radians)", value=float(0.0), format="%.6f")
                lam = st.number_input("Lambda (radians)", value=float(np.pi), format="%.6f")
                params = [float(th), float(phi), float(lam)]
            elif k == "global_phase":
                theta = st.number_input("Global phase (radians)", value=float(0.0), format="%.6f")

            if st.button("Add Gate", use_container_width=True):
                try:
                    _push_history()
                    ng = Gate(
                        name=str(chosen_item["key"]),
                        label=str(chosen_item["label"].split(" (", 1)[0]),
                        qubit=int(tgt),
                        position=int(pos),
                        control=(None if ctrl is None else int(ctrl)),
                        controls=controls,
                        param=theta,
                        params=params,
                        description="Added via drop menu",
                    )
                    if ng.name in {"cx", "cz", "cy", "ch", "cs", "ct", "swap", "iswap", "sqrt_swap", "cp", "crx", "cry", "crz", "rxx", "ryy", "rzz", "xx", "yy", "zz", "ecr"} or ng.name.startswith("custom:"):
                        # validate control != target for 2q ops (custom arity=2 also uses control field)
                        if ctrl is None and not controls:
                            raise ValueError("This gate requires a Control/Other qubit.")
                        if ctrl is not None and int(ctrl) == int(tgt):
                            raise ValueError("Control/Other qubit must be different from target qubit.")
                        if controls and int(tgt) in controls:
                            raise ValueError("Controls must be different from target qubit.")
                    if ng.name in {"ccx", "ccz", "mcx", "mcz"}:
                        if not ng.controls or len(ng.controls) < (2 if ng.name in {"ccx", "ccz"} else 2):
                            raise ValueError("Select at least 2 control qubits.")
                        if len(set(ng.controls)) != len(ng.controls):
                            raise ValueError("Duplicate control qubits selected.")
                        if int(tgt) in ng.controls:
                            raise ValueError("Controls must be different from target qubit.")
                    _insert_gate_at(ng, int(pos))
                    _push_history()
                    st.session_state.simulation_result = None
                    st.success("Gate added.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not add gate: {e}")
        
        if st.session_state.gates:
            st.write(f"**{len(st.session_state.gates)} gates added**")
            
            # Display mode toggle
            display_mode = st.radio(
                "Circuit View",
                options=['diagram', 'text'],
                horizontal=True,
                key="display_mode_radio",
                index=0
            )
            
            # Build circuit for display
            qc_display = QuantumEngine.build_circuit(num_qubits, st.session_state.gates)
            
            # Render circuit
            if display_mode == 'diagram':
                st.subheader("Circuit Diagram")
                st.markdown('<div class="circuit-surface">', unsafe_allow_html=True)
                render_circuit_diagram(qc_display, num_qubits)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown('<div class="circuit-surface">', unsafe_allow_html=True)
                render_circuit_text(qc_display, num_qubits)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Gate editor
            st.subheader("Gate Configuration")
            edited_gates = []
            for i, gate in enumerate(st.session_state.gates):
                with st.expander(f"Gate {i+1}: {gate.label} on qubit {gate.qubit}", expanded=False):
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        new_qubit = st.number_input(
                            "Qubit",
                            min_value=0,
                            max_value=num_qubits-1,
                            value=gate.qubit,
                            key=f"qubit_{i}"
                        )
                    with col_b:
                        new_position = st.number_input(
                            "Position",
                            min_value=0,
                            value=gate.position,
                            key=f"pos_{i}"
                        )
                    with col_c:
                        if st.button("Remove", key=f"remove_{i}"):
                            _push_history()
                            st.session_state.gates.pop(i)
                            _push_history()
                            st.rerun()

                    new_control = gate.control
                    new_controls = gate.controls[:] if gate.controls else None
                    new_param = gate.param
                    new_params = gate.params[:] if gate.params else None

                    single_control_2q = {
                        "cx",
                        "cz",
                        "cy",
                        "ch",
                        "cs",
                        "ct",
                        "swap",
                        "iswap",
                        "sqrt_swap",
                        "cp",
                        "crx",
                        "cry",
                        "crz",
                        "rxx",
                        "ryy",
                        "rzz",
                        "xx",
                        "yy",
                        "zz",
                        "ecr",
                    }

                    if gate.name in single_control_2q or (gate.name.startswith("custom:") and gate.control is not None):
                        new_control = st.number_input(
                            "Control/Other Qubit",
                            min_value=0,
                            max_value=num_qubits - 1,
                            value=(gate.control if gate.control is not None else 0),
                            key=f"ctrl_{i}",
                        )

                    if gate.name in {"ccx", "ccz"}:
                        c1 = st.number_input(
                            "Control 1",
                            min_value=0,
                            max_value=num_qubits - 1,
                            value=(gate.controls[0] if gate.controls and len(gate.controls) > 0 else 0),
                            key=f"c1_{i}",
                        )
                        c2 = st.number_input(
                            "Control 2",
                            min_value=0,
                            max_value=num_qubits - 1,
                            value=(gate.controls[1] if gate.controls and len(gate.controls) > 1 else min(1, num_qubits - 1)),
                            key=f"c2_{i}",
                        )
                        new_controls = [int(c1), int(c2)]

                    if gate.name in {"mcx", "mcz"}:
                        opts = [qidx for qidx in range(num_qubits) if qidx != int(new_qubit)]
                        picked = st.multiselect(
                            "Controls (2+)",
                            options=opts,
                            default=(gate.controls if gate.controls else opts[:2]),
                            key=f"mctrl_{i}",
                        )
                        new_controls = [int(x) for x in picked]

                    if gate.name in {"rx", "ry", "rz", "crx", "cry", "crz", "rxx", "ryy", "rzz", "xx", "yy", "zz"}:
                        new_param = st.number_input(
                            "Angle (radians)",
                            value=(gate.param if gate.param is not None else float(np.pi / 2)),
                            key=f"param_{i}",
                            format="%.6f",
                        )

                    if gate.name in {"p", "u1", "cp"}:
                        new_param = st.number_input(
                            "Lambda (radians)",
                            value=(gate.param if gate.param is not None else float(np.pi / 4)),
                            key=f"lam_{i}",
                            format="%.6f",
                        )

                    if gate.name == "global_phase":
                        new_param = st.number_input(
                            "Global phase (radians)",
                            value=(gate.param if gate.param is not None else float(0.0)),
                            key=f"gp_{i}",
                            format="%.6f",
                        )

                    if gate.name == "u2":
                        phi = st.number_input(
                            "Phi (radians)",
                            value=(gate.params[0] if gate.params and len(gate.params) > 0 else float(0.0)),
                            key=f"u2_phi_{i}",
                            format="%.6f",
                        )
                        lam = st.number_input(
                            "Lambda (radians)",
                            value=(gate.params[1] if gate.params and len(gate.params) > 1 else float(np.pi)),
                            key=f"u2_lam_{i}",
                            format="%.6f",
                        )
                        new_params = [float(phi), float(lam)]

                    if gate.name == "u3":
                        th = st.number_input(
                            "Theta (radians)",
                            value=(gate.params[0] if gate.params and len(gate.params) > 0 else float(np.pi / 2)),
                            key=f"u3_th_{i}",
                            format="%.6f",
                        )
                        phi = st.number_input(
                            "Phi (radians)",
                            value=(gate.params[1] if gate.params and len(gate.params) > 1 else float(0.0)),
                            key=f"u3_phi_{i}",
                            format="%.6f",
                        )
                        lam = st.number_input(
                            "Lambda (radians)",
                            value=(gate.params[2] if gate.params and len(gate.params) > 2 else float(np.pi)),
                            key=f"u3_lam_{i}",
                            format="%.6f",
                        )
                        new_params = [float(th), float(phi), float(lam)]
                    
                    edited_gate = Gate(
                        name=gate.name,
                        label=gate.label,
                        qubit=new_qubit,
                        position=new_position,
                        control=new_control,
                        controls=new_controls,
                        param=new_param,
                        params=new_params,
                        description=gate.description,
                        id=gate.id
                    )
                    edited_gates.append(edited_gate)
            
            if _snapshot_gates() != [_gate_to_dict(g) for g in edited_gates]:
                _push_history()
                st.session_state.gates = edited_gates
                _push_history()
            
        else:
            st.info("Add gates from the sidebar to build your circuit.")
    
    with col2:
        st.header("Visualizations")
        
        if st.session_state.simulation_result:
            result = st.session_state.simulation_result

            tab_bloch, tab_probs, tab_amps, tab_density, tab_timeline, tab_code = st.tabs(
                ["Bloch", "Probabilities", "Amplitudes", "Density", "Timeline", "Code"]
            )

            with tab_bloch:
                st.subheader("Bloch Sphere")
                bloch_fig = plot_bloch_sphere(result.get("bloch_data", []))
                st.plotly_chart(bloch_fig, use_container_width=True)

            with tab_probs:
                st.subheader("Statevector Probabilities")
                sv_fig = plot_statevector(result.get("probabilities", {}))
                st.plotly_chart(sv_fig, use_container_width=True)

                if result.get("counts"):
                    st.subheader("Measurements")
                    meas_fig = plot_measurements(result["counts"])
                    st.plotly_chart(meas_fig, use_container_width=True)

            with tab_amps:
                st.subheader("Complex Amplitudes")
                amps_fig = plot_statevector_complex(result["statevector"], num_qubits)
                st.plotly_chart(amps_fig, use_container_width=True)

                st.subheader("Amplitude Table")
                st.dataframe(
                    statevector_table(result["statevector"], num_qubits),
                    use_container_width=True,
                    hide_index=True,
                )

            with tab_density:
                if result.get("density") is not None:
                    st.subheader("Density Matrix")
                    dm_fig = plot_density_matrix_heatmap(result["density"])
                    st.plotly_chart(dm_fig, use_container_width=True)

                st.subheader("Circuit Metrics")
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Qubits", num_qubits)
                col_m2.metric("Gates", result['num_gates'])
                col_m3.metric("Time", f"{result['time']:.3f}s")

                try:
                    qc_m = result["circuit"]
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Depth", int(qc_m.depth()))
                    col_b.metric("Size", int(qc_m.size()))
                    col_c.metric("Width", int(qc_m.width()))
                    st.caption("Gate counts: " + ", ".join([f"{k}={v}" for k, v in qc_m.count_ops().items()]))
                except Exception:
                    pass

                if result.get("entropies") is not None:
                    st.caption(
                        "Single-qubit entropies (0=pure, 1=maximally mixed): "
                        + ", ".join([f"q{i}={v:.3f}" for i, v in enumerate(result['entropies'])])
                    )

            with tab_code:
                st.subheader("Circuit Code")
                q2, q3, q2o, q3o = st.tabs(["OpenQASM 2", "OpenQASM 3", "Optimized QASM 2", "Optimized QASM 3"])
                with q2:
                    st.code(result.get("qasm", ""), language="qasm")
                    st.download_button(
                        "Download OpenQASM 2",
                        data=result.get("qasm", ""),
                        file_name=f"circuit_{int(time.time())}.qasm",
                        mime="text/plain",
                        use_container_width=True,
                    )
                with q3:
                    if result.get("qasm3"):
                        st.code(result["qasm3"], language="qasm")
                        st.download_button(
                            "Download OpenQASM 3",
                            data=result["qasm3"],
                            file_name=f"circuit_{int(time.time())}.qasm3",
                            mime="text/plain",
                            use_container_width=True,
                        )
                    else:
                        st.info("OpenQASM 3 export not available in this Qiskit install.")
                with q2o:
                    if result.get("qasm_opt"):
                        st.code(result["qasm_opt"], language="qasm")
                        st.download_button(
                            "Download Optimized OpenQASM 2",
                            data=result["qasm_opt"],
                            file_name=f"circuit_optimized_{int(time.time())}.qasm",
                            mime="text/plain",
                            use_container_width=True,
                        )
                    else:
                        st.info("Enable 'Optimize circuit (transpile)' in Settings and re-run.")
                with q3o:
                    if result.get("qasm3_opt"):
                        st.code(result["qasm3_opt"], language="qasm")
                        st.download_button(
                            "Download Optimized OpenQASM 3",
                            data=result["qasm3_opt"],
                            file_name=f"circuit_optimized_{int(time.time())}.qasm3",
                            mime="text/plain",
                            use_container_width=True,
                        )
                    else:
                        st.info("Enable 'Optimize circuit (transpile)' in Settings and re-run.")

            with tab_timeline:
                tl = result.get("timeline")
                if not tl:
                    st.info("Enable 'Enable timeline (step-by-step)' in Settings and re-run.")
                else:
                    ops = tl.get("ops", [])
                    states = tl.get("states", [])
                    if len(states) <= 1:
                        st.info("Timeline has no steps (measurements or unsupported ops).")
                    else:
                        step = st.slider("Step", 0, len(states) - 1, 0)
                        if step == 0:
                            st.caption("Initial state |0...0>")
                        else:
                            op = ops[step - 1] if step - 1 < len(ops) else {"name": "op"}
                            st.caption(f"After step {step}: {op.get('label', op.get('name'))} on {op.get('qubits')}")
                        sv_step = states[step]
                        probs_step = QuantumEngine.calculate_probabilities(sv_step, num_qubits)
                        bloch_step = QuantumEngine.generate_bloch_data(sv_step, num_qubits)
                        st.plotly_chart(plot_statevector(probs_step), use_container_width=True)
                        st.plotly_chart(plot_bloch_sphere(bloch_step), use_container_width=True)
        else:
            st.info("Run simulation to see visualizations.")
    
    # Footer
    st.divider()
    st.markdown("""
        <div style='text-align: center; color: #94a3b8; padding: 1rem;'>
            <p>Qanvas Studio | Built with Streamlit + Qiskit</p>
            <p>Made By Sourish Dey</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
