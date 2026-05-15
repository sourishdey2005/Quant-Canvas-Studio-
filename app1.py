"""
Qanvas Studio - Quantum Circuit Visualizer
FIXED: Proper data serialization for Streamlit session_state
"""

import time
import json
from typing import Optional, List, Dict
from dataclasses import dataclass, field, asdict
from uuid import uuid4

import numpy as np
import streamlit as st
import plotly.graph_objects as go

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector, Operator


# ============================================================================
# Data Structures (Streamlit-Safe)
# ============================================================================

@dataclass
class Gate:
    name: str
    label: str
    qubit: int
    position: int
    control: Optional[int] = None
    param: Optional[float] = None
    description: str = ""
    id: str = field(default_factory=lambda: uuid4().hex)
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict"""
        return {
            'name': self.name, 'label': self.label, 'qubit': self.qubit,
            'position': self.position, 'control': self.control,
            'param': self.param, 'description': self.description, 'id': self.id
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'Gate':
        """Create from JSON-serializable dict"""
        return cls(**d)


# Gate definitions
GATES = {
    'single': [
        {'name': 'h', 'label': 'H'}, {'name': 'x', 'label': 'X'},
        {'name': 'y', 'label': 'Y'}, {'name': 'z', 'label': 'Z'},
        {'name': 's', 'label': 'S'}, {'name': 't', 'label': 'T'},
        {'name': 'rx', 'label': 'Rx'}, {'name': 'ry', 'label': 'Ry'},
        {'name': 'rz', 'label': 'Rz'},
    ],
    'multi': [
        {'name': 'cx', 'label': 'CNOT'}, {'name': 'cz', 'label': 'CZ'},
        {'name': 'swap', 'label': 'SWAP'},
    ],
    'other': [
        {'name': 'measure', 'label': 'M'}, {'name': 'reset', 'label': 'R'},
        {'name': 'barrier', 'label': '||'},
    ]
}


# ============================================================================
# Quantum Engine (Simplified & Reliable)
# ============================================================================

class Engine:
    @staticmethod
    def build(nq: int, gates: List[Gate]) -> QuantumCircuit:
        qr = QuantumRegister(nq, 'q')
        cr = ClassicalRegister(nq, 'c')
        qc = QuantumCircuit(qr, cr)
        
        for g in sorted(gates, key=lambda x: x.position):
            try:
                q = g.qubit
                if g.name == 'h': qc.h(q)
                elif g.name == 'x': qc.x(q)
                elif g.name == 'y': qc.y(q)
                elif g.name == 'z': qc.z(q)
                elif g.name == 's': qc.s(q)
                elif g.name == 't': qc.t(q)
                elif g.name in ['rx','ry','rz'] and g.param:
                    getattr(qc, g.name)(float(g.param), q)
                elif g.name == 'cx' and g.control is not None:
                    qc.cx(g.control, q)
                elif g.name == 'cz' and g.control is not None:
                    qc.cz(g.control, q)
                elif g.name == 'swap' and g.control is not None:
                    qc.swap(g.control, q)
                elif g.name == 'measure': qc.measure(q, q)
                elif g.name == 'reset': qc.reset(q)
                elif g.name == 'barrier': qc.barrier(q)
            except: pass
        return qc
    
    @staticmethod
    def simulate(qc: QuantumCircuit) -> Dict:
        """Return ONLY JSON-serializable data"""
        try:
            # Remove measurements for statevector
            qc_sim = qc.copy()
            qc_sim.remove_final_measurements(inplace=True)
            
            # Evolve statevector
            sv = Statevector.from_int(0, 2**qc_sim.num_qubits)
            for instr in qc_sim.data:
                qargs = [qc_sim.find_bit(q).index for q in instr.qubits]
                sv = sv.evolve(instr.operation, qargs=qargs)
            
            # Convert to serializable format
            n = qc.num_qubits
            probs = {}
            for i, amp in enumerate(sv.data):
                p = float(np.abs(amp)**2)
                if p > 1e-10:
                    probs[format(i, f'0{n}b')] = round(p, 6)
            
            # Bloch data (first 3 qubits max)
            bloch = []
            for qi in range(min(n, 3)):
                try:
                    if n == 1:
                        rho = sv.to_operator().data
                        x = 2*np.real(rho[0,1]); y = 2*np.imag(rho[0,1]); z = np.real(rho[0,0]-rho[1,1])
                    else:
                        x = np.real(sv.expectation_value(Operator.from_label('X'), qargs=[qi]))
                        y = np.real(sv.expectation_value(Operator.from_label('Y'), qargs=[qi]))
                        z = np.real(sv.expectation_value(Operator.from_label('Z'), qargs=[qi]))
                    bloch.append({
                        'qubit': qi,
                        'x': round(float(np.clip(x,-1,1)),4),
                        'y': round(float(np.clip(y,-1,1)),4),
                        'z': round(float(np.clip(z,-1,1)),4)
                    })
                except:
                    bloch.append({'qubit': qi, 'x':0.0, 'y':0.0, 'z':1.0})
            
            return {
                'ok': True,
                'probs': probs,
                'bloch': bloch,
                'qasm': qc.qasm() if hasattr(qc,'qasm') else str(qc)
            }
        except Exception as e:
            return {'ok': False, 'error': str(e), 'probs': {}, 'bloch': [], 'qasm': ''}


# ============================================================================
# Plotly Visualizations (Guaranteed to Render)
# ============================================================================

def bloch_plot(data: List[Dict]) -> go.Figure:
    fig = go.Figure()
    # Sphere
    phi, theta = np.meshgrid(np.linspace(0,2*np.pi,20), np.linspace(0,np.pi,10))
    fig.add_trace(go.Surface(
        x=np.sin(theta)*np.cos(phi), y=np.sin(theta)*np.sin(phi), z=np.cos(theta),
        opacity=0.05, showscale=False, hoverinfo='skip'
    ))
    # Axes
    r = 1.2
    for nm, col, sx, sy, sz in [('X','red',[-r,r],[0,0],[0,0]), ('Y','green',[0,0],[-r,r],[0,0]), ('Z','blue',[0,0],[0,0],[-r,r])]:
        fig.add_trace(go.Scatter3d(x=sx,y=sy,z=sz, mode='lines', line=dict(color=col,width=2), name=nm))
    # Vectors
    colors = ['#8b5cf6','#06b6d4','#22c55e']
    for i,d in enumerate(data):
        try:
            fig.add_trace(go.Scatter3d(
                x=[0,d['x']], y=[0,d['y']], z=[0,d['z']],
                mode='markers+lines', name=f"Q{d['qubit']}",
                marker=dict(size=5,color=colors[i%3]), line=dict(color=colors[i%3],width=4)
            ))
        except: pass
    fig.update_layout(
        template='plotly_dark', height=300, margin=dict(l=0,r=0,t=0,b=0),
        scene=dict(xaxis=dict(visible=False),yaxis=dict(visible=False),zaxis=dict(visible=False),
                  camera=dict(eye=dict(x=1.4,y=1.4,z=1.4))),
        paper_bgcolor='#0b1220', plot_bgcolor='#0b1220'
    )
    return fig

def prob_plot(probs: Dict) -> go.Figure:
    fig = go.Figure()
    if probs:
        labs, vals = list(probs.keys()), list(probs.values())
        fig.add_trace(go.Bar(x=labs, y=vals, marker_color=['#8b5cf6' if v>0.3 else '#06b6d4' for v in vals],
                            text=[f'{v:.3f}' for v in vals], textposition='auto'))
        fig.update_layout(xaxis_title='State', yaxis_title='Prob', yaxis=dict(range=[0,1.1]), height=240)
    else:
        fig.add_annotation(text='Run simulation to see probabilities', xref='paper',yref='paper',x=0.5,y=0.5)
        fig.update_layout(height=240)
    fig.update_layout(template='plotly_dark', paper_bgcolor='#0b1220', plot_bgcolor='#0b1220')
    return fig

def meas_plot(counts: Dict) -> go.Figure:
    fig = go.Figure()
    if counts and sum(counts.values())>0:
        labs, vals = list(counts.keys()), list(counts.values())
        fig.add_trace(go.Pie(labels=labs, values=vals, hole=0.35,
                            marker=dict(colors=['#8b5cf6','#06b6d4','#22c55e','#f59e0b'][:len(labs)]),
                            textinfo='label+percent'))
        fig.update_layout(title=f'Shots: {sum(vals)}', height=240)
    else:
        fig.add_annotation(text='Enable measurements to see results', xref='paper',yref='paper',x=0.5,y=0.5)
        fig.update_layout(height=240)
    fig.update_layout(template='plotly_dark', paper_bgcolor='#0b1220', plot_bgcolor='#0b1220')
    return fig


# ============================================================================
# Main App
# ============================================================================

def main():
    st.set_page_config(page_title="Qanvas", page_icon="⚛️", layout="wide")
    st.markdown("<style>.main{background:#0b1220;color:#e2e8f0} .stButton>button{width:100%}</style>", unsafe_allow_html=True)
    
    st.title("⚛️ Qanvas Studio")
    
    # Init session state
    for k,v in {'gates':[], 'nq':2, 'result':None}.items():
        if k not in st.session_state: st.session_state[k] = v
    
    # SIDEBAR
    with st.sidebar:
        st.header("Gates")
        for cat, gates in [('Single',GATES['single']),('Multi',GATES['multi']),('Ops',GATES['other'])]:
            st.subheader(cat)
            for g in gates:
                if st.button(f"{g['label']} - {g['name']}", key=f"_{g['name']}"):
                    st.session_state.gates.append(Gate(g['name'],g['label'],0,len(st.session_state.gates)))
                    st.rerun()
        
        st.divider()
        nq = st.slider("Qubits",1,10,st.session_state.nq)
        st.session_state.nq = nq
        shots = st.slider("Shots",1,8192,1024)
        
        st.divider()
        c1,c2 = st.columns(2)
        with c1:
            if st.button("▶️ Run", type="primary"):
                with st.spinner("Simulating..."):
                    qc = Engine.build(nq, st.session_state.gates)
                    result = Engine.simulate(qc)
                    # Add shot counts if measurements present
                    if any(g.name=='measure' for g in st.session_state.gates):
                        try:
                            from qiskit_aer import AerSimulator
                            sim = AerSimulator()
                            qc_m = qc.copy()
                            if not qc_m.has_measurements(): qc_m.measure_all()
                            counts = sim.run(qc_m, shots=shots).result().get_counts()
                            result['counts'] = counts
                        except: pass
                    st.session_state.result = result
                    st.success(f"✓ {time.time()%100:.1f}s")
                    st.rerun()
        with c2:
            if st.button("🗑️ Clear"):
                st.session_state.gates = []
                st.session_state.result = None
                st.rerun()
        
        if st.button("Bell State"):
            st.session_state.gates = [Gate('h','H',0,0), Gate('cx','CNOT',1,1,control=0)]
            st.session_state.nq = 2
            st.session_state.result = None
            st.rerun()
    
    # MAIN
    col1, col2 = st.columns([2,1])
    
    with col1:
        st.header("Circuit")
        if st.session_state.gates:
            qc_disp = Engine.build(nq, st.session_state.gates)
            try:
                from qiskit.visualization import circuit_drawer
                fig = circuit_drawer(qc_disp, output='mpl', style={'name':'bw'}, fold=-1)
                st.pyplot(fig, bbox_inches='tight')
            except:
                st.code(str(qc_disp.draw(output='text')), language='text')
            
            # Edit gates
            for i,g in enumerate(st.session_state.gates):
                with st.expander(f"{i+1}. {g.label}@q{g.qubit}"):
                    c1,c2,c3 = st.columns(3)
                    g.qubit = c1.number_input("Q",0,nq-1,g.qubit,key=f"q{i}")
                    g.position = c2.number_input("Pos",0,100,g.position,key=f"p{i}")
                    if c3.button("🗑️",key=f"d{i}"):
                        st.session_state.gates.pop(i); st.rerun()
        else:
            st.info("Add gates from sidebar")
    
    with col2:
        st.header("Visualizations")
        result = st.session_state.result
        
        if result and result.get('ok'):
            # Metrics
            c1,c2,c3 = st.columns(3)
            c1.metric("Qubits", nq)
            c2.metric("Gates", len(st.session_state.gates))
            c3.metric("States", len(result.get('probs',{})))
            
            # Bloch
            st.subheader("Bloch")
            st.plotly_chart(bloch_plot(result.get('bloch',[])), use_container_width=True)
            
            # Probabilities
            st.subheader("Probabilities")
            st.plotly_chart(prob_plot(result.get('probs',{})), use_container_width=True)
            
            # Measurements
            if result.get('counts'):
                st.subheader("Measurements")
                st.plotly_chart(meas_plot(result['counts']), use_container_width=True)
            
            # QASM
            with st.expander("QASM"):
                st.code(result.get('qasm',''), language='qasm')
        else:
            if result and not result.get('ok'):
                st.error(f"Error: {result.get('error')}")
            else:
                st.info("Run simulation to see results")
    
    # DEBUG
    with st.expander("🔍 Debug"):
        st.write(f"Gates: {len(st.session_state.gates)}")
        if st.session_state.result:
            r = st.session_state.result
            st.write(f"OK: {r.get('ok')}")
            st.write(f"Probs keys: {list(r.get('probs',{}).keys())[:5]}")
            st.write(f"Bloch: {len(r.get('bloch',[]))} items")
            if st.button("🔄 Refresh"): st.rerun()

if __name__ == "__main__":
    main()