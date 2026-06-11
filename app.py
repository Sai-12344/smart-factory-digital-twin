import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import random
import math
import time

st.set_page_config(page_title="Smart Factory Digital Twin", page_icon="🏭", layout="wide")

class FailureCurve:
    def __init__(self, mode="hybrid"):
        self.mode = mode
    def compute_wear(self, cycle, load, temperature):
        base = (1 - math.exp(-0.004 * cycle))
        shock = random.uniform(0, 0.05) if random.random() < 0.02 else 0
        temp_factor = 1 + max(0, temperature - 55) * 0.012
        return min(base * temp_factor + shock, 1.0)
    def remaining_life(self, wear):
        if wear >= 1: return 0
        return int(200 * (1 - wear) ** 1.5)

class MachinePhysics:
    def __init__(self, machine_id):
        self.machine_id = machine_id
        self.curve = FailureCurve()
        self.cycle = 0
        self.load = random.uniform(0.3, 0.9)
        self.temperature = 40
        self.vibration = 0.02
        self.wear = 0
        self.health = 1.0
        self.ambient_temp = 27
    def apply_action(self, action):
        if action == "run": self.load = min(1.0, self.load + random.uniform(-0.05, 0.1))
        elif action == "cool": self.temperature -= 4; self.load *= 0.9
        elif action == "idle": self.load *= 0.7; self.temperature -= 2
        elif action == "maintain": self.wear *= 0.5; self.temperature -= 6; self.vibration *= 0.7; self.health = min(1.0, self.health + 0.4)
    def simulate_step(self, action="run"):
        self.cycle += 1
        self.apply_action(action)
        self.wear = self.curve.compute_wear(self.cycle, self.load, self.temperature)
        self.temperature += self.load * 1.5 + self.wear * 1.2 - (self.temperature - self.ambient_temp) * 0.03
        self.vibration += self.wear * 0.008 + random.uniform(-0.002, 0.003)
        self.health = max(0, 1 - self.wear)
        fp = min(1.0, self.wear*0.55 + max(0,(self.temperature-65)*0.015) + max(0,(self.vibration-0.05)*0.2))
        return {"id":self.machine_id,"cycle":self.cycle,"temp":round(self.temperature,2),"vibration":round(self.vibration,3),"load":round(self.load,3),"wear":round(self.wear,3),"health":round(self.health,3),"failure_prob":round(fp,3),"RUL":self.curve.remaining_life(self.wear)}

class PPOAgent:
    def __init__(self, state_dim=5, action_dim=4):
        self.W1 = np.random.randn(state_dim, 64) * 0.1
        self.W2 = np.random.randn(64, action_dim) * 0.1
    def act(self, state):
        h = np.tanh(state @ self.W1)
        logits = h @ self.W2
        logits -= logits.max()
        probs = np.exp(logits) / np.exp(logits).sum()
        return np.random.choice(len(probs), p=probs), probs

class DroneInspector:
    def detect_hotspots(self, heatmap):
        threshold = np.percentile(heatmap, 90)
        return [{"x":i,"y":j,"intensity":float(heatmap[i,j])} for i in range(heatmap.shape[0]) for j in range(heatmap.shape[1]) if heatmap[i,j] >= threshold]

if "factory" not in st.session_state:
    st.session_state.factory = [MachinePhysics(i) for i in range(10)]
    st.session_state.agent = PPOAgent()
    st.session_state.drone = DroneInspector()
    st.session_state.action_map = ["run","cool","idle","maintain"]
    st.session_state.history = []
    st.session_state.step = 0

factory = st.session_state.factory
agent = st.session_state.agent
drone = st.session_state.drone
action_map = st.session_state.action_map

machine_states = [m.simulate_step("run") for m in factory]
s0 = np.array([machine_states[0]["temp"],machine_states[0]["vibration"],machine_states[0]["load"],machine_states[0]["wear"],machine_states[0]["failure_prob"]],dtype=np.float32)
action_idx, policy = agent.act(s0)
rl_action = action_map[action_idx]
machine_states[0] = factory[0].simulate_step(rl_action)
gnn_risk = float(np.mean([m["failure_prob"] for m in machine_states]))
heatmap = np.random.uniform(30, 90, (6, 6))
hotspots = drone.detect_hotspots(heatmap)
st.session_state.step += 1
st.session_state.history.append(gnn_risk)
if len(st.session_state.history) > 60: st.session_state.history.pop(0)

st.title("🏭 Smart Factory Digital Twin")
st.caption(f"Step {st.session_state.step} — AI + Digital Twin + GNN + RL + Drone Inspection")

c1,c2,c3,c4 = st.columns(4)
c1.metric("GNN Risk Score", f"{gnn_risk:.3f}")
c2.metric("RL Action (M0)", rl_action)
c3.metric("Hotspots", len(hotspots))
c4.metric("Avg Failure Prob", f"{np.mean([m['failure_prob'] for m in machine_states]):.1%}")

st.divider()
st.subheader("⚡ Live Machine Status")
df = pd.DataFrame(machine_states)
df.columns = ["ID","Cycle","Temp °C","Vibration","Load","Wear","Health","Failure Prob","RUL"]
st.dataframe(df.style.background_gradient(subset=["Failure Prob","Wear"],cmap="RdYlGn_r").background_gradient(subset=["Health"],cmap="RdYlGn"),use_container_width=True)

st.divider()
col1,col2 = st.columns(2)
with col1:
    st.subheader("🔥 Drone Thermal Heatmap")
    fig_heat = px.imshow(heatmap,color_continuous_scale="hot",labels={"color":"Temp °C"},title=f"{len(hotspots)} hotspots detected")
    fig_heat.update_layout(margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig_heat,use_container_width=True)
with col2:
    st.subheader("🤖 PPO Action Probabilities")
    fig_bar = go.Figure(go.Bar(x=action_map,y=policy,marker_color=["#378ADD","#1D9E75","#888780","#BA7517"],text=[f"{p:.1%}" for p in policy],textposition="outside"))
    fig_bar.update_layout(yaxis=dict(range=[0,1],title="Probability"),xaxis_title="Action",margin=dict(l=0,r=0,t=10,b=0),height=300)
    st.plotly_chart(fig_bar,use_container_width=True)

st.divider()
st.subheader("📈 Factory Risk Over Time")
fig_line = go.Figure(go.Scatter(y=st.session_state.history,mode="lines",fill="tozeroy",line=dict(color="#E24B4A",width=2),fillcolor="rgba(226,75,74,0.1)"))
fig_line.update_layout(yaxis=dict(range=[0,1],title="GNN Risk"),xaxis_title="Steps",margin=dict(l=0,r=0,t=10,b=0),height=200)
st.plotly_chart(fig_line,use_container_width=True)

time.sleep(1)
st.rerun()
