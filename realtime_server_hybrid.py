# realtime_server_hybrid.py
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import asyncio
import json
import time
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

import websockets

# ---------------------------
# TRY IMPORTS FROM YOUR PROJECT
# ---------------------------
try:
    from digital_twin.machine_physics import MachinePhysics
except Exception as e:
    raise ImportError("Cannot import MachinePhysics from digital_twin. Fix PYTHONPATH or file location.") from e

try:
    from gnn.factory_graph import FactoryGraph
except Exception:
    # If you don't have a GNN, we provide a tiny fallback that computes avg failure prob
    class FactoryGraph:
        def __init__(self, num_machines=10):
            self.num_machines = num_machines
        def to_graph_data(self, machine_states):
            # Build dummy adjacency & features
            X = np.array([[s.get("temp",0), s.get("vibration",0), s.get("load",0), s.get("wear",0), s.get("failure_prob",0)] for s in machine_states], dtype=np.float32)
            A = np.eye(len(machine_states))
            return A, X

try:
    from vision.drone_inspector import DroneInspector
except Exception:
    # Fallback simple drone inspector
    class DroneInspector:
        def detect_hotspots(self, heatmap):
            heat = np.array(heatmap)
            thr = heat.mean() + heat.std()
            hotspots = (heat > thr).sum()
            return int(hotspots)

# VLA and PPO paths
from vla.vla_agent import VLAAgent

# Path to your trained PPO (you said ppo_final.pth is inside vla/)
PPO_PATH = os.path.join("vla", "ppo_final.pth")

# ---------------------------
# ActorCritic fallback (same arch used in training)
# ---------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim=517, action_dim=4, hidden=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        self.policy = nn.Linear(hidden, action_dim)
        self.value = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.policy(h), self.value(h).squeeze(-1)

# ---------------------------
# Helper: try to use existing rl_agent.PPOAgent class if available
# ---------------------------
def try_load_custom_ppo(device):
    try:
        from rl_agent.ppo_agent import PPOAgent as OldPPOAgent
        # instantiate and try to load / adapt if class supports state_dict load
        # many custom PPO wrappers implement .load or .model
        agent = OldPPOAgent(state_dim=5, action_dim=4)  # placeholder init (may differ)
        # if there's a load method
        if hasattr(agent, "load_state_dict"):
            sd = torch.load(PPO_PATH, map_location=device)
            agent.load_state_dict(sd)
        elif hasattr(agent, "load"):
            agent.load(PPO_PATH)
        return agent, "custom"
    except Exception:
        return None, None

# ---------------------------
# Setup policy model (fallback)
# ---------------------------
def load_policy(device):
    # Try custom rl_agent.PPOAgent first
    agent, tag = try_load_custom_ppo(device)
    if agent is not None:
        print("[realtime_server] Loaded custom rl_agent.PPOAgent")
        return agent, tag

    # Otherwise load simple ActorCritic and load state dict
    policy = ActorCritic(obs_dim=517, action_dim=4, hidden=256).to(device)
    if not os.path.exists(PPO_PATH):
        raise FileNotFoundError(f"PPO checkpoint not found at {PPO_PATH}")
    sd = torch.load(PPO_PATH, map_location=device)

    # sd might be dict with 'model_state' or raw state_dict
    if isinstance(sd, dict) and "model_state" in sd:
        state_dict = sd["model_state"]
    else:
        # If saved as state_dict directly
        state_dict = sd

    # sometimes keys are prefixed; attempt to load forgivingly
    try:
        policy.load_state_dict(state_dict)
    except RuntimeError:
        # attempt to strip "net." or "module." prefixes
        new_sd = {}
        for k, v in state_dict.items():
            new_k = k
            if k.startswith("net."):
                new_k = k[len("net."):]
            if k.startswith("module."):
                new_k = new_k[len("module."):]
            new_sd[new_k] = v
        policy.load_state_dict(new_sd)
    policy.eval()
    print(f"[realtime_server] Loaded fallback ActorCritic from {PPO_PATH}")
    return policy, "actorcritic"

# ---------------------------
# RealTimeServer Hybrid
# ---------------------------
class RealTimeServerHybrid:
    def __init__(self, host="localhost", port=8765, device=None):
        self.host = host
        self.port = port
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        # old components
        self.factory = [MachinePhysics(i) for i in range(10)]
        self.graph = FactoryGraph(num_machines=10)
        self.drone = DroneInspector()

        # VLA (for building full obs)
        self.vla = VLAAgent()

        # policy (either custom wrapper or ActorCritic)
        self.policy, self.policy_type = load_policy(self.device)

        self.action_map = ["run", "cool", "idle", "maintain"]

    async def stream_data(self, websocket):
        step = 0
        print(f"[realtime_server] client connected, streaming...")
        while True:
            # 1) Simulate one step for all machines with a default 'run' action to get consistent states
            machine_states = [m.simulate_step("run") for m in self.factory]

            # 2) Compute GNN risk (fallback uses avg failure prob)
            try:
                A, X = self.graph.to_graph_data(machine_states)
                gnn_risk = float(np.mean(X[:, -1]))
            except Exception:
                gnn_risk = float(np.mean([m.get("failure_prob", 0.0) for m in machine_states]))

            # 3) Drone heatmap / hotspots
            heatmap = np.random.uniform(30, 90, (6, 6))
            hotspots = self.drone.detect_hotspots(heatmap) if hasattr(self.drone, "detect_hotspots") else 0

            # 4) Build observation for machine 0 (use VLA fused embedding + sensors)
            m0 = machine_states[0]
            # create a tiny synthetic heat image (6x6 -> 224x224) from the drone heatmap
            heat_img = (255 * (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)).astype("uint8")
            pil_img = Image.fromarray(heat_img).resize((224, 224)).convert("RGB")

            caption = f"Machine 0 temp {m0.get('temp',0):.1f}C vib {m0.get('vibration',0):.3f} wear {m0.get('wear',0):.2f}"
            try:
                v_enc = self.vla.vision.encode(pil_img)
                t_enc = self.vla.text.encode(caption)
                fused = self.vla.fusion.encode(v_enc, t_enc)  # expected shape (512,)
            except Exception as e:
                # If VLA methods fail, fallback to zeros (still keep sensors)
                print("[realtime_server] VLA encode failed:", e)
                fused = np.zeros(512, dtype=np.float32)

            sensors = np.array([
                m0.get("temp", 0.0),
                m0.get("vibration", 0.0),
                m0.get("load", 0.0),
                m0.get("wear", 0.0),
                m0.get("failure_prob", 0.0)
            ], dtype=np.float32)

            obs = np.concatenate([fused.astype(np.float32), sensors], axis=0)  # shape (517,)

            # 5) Query policy
            try:
                if self.policy_type == "custom":
                    # custom wrapper likely expects numpy state or its own method
                    # Try .act(obs) or .act(torch.tensor(...))
                    if hasattr(self.policy, "act"):
                        action_idx, policy_out = self.policy.act(obs)
                        # ensure action_idx integer
                        action_idx = int(action_idx)
                        probs = None
                    else:
                        # try calling like a torch model
                        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                        logits, _ = self.policy(obs_t)
                        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
                        action_idx = int(np.argmax(probs))
                else:
                    # actorcritic forward pass
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    with torch.no_grad():
                        logits, _ = self.policy(obs_t)
                        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
                        action_idx = int(np.argmax(probs))
            except Exception as e:
                print("[realtime_server] Policy inference failed:", e)
                # fallback action
                action_idx = 0
                probs = None

            action = self.action_map[action_idx]

            # 6) Apply action to machine 0 (simulate_step expects action string)
            machine_states[0] = self.factory[0].simulate_step(action)

            # 7) build and send packet
            packet = {
                "machines": machine_states,
                "drone_hotspots": int(hotspots),
                "gnn_risk": float(gnn_risk),
                "rl_action": action,
                "vla_caption": caption,
                "vla_scores": {self.action_map[i]: float(probs[i]) if probs is not None else None for i in range(4)},
                "heatmap": heatmap.tolist(),
                "step": int(step)
            }

            try:
                await websocket.send(json.dumps(packet))
            except Exception as e:
                print("[realtime_server] send failed:", e)
                break

            step += 1
            await asyncio.sleep(1.0)

    async def start(self):
        print(f"[realtime_server] running on ws://{self.host}:{self.port}")
        async with websockets.serve(self.stream_data, self.host, self.port):
            await asyncio.Future()

# -------------------------------------------------
# CLI entry
# -------------------------------------------------
if __name__ == "__main__":
    server = RealTimeServerHybrid()
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        print("Server stopped by user")