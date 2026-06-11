import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json
import websockets
import numpy as np

from digital_twin.machine_physics import MachinePhysics
from gnn.factory_graph import FactoryGraph
from rl_agent.ppo_agent import PPOAgent
from vision.drone_inspector import DroneInspector


class RealTimeServer:
    """
    Real-time WebSocket data stream for smart factory dashboard.
    Streams:
    - Digital Twin machine physics data
    - GNN risk score
    - Drone hotspot detection
    - PPO RL actions
    """

    def __init__(self, host="localhost", port=8765):
        self.host = host
        self.port = port

        # Initialize 10 physics-based machines
        self.factory = [MachinePhysics(i) for i in range(10)]

        # Factory graph for GNN
        self.graph = FactoryGraph(num_machines=10)

        # Drone inspector (thermal hotspot detection)
        self.drone = DroneInspector()

        # PPO reinforcement learning agent
        self.agent = PPOAgent(
            state_dim=5,       # temp, vibration, load, wear, failure_prob
            action_dim=4       # actions: run, cool, idle, maintain
        )

        self.action_map = ["run", "cool", "idle", "maintain"]

    # ------------------------------------------------------
    # MAIN STREAM LOOP
    # ------------------------------------------------------
    async def stream_data(self, websocket):
        while True:

            # 1️⃣ Simulate digital twin for ALL machines
            machine_states = [m.simulate_step("run") for m in self.factory]

            # 2️⃣ Compute GNN graph-level risk
            A, X = self.graph.to_graph_data(machine_states)
            gnn_risk = float(np.mean(X[:, -1]))  # failure_prob avg

            # 3️⃣ Drone hotspot detection with synthetic heatmap
            heatmap = np.random.uniform(30, 90, (6, 6))
            hotspots = self.drone.detect_hotspots(heatmap)

            # 4️⃣ PPO RL agent takes action for machine 0
            s0 = np.array([
                machine_states[0]["temp"],
                machine_states[0]["vibration"],
                machine_states[0]["load"],
                machine_states[0]["wear"],
                machine_states[0]["failure_prob"]
            ])

            action_idx, policy = self.agent.act(s0)
            action = self.action_map[action_idx]

            # Apply RL action to machine 0
            machine_states[0] = self.factory[0].simulate_step(action)

            # 5️⃣ Package the message to send to dashboard
            payload = {
                "machines": machine_states,
                "drone_hotspots": hotspots,
                "gnn_risk": gnn_risk,
                "rl_action": action,
                "policy": policy.tolist(),
                "heatmap": heatmap.tolist(),
            }

            await websocket.send(json.dumps(payload))

            await asyncio.sleep(1)  # every 1 sec update

    # ------------------------------------------------------
    # START SERVER
    # ------------------------------------------------------
    async def start(self):
        print(f"Real-time server running at ws://{self.host}:{self.port}")
        async with websockets.serve(self.stream_data, self.host, self.port):
            await asyncio.Future()  # run forever


# Entry point
if __name__ == "__main__":
    server = RealTimeServer()
    asyncio.run(server.start())