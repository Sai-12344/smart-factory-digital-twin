import streamlit as st
import asyncio
import websockets
import json
import pandas as pd
import numpy as np
import plotly.express as px
import time

st.set_page_config(page_title="Smart Factory", page_icon="🏭", layout="wide")
st.title("🏭 Real-Time Smart Factory (AI + Digital Twin + GNN + RL + Drones)")

async def fetch_async():
    uri = "ws://localhost:8765"
    try:
        async with websockets.connect(uri) as websocket:
            data = await websocket.recv()
            return json.loads(data)
    except:
        return None

def fetch_data_sync():
    return asyncio.run(fetch_async())

placeholder = st.empty()

while True:
    data = fetch_data_sync()

    if data is None:
        with placeholder.container():
            st.error("Unable to connect to real-time server. Make sure realtime_server.py is running.")
        time.sleep(2)
        continue

    machines = data["machines"]
    drone_hotspots = data["drone_hotspots"]
    rl_action = data["rl_action"]
    gnn_risk = data["gnn_risk"]
    heatmap = data["heatmap"]

    df = pd.DataFrame(machines)

    with placeholder.container():
        st.subheader("⚡ Live Machine Status")
        st.dataframe(df.style.background_gradient(cmap="RdYlGn_r"), use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔥 Drone Heatmap")
            fig = px.imshow(np.array(heatmap), color_continuous_scale="hot")
            st.plotly_chart(fig, use_container_width=True)
            st.write("Hotspots detected:", len(drone_hotspots))

        with col2:
            st.subheader("🧠 GNN Factory Risk")
            st.metric("Risk Level", f"{gnn_risk:.3f}")
            st.subheader("🤖 PPO RL Action")
            st.metric("Decision", rl_action)

    time.sleep(1)
