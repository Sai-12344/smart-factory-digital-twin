import streamlit as st
import numpy as np
import plotly.express as px
import pandas as pd


# --------------------------
# MACHINE HEALTH CARD
# --------------------------
def machine_health_card(machine_id, rul, health_score, anomaly, error):
    color = "red" if anomaly else "green"
    st.markdown(
        f"""
        <div style='background:#1a1a1a; padding:15px; border-radius:10px;'>
            <h3 style='color:{color};'>Machine {machine_id}</h3>
            <p><b>RUL:</b> {rul:.2f} hours</p>
            <p><b>Health Score:</b> {health_score:.2f}</p>
            <p><b>Anomaly:</b> {"YES" if anomaly else "NO"}</p>
            <p><b>Reconstruction Error:</b> {error:.5f}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# --------------------------
# LIVE SENSOR LINE CHART
# --------------------------
def live_sensor_chart(sensor_data_df):
    fig = px.line(sensor_data_df, title="Live Sensor Data")
    st.plotly_chart(fig, use_container_width=True)


# --------------------------
# DRONE HEATMAP
# --------------------------
def drone_heatmap(heatmap):
    df = pd.DataFrame(heatmap)
    fig = px.imshow(df, color_continuous_scale="RdYlGn", title="Drone Heatmap")
    st.plotly_chart(fig, use_container_width=True)


# --------------------------
# ENERGY USAGE PANEL
# --------------------------
def energy_usage_panel(total_energy, actions):
    st.subheader("⚡ Energy Usage Summary")
    st.write(f"**Total Energy Consumption:** {total_energy:.2f} units")

    if actions:
        st.write("### 🛠 Suggested Optimizations:")
        for machine_id, suggestion in actions:
            st.write(f"• Machine {machine_id}: {suggestion}")
    else:
        st.write("All machines operating efficiently!")