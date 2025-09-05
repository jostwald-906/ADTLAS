# main.py
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import json
import os
import openai

from simulation import run_simulation
from cost import compute_task_costs, compute_time_phased_costs, compute_npv_or_roi, scenario_comparison
from visualization import (
    plot_histograms, plot_depot_utilization, plot_tasks_over_time,
    plot_geospatial_depots, plot_supplier_inventory
)
from supply import suppliers
from config import SIM_TIME, BASE_INTERARRIVAL, SURGE_START, SURGE_END, SURGE_MULTIPLIER, QUEUE_THRESHOLD, INVESTMENT_COST_PER_CAPACITY_UNIT
from exec_summary_genai import generate_exec_summary_genai

st.set_page_config(page_title="ADTLAS Digital Twin", layout="wide")
st.title("ADTLAS: Advanced Digital Twin for Logistics & Sustainment")

# ---------------------- SIDEBAR ----------------------
st.sidebar.header("Simulation Parameters & Scenario Builder")

# Basic simulation inputs (all in days)
sim_time_input = st.sidebar.number_input("Simulation Time (days)", min_value=1, max_value=365, value=int(SIM_TIME), step=1)
base_interarrival_input = st.sidebar.number_input("Baseline Mean Interarrival (days)", min_value=0.001, max_value=1.0, value=BASE_INTERARRIVAL, step=0.001)
surge_start_input = st.sidebar.number_input("Surge Start (day)", min_value=0, max_value=sim_time_input, value=int(SURGE_START), step=1)
surge_end_input = st.sidebar.number_input("Surge End (day)", min_value=0, max_value=sim_time_input, value=int(SURGE_END), step=1)
surge_multiplier_input = st.sidebar.number_input("Surge Multiplier", min_value=0.1, max_value=5.0, value=SURGE_MULTIPLIER, step=0.1)
queue_threshold_input = st.sidebar.number_input("Queue Threshold", min_value=1, max_value=20, value=int(QUEUE_THRESHOLD), step=1)

# Policy toggles
st.sidebar.subheader("Policy Toggles")
policy_itar = st.sidebar.checkbox("ITAR Restrictions", value=True, key="itar_toggle")
policy_allied = st.sidebar.checkbox("Allow Allied Integration", value=True, key="allied_toggle")

# Economic discount rate for NPV
daily_discount_rate = st.sidebar.number_input("Daily Discount Rate", min_value=0.0, max_value=0.01, value=0.00095, step=0.00001)

# Scenario naming & running
st.sidebar.subheader("Create & Run Scenario")
scenario_name = st.sidebar.text_input("Scenario Name", value="My Scenario", key="scenario_name")
if "scenario_data" not in st.session_state:
    st.session_state.scenario_data = {}

def run_scenario():
    # Run simulation with user-defined parameters
    df_tasks, depot_data = run_simulation(
        sim_time=sim_time_input,
        base_interarrival=base_interarrival_input,
        surge_start=surge_start_input,
        surge_end=surge_end_input,
        surge_multiplier=surge_multiplier_input,
        queue_threshold=queue_threshold_input
    )
    # Compute cost breakdown per task
    df_costs = compute_task_costs(df_tasks)
    # Compute time-phased (daily) cost breakdown
    df_day = compute_time_phased_costs(df_costs)
    npc = compute_npv_or_roi(df_day, daily_discount_rate)
    st.session_state.scenario_data[scenario_name] = {
        "df_tasks": df_tasks,
        "df_costs": df_costs,
        "df_day": df_day,
        "depot_data": depot_data,
        "npc": npc,
        "sim_time": float(sim_time_input), 
    }
    st.success(f"Scenario '{scenario_name}' has been run and saved.")

st.sidebar.button("Run Scenario", on_click=run_scenario, key="run_scenario_button")

# Scenario Comparison Multi-select (for later comparison)
st.sidebar.subheader("Scenario Comparison")
if st.session_state.scenario_data:
    scenario_names = list(st.session_state.scenario_data.keys())
    selected_scenarios = st.sidebar.multiselect("Select Scenarios to Compare", scenario_names, default=scenario_names, key="scenario_compare")
else:
    selected_scenarios = []

# ---------------------- MAIN TABS ----------------------
tabs = st.tabs([
    "Overview",
    "Executive Summary (GenAI)",
    "Simulation Output",
    "Economic Analysis",
    "Geospatial",
    "Inventory",
    "Scenario Comparison",
    "AI Copilot"
])

# Tab 1: Overview
with tabs[0]:
    st.header("Overview")
    st.markdown("""
    **ADTLAS: Logistics & Sustainment Digital Twin**
    
    This model simulates the sustainment capabilities of various maintenance depots across different services and allied nations.
    
    **Inputs:**
    - **Simulation Time (days):** Total days to simulate operations.
    - **Baseline Mean Interarrival (days):** Average time between incoming maintenance tasks.
    - **Surge Parameters:** Define a period where maintenance demand increases.
    - **Queue Threshold:** Determines when tasks are re-routed to alternative depots.
    - **Policy Toggles:** Enable ITAR restrictions and allied integration.
    
    **Approach:**
    - Uses **SimPy** to model depot capacity, task processing, and supply chain dynamics.
    - Each depot handles multiple maintenance task types (e.g., engine, avionics, structural, routine) with specific service times.
    - A supply chain module models parts inventory for each aircraft type and repair category.
    - Economic analysis breaks down costs (labor, overhead, parts, shipping, downtime) and computes Net Present Cost (NPC).
    - Scenarios can be created, run, and compared.
    
    **Outputs:**
    - **Simulation Output:** Task metrics, depot utilization, production statistics.
    - **Economic Analysis:** Detailed cost breakdowns and NPC.
    - **Geospatial Visualization:** Depot locations and utilization mapping.
    - **Inventory Levels:** Trends in supplier inventory over time.
    - **Executive Summary:** A dynamic, GenAI-driven summary with strategic recommendations.
    - **Scenario Comparison:** Side-by-side comparison of scenario cost profiles.
    
    Use the sidebar to adjust parameters, run new scenarios, and name them for later comparison.
    """)

# Tab 2: Executive Summary (GenAI)
# Tab 2: Executive Summary (GenAI-driven)
with tabs[1]:
    st.header("Executive Summary (GenAI)")
    if st.session_state.scenario_data:
        selected_scenario_exec = st.selectbox("Select Scenario for Executive Summary", list(st.session_state.scenario_data.keys()), key="exec_select")
        scenario_dict = st.session_state.scenario_data[selected_scenario_exec]
        df_tasks = scenario_dict["df_tasks"]
        # Convert days to hours for summary clarity
        avg_wait = df_tasks['wait_time'].mean() * 24
        avg_service = df_tasks['service_time'].mean() * 24
        st.markdown(f"**Scenario:** {selected_scenario_exec}")
        st.markdown(f"**Total Tasks Processed:** {len(df_tasks)}")
        st.markdown(f"**Average Wait Time:** {avg_wait:.2f} hours")
        st.markdown(f"**Average Service Time:** {avg_service:.2f} hours")
        
        # Get the JSON output from your GenAI function
        exec_summary_json = generate_exec_summary_genai(df_tasks, scenario_dict["depot_data"], sim_time_input)
        
        # Attempt to parse the JSON and render a custom narrative layout
        try:
            
            summary_data = json.loads(exec_summary_json)
            
            # Render the content in a more narrative style
            st.markdown("### Executive Summary")
            
            st.markdown(summary_data.get("summary_text", "_No summary text._"))
        
            
            st.markdown("#### Key Metrics")
            st.markdown(f"- **Total Tasks**: {summary_data['total_tasks']}")
            st.markdown(f"- **Avg Wait (hours)**: {summary_data['avg_wait_hours']:.2f}")
            st.markdown(f"- **Avg Service (hours)**: {summary_data['avg_service_hours']:.2f}")
            
            st.markdown("#### Depot Utilizations")
            depot_utils = summary_data.get("depot_utilizations", {})
            if depot_utils:
                for depot, utilization in depot_utils.items():
                    st.markdown(f"- **{depot}**: {utilization}%")
            else:
                st.markdown("_No depot utilization data._")
            
            st.markdown("#### Inventory Availability")
            inventory_avail = summary_data.get("inventory_availability", {})
            if inventory_avail:
                for part, level in inventory_avail.items():
                    st.markdown(f"- **{part}**: average = {level['average']:.2f}, stockouts = {level['stockouts']}")
            else:
                st.markdown("_No inventory data._")
            
            st.markdown("#### Recommendations")
            recs = summary_data.get("recommendations", [])
            if recs:
                for i, rec in enumerate(recs, start=1):
                    st.markdown(f"{i}. {rec}")
            else:
                st.markdown("_No recommendations provided._")
            
            
        except json.JSONDecodeError:
            # If the output wasn't valid JSON, fallback to raw
            st.error("Could not parse the GenAI output as valid JSON.")
            st.markdown(exec_summary_json)
    else:
        st.info("No scenarios available for executive summary.")


# Tab 3: Simulation Output
with tabs[2]:
    st.header("Simulation Output")
    if st.session_state.scenario_data:
        selected_scenario_sim = st.selectbox("Select Scenario for Simulation Output", list(st.session_state.scenario_data.keys()), key="sim_select")
        scenario_dict = st.session_state.scenario_data[selected_scenario_sim]
        df_tasks = scenario_dict["df_tasks"]
        depot_data = scenario_dict["depot_data"]
        st.subheader("Summary Statistics")
        st.write("Total tasks processed:", len(df_tasks))
        st.dataframe(df_tasks.describe())
        st.subheader("Histograms")
        fig_hist = plot_histograms(df_tasks)
        st.pyplot(fig_hist)
        st.subheader("Depot Utilization")
        fig_util = plot_depot_utilization(depot_data, sim_time_input)
        st.pyplot(fig_util)
        st.subheader("Tasks Generated Over Time")
        fig_time = plot_tasks_over_time(df_tasks)
        st.plotly_chart(fig_time, use_container_width=True)
    else:
        st.info("No scenario data available.")

# Tab 4: Economic Analysis
with tabs[3]:
    st.header("Economic Analysis")
    if st.session_state.scenario_data:
        selected_scenario_econ = st.selectbox("Select Scenario for Economic Analysis", list(st.session_state.scenario_data.keys()), key="econ_select")
        scenario_dict = st.session_state.scenario_data[selected_scenario_econ]
        df_costs = scenario_dict["df_costs"]
        df_day = scenario_dict["df_day"]
        npc = scenario_dict["npc"]
        st.subheader("Cost Breakdown per Task")
        st.dataframe(df_costs[['task_id','aircraft_type','repair_type','labor_cost','overhead_cost','parts_cost','shipping_cost','downtime_cost','total_cost']])
        st.subheader("Daily Cost Breakdown")
        st.dataframe(df_day)
        st.subheader("Daily Cost Chart")
        fig_day = px.line(df_day, x='day', y=['labor_cost','overhead_cost','parts_cost','shipping_cost','downtime_cost'],
                           title="Daily Cost by Category", labels={"value": "Cost ($)", "day": "Day"}, markers=True)
        st.plotly_chart(fig_day, use_container_width=True)
        st.subheader("Net Present Cost")
        st.write(f"NPC for scenario '{selected_scenario_econ}': ${npc:,.2f}")
    else:
        st.info("No scenario data available.")

# Tab 5: Geospatial Visualization
with tabs[4]:
    st.header("Geospatial Visualization")
    if st.session_state.scenario_data:
        selected_scenario_geo = st.selectbox("Select Scenario for Geospatial View", list(st.session_state.scenario_data.keys()), key="geo_select")
        depot_data = st.session_state.scenario_data[selected_scenario_geo]["depot_data"]
        fig_geo = plot_geospatial_depots(depot_data, sim_time_input)
        st.plotly_chart(fig_geo, use_container_width=True)
    else:
        st.info("No scenario data available.")

# Tab 6: Inventory
with tabs[5]:
    st.header("Inventory Levels")
    supplier_keys = list(suppliers.keys())
    if supplier_keys:
        supplier_options = {f"{m} - {r}": (m, r) for m, r in supplier_keys}
        selected_supplier_str = st.selectbox("Select Inventory Type", list(supplier_options.keys()), key="inventory_select")
        selected_supplier = supplier_options[selected_supplier_str]
        fig_inventory = plot_supplier_inventory(suppliers, selected_supplier)
        if fig_inventory:
            st.plotly_chart(fig_inventory, use_container_width=True)
        else:
            st.info("No inventory history available for the selected supplier.")
    else:
        st.info("No suppliers defined.")

# Tab: Scenario Comparison
with tabs[6]:
    st.header("Scenario Comparison")
    scenario_list = list(st.session_state.scenario_data.keys())
    if scenario_list:
        selected_scenarios_cmp = st.multiselect("Select Scenarios to Compare", scenario_list, default=scenario_list, key="compare_multiselect")
        from cost import scenario_comparison
        df_compare = scenario_comparison({sc: st.session_state.scenario_data[sc]["df_day"] for sc in selected_scenarios_cmp})
        if not df_compare.empty:
            st.subheader("Daily Cost Comparison")
            st.dataframe(df_compare)
            df_melt = df_compare.melt(id_vars=['day'], var_name='scenario', value_name='cost')
            fig_cmp = px.line(df_melt, x='day', y='cost', color='scenario',
                              title="Daily Cost Comparison Across Scenarios", markers=True)
            st.plotly_chart(fig_cmp, use_container_width=True)
        else:
            st.info("No data to compare.")
    else:
        st.info("No scenarios have been run yet.")

# --- Tab: AI Copilot ---
with tabs[7]:
    st.header("AI Copilot (Chat)")

    import os, openai, json
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Base system instruction (kept across turns)
    BASE_SYSTEM = {
        "role": "system",
        "content": (
            "You are the ADTLAS sustainment copilot for USAF. "
            "Be concise, action-oriented, and specific. Focus on readiness, availability, cost efficiency, and ROI. "
            "If asked for metrics, only use the scenario context provided."
        )
    }

    # Init chat state
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []  # we'll rebuild every send
    if "last_chat_scenario" not in st.session_state:
        st.session_state.last_chat_scenario = None

    if not st.session_state.scenario_data:
        st.info("No scenarios available for chat.")
        st.stop()

    # Select scenario for the chat
    selected_for_chat = st.selectbox(
        "Select scenario to prime the Copilot context",
        list(st.session_state.scenario_data.keys()),
        key="chat_scenario_select",
    )

    # If scenario changed, clear chat to avoid stale context  (NEW)
    if st.session_state.last_chat_scenario != selected_for_chat:
        st.session_state.chat_messages = []  # clear prior messages
        st.session_state.last_chat_scenario = selected_for_chat

    scen = st.session_state.scenario_data[selected_for_chat]
    df_tasks_ctx = scen["df_tasks"]
    depot_data_ctx = scen["depot_data"]
    scen_sim_time = float(scen.get("sim_time", 1.0)) or 1.0  # safety

    # Build fresh scenario context (percent utilization, scenario sim_time)  (NEW)
    try:
        avg_wait_h = (df_tasks_ctx["wait_time"].mean() * 24) if not df_tasks_ctx.empty else 0.0
        avg_service_h = (df_tasks_ctx["service_time"].mean() * 24) if not df_tasks_ctx.empty else 0.0
        util_lines = []
        for dkey, d in depot_data_ctx.items():
            # percent utilization for this scenario
            util_pct = (d.total_service_time / (scen_sim_time * d.capacity) * 100.0) if d.capacity else 0.0
            util_lines.append(f"{d.name}: {util_pct:.2f}%")
        context_blob = (
            f"Scenario: {selected_for_chat}\n"
            f"Total tasks: {len(df_tasks_ctx)}\n"
            f"Avg wait (h): {avg_wait_h:.2f}, Avg service (h): {avg_service_h:.2f}\n"
            f"Depot utilization (%): {', '.join(util_lines)}\n"
            "Use ONLY this context when answering questions."
        )
    except Exception:
        context_blob = f"Scenario: {selected_for_chat}\nContext build error; answer generally."

    # Chat settings
    col1, col2, col3 = st.columns([1.3, 1, 1])
    with col1:
        model_name = st.selectbox("Model", ["gpt-4o", "gpt-4", "gpt-3.5-turbo"], index=0, key="chat_model")
    with col2:
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05, key="chat_temp")
    with col3:
        if st.button("Clear chat"):
            st.session_state.chat_messages = []
            st.experimental_rerun()

    # Render history (we render only user/assistant; context is rebuilt each send)  (CHANGED)
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_msg = st.chat_input("Ask the ADTLAS Copilot...")
    if user_msg:
        # Append user message to UI history
        st.session_state.chat_messages.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)

        try:
            # Build messages fresh each call: base system + scenario context + past convo + latest user (NEW)
            # We include prior user/assistant messages to keep short memory, but scenario context is always rebuilt.
            short_hist = [m for m in st.session_state.chat_messages if m["role"] in ("user", "assistant")][-6:]
            messages = [
                BASE_SYSTEM,
                {"role": "system", "name": "scenario_context", "content": context_blob},
                *short_hist
            ]

            resp = openai.ChatCompletion.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=800,
            )
            assistant_text = resp["choices"][0]["message"]["content"]
            with st.chat_message("assistant"):
                st.markdown(assistant_text)
            st.session_state.chat_messages.append({"role": "assistant", "content": assistant_text})

        except Exception as e:
            st.error(f"OpenAI error: {e}")



