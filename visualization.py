# visualization.py
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

def plot_histograms(df):
    """Return a Matplotlib figure with histograms for wait, service, and total times (in days)."""
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    axs[0].hist(df['wait_time'], bins=20, edgecolor='black', color='#4e79a7')
    axs[0].set_title('Wait Time Distribution (days)')
    axs[0].set_xlabel('Wait Time (days)')
    axs[0].set_ylabel('Frequency')
    
    axs[1].hist(df['service_time'], bins=20, edgecolor='black', color='#f28e2c')
    axs[1].set_title('Service Time Distribution (days)')
    axs[1].set_xlabel('Service Time (days)')
    axs[1].set_ylabel('Frequency')
    
    axs[2].hist(df['total_time'], bins=20, edgecolor='black', color='#76b7b2')
    axs[2].set_title('Total Time Distribution (days)')
    axs[2].set_xlabel('Total Time (days)')
    axs[2].set_ylabel('Frequency')
    fig.tight_layout()
    return fig

def plot_depot_utilization(depot_data, sim_time):
    """Return a Matplotlib bar chart of depot utilization (in days)."""
    names, utilizations, tasks = [], [], []
    for key, depot in depot_data.items():
        util = depot.total_service_time / (sim_time * depot.capacity)
        names.append(depot.name)
        utilizations.append(util)
        tasks.append(depot.tasks_processed)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, utilizations, color='#59a14f', edgecolor='black')
    ax.set_title('Depot Utilization (Fraction of Capacity)')
    ax.set_xlabel('Depot')
    ax.set_ylabel('Utilization')
    ax.set_ylim(0, 1.0)
    for bar, t in zip(bars, tasks):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, height, f'{t}', ha='center', va='bottom')
    return fig

def plot_tasks_over_time(df, interval=1):
    """
    Return a Plotly line chart of tasks generated per day.
    Interval is in days.
    """
    df = df.copy()
    df['day'] = df['arrival_time'] // interval
    tasks_day = df.groupby('day').size().reset_index(name='tasks')
    fig = px.line(tasks_day, x='day', y='tasks', markers=True,
                  title="Tasks Generated Per Day",
                  labels={"day": "Day", "tasks": "Number of Tasks"})
    return fig

def plot_geospatial_depots(depot_data, sim_time):
    """Return a Plotly map showing depot locations and utilization."""
    data = []
    for key, depot in depot_data.items():
        util = depot.total_service_time / (sim_time * depot.capacity)
        data.append({
            'Depot': depot.name,
            'Latitude': depot.geo[0],
            'Longitude': depot.geo[1],
            'Utilization': util
        })
    df_geo = pd.DataFrame(data)
    fig = px.scatter_mapbox(df_geo, lat="Latitude", lon="Longitude", hover_name="Depot",
                            size="Utilization", color="Utilization",color_continuous_scale="Viridis",# pick any named scale
                            range_color=[0, 1],size_max=15, zoom=1,
                            mapbox_style="open-street-map",
                            title="Depot Locations & Utilization")
    return fig

def plot_supplier_inventory(suppliers, selected_supplier):
    """
    Return a Plotly line chart showing inventory history over time for the selected supplier.
    
    Parameters:
      suppliers: Global supplier dictionary.
      selected_supplier: A key (tuple) identifying the supplier (e.g., ('B-52','engine')).
    """
    supplier = suppliers.get(selected_supplier)
    if supplier and supplier.history:
        df_hist = pd.DataFrame(supplier.history, columns=['time', 'inventory'])
        fig = px.line(df_hist, x='time', y='inventory',
                      title=f"Inventory Over Time: {supplier.name}",
                      labels={'time': 'Time (days)', 'inventory': 'Inventory Level'})
        return fig
    else:
        return None
