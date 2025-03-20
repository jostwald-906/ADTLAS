# cost.py
import pandas as pd
import numpy as np

from config import COST_PER_HOUR_DOWNTIME, REPAIR_COST_MULTIPLIER, INVESTMENT_COST_PER_CAPACITY_UNIT

def compute_task_costs(df_tasks, labor_rate=500, overhead_rate=2000, shipping_rate=10000):
    """
    Assign multiple cost categories to each task in df_tasks:
      - labor_cost: (service_time in hours) * labor_rate
      - parts_cost: (some function of parts usage)
      - overhead_cost: (overhead_rate * fraction_of_day?)
      - shipping_cost: e.g., if re-routed, we can add shipping cost
      - downtime_cost: (wait_time in hours) * COST_PER_HOUR_DOWNTIME

    Returns a new DataFrame with cost columns appended.
    """
    df = df_tasks.copy()
    
    # Convert days to hours for cost calculations
    df['wait_hrs'] = df['wait_time'] * 24
    df['service_hrs'] = df['service_time'] * 24
    
    # Example labor cost = service_hrs * labor_rate
    df['labor_cost'] = df['service_hrs'] * labor_rate
    
    # Example overhead cost (assume overhead per day). You can refine as needed.
    # We'll approximate overhead for the fraction of day in service_time
    df['overhead_cost'] = df['service_time'] * overhead_rate
    
    # Example shipping cost: if a task was re-routed, we add shipping.
    # We'll do a simple check if assigned_depot != primary_depot
    # shipping_rate is cost per hour of transport; let's assume 2 hours if re-routed
    df['shipping_cost'] = np.where(
        df['assigned_depot'] != df['primary_depot'],
        2 * shipping_rate,
        0
    )
    
    # Parts cost: we can approximate by (service_time * REPAIR_COST_MULTIPLIER * 1000) 
    # or look at actual parts usage from your supply chain. 
    # For simplicity, we'll just do a fraction of service_time.
    df['parts_cost'] = df['service_hrs'] * REPAIR_COST_MULTIPLIER * 10  # arbitrary factor
    
    # Downtime cost: wait_hrs * COST_PER_HOUR_DOWNTIME
    df['downtime_cost'] = df['wait_hrs'] * COST_PER_HOUR_DOWNTIME
    
    # total_cost for each task
    df['total_cost'] = (df['labor_cost'] + df['overhead_cost'] +
                        df['shipping_cost'] + df['parts_cost'] +
                        df['downtime_cost'])
    return df

def compute_time_phased_costs(df_costs):
    """
    Create a day-by-day cost breakdown. We'll assume each task's cost is
    incurred continuously from start_service to end_service, or you can
    approximate daily cost as total_cost allocated to that day.

    For simplicity, we'll allocate each task's total_cost evenly across its service_time.
    We'll then sum for each day.

    Returns a DataFrame with columns: day, labor_cost, overhead_cost, parts_cost, shipping_cost, downtime_cost, total_cost
    """
    records = []
    for _, row in df_costs.iterrows():
        start = row['start_service']
        end = row['end_service']
        total_days = end - start if (end > start) else 0
        # If total_days is 0, cost is immediate or negligible. We'll just put it in start day.
        daily_allocation = {}
        for cat in ['labor_cost', 'overhead_cost', 'parts_cost', 'shipping_cost', 'downtime_cost']:
            daily_allocation[cat] = row[cat] / (total_days if total_days else 1)
        
        day = start
        while day < end:
            fraction = 1.0
            # We'll allocate cost in increments of 1 day.
            next_day = np.floor(day+1)
            if next_day > end:
                fraction = end - day
            elif next_day - day < 1.0:
                fraction = next_day - day
            
            # day_idx = int(np.floor(day))  # or keep as float for partial days
            day_idx = int(np.floor(day)) 
            rec = {
                'day': day_idx
            }
            # Allocate fraction of daily_allocation
            for cat in daily_allocation:
                rec[cat] = daily_allocation[cat] * fraction
            records.append(rec)
            day += fraction
    
    df_day = pd.DataFrame(records)
    if df_day.empty:
        # fallback if no tasks or no service_time
        return pd.DataFrame(columns=['day','labor_cost','overhead_cost','parts_cost','shipping_cost','downtime_cost','total_cost'])
    df_day = df_day.groupby('day').sum().reset_index()
    df_day['total_cost'] = df_day[['labor_cost','overhead_cost','parts_cost','shipping_cost','downtime_cost']].sum(axis=1)
    return df_day

def compute_npv_or_roi(df_day, daily_discount_rate=0.00095):
    """
    Example function to compute the NPV of the total cost stream across days,
    using a daily discount rate. 
    daily_discount_rate ~ 0.00095 => about 3.5% annual discount rate.

    Returns a single float representing the net present cost (NPC).
    """
    df = df_day.copy()
    df['discount_factor'] = (1.0 / ((1+daily_discount_rate) ** df['day']))
    df['discounted_cost'] = df['total_cost'] * df['discount_factor']
    return df['discounted_cost'].sum()

def scenario_comparison(dict_of_scenarios):
    """
    If you store multiple scenario runs in a dictionary:
       dict_of_scenarios[scenario_name] = df_day
    you can produce a comparison DataFrame or charts.

    Returns a DataFrame that merges each scenario's daily cost side by side.
    """
    all_dfs = []
    for scenario_name, df_day in dict_of_scenarios.items():
        df_temp = df_day[['day','total_cost']].copy()
        df_temp.rename(columns={'total_cost': f'{scenario_name}_cost'}, inplace=True)
        all_dfs.append(df_temp)
    if not all_dfs:
        return pd.DataFrame()
    df_merged = all_dfs[0]
    for df_temp in all_dfs[1:]:
        df_merged = pd.merge(df_merged, df_temp, on='day', how='outer')
    df_merged = df_merged.fillna(0)
    return df_merged
