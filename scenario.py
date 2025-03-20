# scenario.py
# Pre-defined scenarios using days directly (no division)

from config import SIM_TIME, BASE_INTERARRIVAL, POLICY_ITAR_RESTRICTED

def get_conflict_scenario():
    """
    Conflict scenario: increased frequency of tasks, longer surge, and reduced depot capacity.
    """
    scenario = {
        "sim_time": SIM_TIME,              
        "base_interarrival": BASE_INTERARRIVAL * 0.9,  # Slightly more frequent tasks
        "surge_start": 10,                 
        "surge_end": 16,                   
        "surge_multiplier": 0.4,           
        "queue_threshold": 5,
        "depot_capacity_adjustments": {
            "Tinker": -1,
            "FRCSW": 0,
            "Waddington": 0,
            "JASDF": 0
        },
        "policy_itar": True,
        "supplier_disruption": False
    }
    return scenario

def get_natural_disaster_scenario():
    """
    Natural disaster: FRCSW experiences reduced capacity and supplier disruption.
    """
    scenario = {
        "sim_time": SIM_TIME,
        "base_interarrival": BASE_INTERARRIVAL,
        "surge_start": 13,                 
        "surge_end": 21,                   
        "surge_multiplier": 0.6,           
        "queue_threshold": 7,
        "depot_capacity_adjustments": {
            "FRCSW": -2  
        },
        "policy_itar": POLICY_ITAR_RESTRICTED,
        "supplier_disruption": True        
    }
    return scenario

def get_long_term_downtime_scenario():
    """
    Long-term downtime: key depot (Tinker) is offline.
    """
    scenario = {
        "sim_time": SIM_TIME,
        "base_interarrival": BASE_INTERARRIVAL,
        "surge_start": 0,
        "surge_end": 0,  
        "surge_multiplier": 1.0,
        "queue_threshold": 5,
        "depot_capacity_adjustments": {
            "Tinker": -5  
        },
        "policy_itar": POLICY_ITAR_RESTRICTED,
        "supplier_disruption": False
    }
    return scenario
