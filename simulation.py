# simulation.py
import simpy, random, pandas as pd
from config import SIM_TIME, BASE_INTERARRIVAL, SURGE_START, SURGE_END, SURGE_MULTIPLIER, QUEUE_THRESHOLD, POLICY_ALLIED_INTEGRATION
from supply import initialize_suppliers, suppliers

# Depot model with service times now in days
class Depot:
    def __init__(self, env, name, capacity, service_times, service_types, location, is_allied=False, geo=None):
        """
        env: simpy.Environment
        name: Depot name (e.g., "Tinker AFB")
        capacity: Number of maintenance teams
        service_times: Dict mapping (aircraft_type, repair_type) to mean service time (in days)
        service_types: List of aircraft types (MDS) the depot supports
        location: Location string for display purposes
        is_allied: True if the depot is allied
        geo: Tuple (lat, lon) for geospatial mapping
        """
        self.env = env
        self.name = name
        self.capacity = capacity
        self.service_times = service_times
        self.service_types = service_types
        self.location = location
        self.is_allied = is_allied
        self.geo = geo if geo else (0.0, 0.0)
        self.resource = simpy.Resource(env, capacity=capacity)
        self.tasks_processed = 0
        self.total_service_time = 0.0
        self.total_wait_time = 0.0

depots = {}

def find_alternative(primary_depot_key, aircraft_type):
    """
    Return an alternative depot key if available based on simple routing rules.
    For USAF/USN, try switching between Tinker and FRCSW if applicable.
    For allied depots, switch between Waddington and JASDF if allowed.
    """
    alternatives = []
    # Check cross-service alternatives
    if primary_depot_key == 'FRCSW' and aircraft_type in depots['Tinker'].service_types:
        alternatives.append('Tinker')
    elif primary_depot_key == 'Tinker' and aircraft_type in depots['FRCSW'].service_types:
        alternatives.append('FRCSW')
    
    # Check allied alternatives if policy allows
    if POLICY_ALLIED_INTEGRATION:
        if primary_depot_key == 'Waddington' and aircraft_type in depots['JASDF'].service_types:
            alternatives.append('JASDF')
        elif primary_depot_key == 'JASDF' and aircraft_type in depots['Waddington'].service_types:
            alternatives.append('Waddington')
    
    if alternatives:
        # Return the alternative with the smallest queue length.
        return min(alternatives, key=lambda key: len(depots[key].resource.queue))
    return None

def initialize_depots(env):
    # Tinker AFB: service times in days
    depots['Tinker'] = Depot(
        env,
        name='Tinker AFB',
        capacity=5,
        service_times={
            ('B-1B','engine'): 55/24,
            ('B-1B','avionics'): 35/24,
            ('B-1B','structure'): 60/24,
            ('B-1B','routine'): 20/24,
            ('B-52','engine'): 75/24,
            ('B-52','avionics'): 45/24,
            ('B-52','structure'): 70/24,
            ('B-52','routine'): 25/24,
            ('KC-135','engine'): 45/24,
            ('KC-135','avionics'): 30/24,
            ('KC-135','structure'): 50/24,
            ('KC-135','routine'): 20/24
        },
        service_types=['B-1B','B-52','KC-135'],
        location='Oklahoma, USA',
        is_allied=False,
        geo=(35.22, -97.44)
    )
    # FRCSW: service times for tactical aircraft
    depots['FRCSW'] = Depot(
        env,
        name='FRCSW',
        capacity=4,
        service_times={
            ('F/A-18','engine'): 28/24,
            ('F/A-18','avionics'): 22/24,
            ('F/A-18','structure'): 30/24,
            ('F/A-18','routine'): 15/24,
            ('H-60','engine'): 22/24,
            ('H-60','avionics'): 18/24,
            ('H-60','structure'): 25/24,
            ('H-60','routine'): 12/24,
            ('V-22','engine'): 33/24,
            ('V-22','avionics'): 26/24,
            ('V-22','structure'): 35/24,
            ('V-22','routine'): 18/24
        },
        service_types=['F/A-18','H-60','V-22'],
        location='California, USA',
        is_allied=False,
        geo=(32.70, -117.20)
    )
    # RAF Waddington: service times for ISR aircraft (in days)
    depots['Waddington'] = Depot(
        env,
        name='RAF Waddington',
        capacity=3,
        service_times={
            ('RC-135','avionics'): 42/24,
            ('RC-135','routine'): 28/24,
            ('ISR','avionics'): 38/24,
            ('ISR','routine'): 25/24
        },
        service_types=['RC-135','ISR'],
        location='UK',
        is_allied=True,
        geo=(53.17, -0.55)
    )
    # JASDF: service times for F-15J (in days)
    depots['JASDF'] = Depot(
        env,
        name='JASDF Depot',
        capacity=3,
        service_times={
            ('F-15J','engine'): 35/24,
            ('F-15J','routine'): 30/24
        },
        service_types=['F-15J'],
        location='Japan',
        is_allied=True,
        geo=(35.0, 135.0)
    )

# Extend MaintenanceTask to include repair_type
class MaintenanceTask:
    def __init__(self, task_id, aircraft_type, repair_type, arrival_time, primary_depot):
        self.task_id = task_id
        self.aircraft_type = aircraft_type
        self.repair_type = repair_type  # e.g., 'engine', 'avionics', etc.
        self.arrival_time = arrival_time
        self.primary_depot = primary_depot  # initially assigned depot key
        self.assigned_depot = primary_depot   # might change if re-routed
        self.start_service = None
        self.end_service = None

task_records = []

# Repair type distribution remains the same.
repair_type_distribution = {
    'B-1B': {'engine': 0.4, 'avionics': 0.3, 'structure': 0.2, 'routine': 0.1},
    'B-52': {'engine': 0.35, 'avionics': 0.3, 'structure': 0.25, 'routine': 0.1},
    'KC-135': {'engine': 0.4, 'avionics': 0.3, 'structure': 0.15, 'routine': 0.15},
    'F/A-18': {'engine': 0.3, 'avionics': 0.4, 'structure': 0.2, 'routine': 0.1},
    'H-60': {'engine': 0.25, 'avionics': 0.35, 'structure': 0.25, 'routine': 0.15},
    'V-22': {'engine': 0.3, 'avionics': 0.35, 'structure': 0.25, 'routine': 0.1},
    'RC-135': {'avionics': 0.6, 'routine': 0.4},
    'ISR': {'avionics': 0.55, 'routine': 0.45},
    'F-15J': {'engine': 0.5, 'routine': 0.5}
}

# Parts required per repair type
repair_parts_required = {
    'engine': 5,
    'avionics': 3,
    'structure': 4,
    'routine': 2
}

def process_task(env, task):
    """Process a maintenance task including a supply chain check."""
    depot = depots[task.assigned_depot]
    req_time = env.now
    with depot.resource.request() as req:
        yield req
        wait_time = env.now - req_time
        task.start_service = env.now
        depot.total_wait_time += wait_time

        # Supply chain check: request parts from the relevant supplier.
        # Key is (aircraft_type, repair_type)
        supplier_key = (task.aircraft_type, task.repair_type)
        if supplier_key in suppliers:
            required_parts = repair_parts_required.get(task.repair_type, 2)
            if required_parts > 0:  # Only request if amount is > 0
                yield suppliers[supplier_key].inventory.get(required_parts)
        
        # Determine service time based on (aircraft_type, repair_type)
        key = (task.aircraft_type, task.repair_type)
        mean_service = depot.service_times.get(key, 35/24)
        service_time = random.expovariate(1.0/mean_service)
        yield env.timeout(service_time)
        task.end_service = env.now

        depot.total_service_time += service_time
        depot.tasks_processed += 1

        task_records.append({
            'task_id': task.task_id,
            'aircraft_type': task.aircraft_type,
            'repair_type': task.repair_type,
            'primary_depot': task.primary_depot,
            'assigned_depot': task.assigned_depot,
            'arrival_time': task.arrival_time,
            'start_service': task.start_service,
            'end_service': task.end_service,
            'wait_time': task.start_service - task.arrival_time,
            'service_time': service_time,
            'total_time': task.end_service - task.arrival_time
        })



def task_generator(env, base_interarrival, surge_start, surge_end, surge_multiplier, queue_threshold):
    """Generate maintenance tasks with repair type selection and possible re-routing."""
    task_id = 0
    aircraft_types = {
        'Tinker': ['B-1B', 'B-52', 'KC-135'],
        'FRCSW': ['F/A-18', 'H-60', 'V-22'],
        'Waddington': ['RC-135', 'ISR'],
        'JASDF': ['F-15J']
    }
    primary_mapping = {t: depot for depot, types in aircraft_types.items() for t in types}
    
    while True:
        if surge_start <= env.now <= surge_end:
            interarrival = base_interarrival * surge_multiplier
        else:
            interarrival = base_interarrival
        yield env.timeout(random.expovariate(1.0/interarrival))
        
        task_id += 1
        aircraft_type = random.choice(list(primary_mapping.keys()))
        primary_depot = primary_mapping[aircraft_type]
        dist = repair_type_distribution.get(aircraft_type, {'routine': 1.0})
        repair_types = list(dist.keys())
        probabilities = list(dist.values())
        repair_type = random.choices(repair_types, weights=probabilities, k=1)[0]
        task = MaintenanceTask(task_id, aircraft_type, repair_type, env.now, primary_depot)
        
        if len(depots[primary_depot].resource.queue) >= queue_threshold:
            alternative = find_alternative(primary_depot, aircraft_type)
            if alternative:
                yield env.timeout(2/24)  # transfer delay of 2 hours in days
                task.assigned_depot = alternative
        env.process(process_task(env, task))

def run_simulation(sim_time=SIM_TIME, base_interarrival=BASE_INTERARRIVAL,
                   surge_start=SURGE_START, surge_end=SURGE_END,
                   surge_multiplier=SURGE_MULTIPLIER, queue_threshold=QUEUE_THRESHOLD):
    """Run the simulation (in days) and return task records and depot data."""
    env = simpy.Environment()
    initialize_depots(env)
    initialize_suppliers(env)
    env.process(task_generator(env, base_interarrival, surge_start, surge_end, surge_multiplier, queue_threshold))
    env.run(until=sim_time)
    df = pd.DataFrame(task_records)
    return df, depots
