# config.py
# All time parameters are now expressed directly in days.

# Simulation parameters (time in days)
SIM_TIME = 30                    # Total simulation time in days
BASE_INTERARRIVAL = 0.0625       # Average interarrival time in days (1.5 hours)
SURGE_START = 10                 # Surge event starts on day 10
SURGE_END = 14                   # Surge event ends on day 14
SURGE_MULTIPLIER = 0.5           # During surge, interarrival is reduced by half
QUEUE_THRESHOLD = 6              # Reroute if depot queue length exceeds threshold

# Policy toggles
POLICY_ITAR_RESTRICTED = True    # ITAR restrictions enabled by default
POLICY_ALLIED_INTEGRATION = True # Allow allied depots to receive tasks

# Economic parameters
COST_PER_HOUR_DOWNTIME = 15000   # $15,000 per hour of downtime
REPAIR_COST_MULTIPLIER = 1.5     # Multiplier for service time cost estimation
INVESTMENT_COST_PER_CAPACITY_UNIT = 750000  # $750,000 per additional capacity unit

# Fleet and readiness assumptions
TOTAL_FLEET = 250

# Supply chain disruption parameters (time in days)
SUPPLIER_DISRUPTION_START = 16.67  # e.g., starts at day 16.67
SUPPLIER_DISRUPTION_END = 18.75    # e.g., ends at day 18.75
