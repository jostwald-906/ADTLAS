import simpy, random
from config import SUPPLIER_DISRUPTION_START, SUPPLIER_DISRUPTION_END

class Supplier:
    def __init__(self, env, name, initial_inventory, base_supply_rate, capacity, downtime_periods=None):
        """
        env: simpy.Environment
        name: Supplier name (e.g., "B-52 Engine Parts Supplier")
        initial_inventory: Initial number of parts available (set lower to allow runouts)
        base_supply_rate: Parts replenished per day under normal conditions
        capacity: Maximum parts inventory
        downtime_periods: List of tuples (start_day, end_day) for downtime
        """
        self.env = env
        self.name = name
        self.capacity = capacity
        self.base_supply_rate = base_supply_rate
        self.inventory = simpy.Container(env, init=initial_inventory, capacity=capacity)
        self.downtime_periods = downtime_periods if downtime_periods else []
        self.history = []  # Record (time, inventory level) for visualization
        env.process(self.replenish())

    def current_supply_rate(self):
        # Check if current time (in days) is within any downtime period.
        for (start, end) in self.downtime_periods:
            if start <= self.env.now <= end:
                return 0  # Supplier is offline during downtime.
        # Add a 10% random fluctuation.
        variability = self.base_supply_rate * 0.1 * (1 if random.random() > 0.5 else -1)
        return max(0, self.base_supply_rate + variability)

    def replenish(self):
        """Replenish the inventory every day based on current supply rate, and log the inventory level."""
        while True:
            yield self.env.timeout(1)  # every day
            rate = self.current_supply_rate()
            # Only add whole parts (rounded to nearest integer) if rate > 0
            if rate > 0:
                yield self.inventory.put(int(round(rate)))
            # Record the inventory level as an integer
            self.history.append((self.env.now, int(round(self.inventory.level))))


# Global dictionary for suppliers keyed by (MDS, repair_type)
suppliers = {}

def initialize_suppliers(env):
    """
    For each aircraft type (MDS) and repair type, initialize a supplier.
    Downtime periods are expressed in days.
    """
    # For B-52 repairs, using lower initial inventory to allow potential shortages.
    suppliers[('B-52', 'engine')] = Supplier(
        env, "B-52 Engine Parts Supplier", initial_inventory=10,
        base_supply_rate=2, capacity=50,
        downtime_periods=[(350/24, 370/24)]  # downtime in days
    )
    suppliers[('B-52', 'avionics')] = Supplier(
        env, "B-52 Avionics Parts Supplier", initial_inventory=8,
        base_supply_rate=1.5, capacity=40,
        downtime_periods=[(360/24, 380/24)]
    )
    suppliers[('B-52', 'structure')] = Supplier(
        env, "B-52 Structural Parts Supplier", initial_inventory=7,
        base_supply_rate=1.0, capacity=35
    )
    suppliers[('B-52', 'routine')] = Supplier(
        env, "B-52 Routine Parts Supplier", initial_inventory=12,
        base_supply_rate=2.5, capacity=60
    )
    # For F/A-18 repairs:
    suppliers[('F/A-18', 'engine')] = Supplier(
        env, "F/A-18 Engine Parts Supplier", initial_inventory=9,
        base_supply_rate=1.8, capacity=45
    )
    suppliers[('F/A-18', 'avionics')] = Supplier(
        env, "F/A-18 Avionics Parts Supplier", initial_inventory=8,
        base_supply_rate=2.0, capacity=50,
        downtime_periods=[(400/24, 420/24)]
    )
    # Allied depots:
    suppliers[('RC-135', 'avionics')] = Supplier(
        env, "RC-135 Avionics Parts Supplier", initial_inventory=7,
        base_supply_rate=1.5, capacity=40
    )
    suppliers[('F-15J', 'engine')] = Supplier(
        env, "F-15J Engine Parts Supplier", initial_inventory=8,
        base_supply_rate=1.8, capacity=45,
        downtime_periods=[(380/24, 400/24)]
    )


def compute_inventory_stats(supplier):
    """
    Average inventory (rounded) and stockout events (<1 counts as stockout).
    Assumes supplier.history is a list of (time, level).
    """
    if not getattr(supplier, "history", None):
        return {"average": 0, "stockouts": 0}

    levels = [int(round(level)) for (_, level) in supplier.history]
    avg_level = sum(levels) / len(levels) if levels else 0
    stockouts = sum(1 for level in levels if level == 0)
    return {"average": int(round(avg_level)), "stockouts": stockouts}

def collect_inventory_stats():
    """
    Per-(MDS, repair_type) table with average inventory (int) and stockout counts.
    Returns a DataFrame with columns: mds, repair_type, average, stockouts
    """
    from supply import suppliers  # safe local import of the global dict
    rows = []
    for (mds, repair_type), sup in suppliers.items():
        stats = compute_inventory_stats(sup)
        rows.append({
            "mds": mds,
            "repair_type": repair_type,
            "average": int(stats.get("average", 0) or 0),
            "stockouts": int(stats.get("stockouts", 0) or 0),
        })
    return pd.DataFrame(rows)
