import os
import openai
import json
import pandas as pd
from pydantic import BaseModel, Field, ValidationError


# Load API key from environment (ensure your .env file is loaded, e.g., using python-dotenv)
openai.api_key = os.getenv("OPENAI_API_KEY")

class ExecSummaryModel(BaseModel):
    total_tasks: int = Field(..., description="Total number of tasks processed")
    avg_wait_hours: float = Field(..., description="Average wait time in hours")
    avg_service_hours: float = Field(..., description="Average service time in hours")
    depot_utilizations: dict = Field(..., description="Mapping of depot names to their utilization percentages")
    inventory_availability: dict = Field(..., description="Mapping of each MDS and repair type to inventory stats (average available parts and stockout count)")
    recommendations: list[str] = Field(..., description="List of strategic recommendations")
    summary_text: str = Field(..., description="A concise textual summary of key insights")

def generate_exec_summary_genai(df_tasks: pd.DataFrame, depot_data: dict, sim_time: float) -> str:
    total_tasks = len(df_tasks)
    avg_wait = df_tasks['wait_time'].mean() if total_tasks > 0 else 0
    avg_service = df_tasks['service_time'].mean() if total_tasks > 0 else 0

    depot_utilizations = {
        depot.name: round(depot.total_service_time / (sim_time * depot.capacity) * 100, 1)
        for depot in depot_data.values()
    }

    # Build inventory stats from suppliers: for each supplier (keyed by MDS and repair type),
    # compute average available parts and stockout events.
    inventory_stats = {}
    for key, supplier in suppliers.items():
        mds, repair_type = key
        inv_key = f"{mds} {repair_type}"
        stats = compute_inventory_stats(supplier)
        inventory_stats[inv_key] = stats

    # Build the prompt with detailed context
    prompt = (
        "You are a defense logistics expert with deep experience in maintenance depot operations, supply chain management, and inventory optimization. "
        "Analyze the following simulation metrics from a digital twin model of maintenance depots. The data is provided by Mission Design Series (MDS) and location, "
        "and includes detailed inventory statistics for critical parts.\n\n"
        "Please provide an executive summary as a JSON object with exactly these keys:\n"
        "  - total_tasks (int): Total number of maintenance tasks processed.\n"
        "  - avg_wait_hours (float): Average wait time in hours.\n"
        "  - avg_service_hours (float): Average service time in hours.\n"
        "  - depot_utilizations (object): Mapping of depot names to their utilization percentages.\n"
        "  - inventory_availability (object): For each MDS and repair type (e.g., 'B-52 engine'), provide an object with keys 'average' (an integer representing the average number of available parts) and 'stockouts' (the number of times inventory was 0).\n"
        "  - recommendations (list of strings): Strategic recommendations based on the data, addressing capacity, task routing, supplier resilience, and policy adjustments.\n"
        "  - summary_text (string): A concise narrative summary of key insights and actionable recommendations.\n\n"
        "Return ONLY the JSON object with all keys and values properly enclosed in quotes, with no additional commentary or markdown formatting.\n\n"
        f"Total tasks processed: {total_tasks}\n"
        f"Average wait time (hours): {avg_wait * 24:.2f}\n"
        f"Average service time (hours): {avg_service * 24:.2f}\n"
        "Depot Utilizations (percentages):\n"
    )
    for depot_name, util in depot_utilizations.items():
        prompt += f"  - {depot_name}: {util}\n"
    prompt += "\nInventory Availabilities (by MDS and repair type):\n"
    for inv_key, stats in inventory_stats.items():
        prompt += f"  - {inv_key}: average = {stats['average']:.2f}, stockouts = {stats['stockouts']}\n"
    prompt += (
        "\nBased on these metrics, provide a JSON object with the keys as described above. "
        "Ensure that your output is valid JSON with all necessary closing braces and quotes, and return only the JSON object."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            #model="gpt-4o",  # Change to your desired model (e.g., "gpt-3.5-turbo" or "gpt-4")
            messages=[
                {"role": "system", "content": "You are a highly experienced defense logistics strategist."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000,  # Increased to allow a longer response
        )
        content = response.choices[0].message.content.strip()
        # Remove markdown formatting if present
        if content.startswith("```"):
            parts = content.split("```")
            if len(parts) >= 2:
                content = parts[1].strip()
        # Remove any leading 'json' keyword (case-insensitive)
        if content.lower().startswith("json"):
            content = content[4:].strip()
            
        print("Raw API response content:")
        print(content)

        if not content:
            return "Error: Empty response from OpenAI API."
        data = json.loads(content)
        exec_summary = ExecSummaryModel(**data)
        return exec_summary.json(indent=2)
    except (ValidationError, json.JSONDecodeError) as e:
        return f"Error parsing executive summary: {e}\nRaw response: {content}"
    except Exception as e:
        return f"Error generating executive summary: {e}"

# For local testing:
if __name__ == '__main__':
    import numpy as np
    np.random.seed(42)
    sample_data = {
        'wait_time': np.random.exponential(scale=0.1, size=100),  # in days
        'service_time': np.random.exponential(scale=0.2, size=100),  # in days
        'arrival_time': np.sort(np.random.uniform(0, 30, 100))
    }
    sample_data['total_time'] = sample_data['wait_time'] + sample_data['service_time']
    df_sample = pd.DataFrame(sample_data)
    
    class DummyDepot:
        def __init__(self, name, total_service_time, capacity):
            self.name = name
            self.total_service_time = total_service_time
            self.capacity = capacity
    depot_sample = {
        'Depot1': DummyDepot("Tinker AFB", 1.0, 5),
        'Depot2': DummyDepot("FRCSW", 0.8, 4)
    }
    
    summary = generate_exec_summary_genai(df_sample, depot_sample, sim_time=30)
    print("Generated Executive Summary:\n", summary)

