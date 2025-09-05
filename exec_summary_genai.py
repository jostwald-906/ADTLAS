import os
import openai
import json
import pandas as pd
from pydantic import BaseModel, Field, ValidationError

openai.api_key = os.getenv("OPENAI_API_KEY")

class ExecSummaryModel(BaseModel):
    total_tasks: int
    avg_wait_hours: float
    avg_service_hours: float
    depot_utilizations: dict
    inventory_availability: dict   # {"B-52 engine": {"average": int, "stockouts": int}, ...}
    recommendations: list[str]
    summary_text: str

def generate_exec_summary_genai(
    df_tasks: pd.DataFrame,
    depot_data: dict,
    sim_time: float,
    inventory_stats_df: pd.DataFrame | None = None   # <-- accept inv stats here
) -> str:
    total_tasks = len(df_tasks)
    avg_wait = df_tasks['wait_time'].mean() if total_tasks > 0 else 0.0
    avg_service = df_tasks['service_time'].mean() if total_tasks > 0 else 0.0

    depot_utilizations = {
        depot.name: round((depot.total_service_time / (sim_time * depot.capacity) * 100.0) if depot.capacity else 0.0, 1)
        for depot in depot_data.values()
    }

    # Build inventory dict from the passed DataFrame (no imports from supply.py)
    inv_avail = {}
    if isinstance(inventory_stats_df, pd.DataFrame) and not inventory_stats_df.empty:
        for _, r in inventory_stats_df.iterrows():
            key = f"{r['mds']} {r['repair_type']}"
            inv_avail[key] = {
                "average": int(r["average"]),
                "stockouts": int(r["stockouts"])
            }

    prompt = (
        "You are a defense logistics expert with deep experience in depot operations and supply chains. "
        "Provide an executive summary as a JSON object with keys: "
        "total_tasks, avg_wait_hours, avg_service_hours, depot_utilizations, inventory_availability, "
        "recommendations, summary_text. Return ONLY valid JSON.\n\n"
        f"Total tasks processed: {total_tasks}\n"
        f"Average wait time (hours): {avg_wait * 24:.2f}\n"
        f"Average service time (hours): {avg_service * 24:.2f}\n"
        "Depot Utilizations (percentages):\n"
    )
    for name, util in depot_utilizations.items():
        prompt += f"  - {name}: {util}\n"

    prompt += "\nInventory Availabilities (by MDS and repair type):\n"
    if inv_avail:
        for k, v in inv_avail.items():
            prompt += f"  - {k}: average = {v['average']}, stockouts = {v['stockouts']}\n"
    else:
        prompt += "  - n/a\n"

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or gpt-4 / gpt-4o
            messages=[
                {"role": "system", "content": "You are a highly experienced defense logistics strategist."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000,
        )
        content = resp.choices[0].message.content.strip()
        if content.startswith("```"):
            parts = content.split("```")
            if len(parts) >= 2:
                content = parts[1].strip()
        if content.lower().startswith("json"):
            content = content[4:].strip()

        if not content:
            return "Error: Empty response from OpenAI API."

        data = json.loads(content)
        # If the model omitted inventory, inject what we computed
        data.setdefault("inventory_availability", inv_avail)

        exec_summary = ExecSummaryModel(**data)
        return exec_summary.json(indent=2)

    except (ValidationError, json.JSONDecodeError) as e:
        return f"Error parsing executive summary: {e}\nRaw response: {content}"
    except Exception as e:
        return f"Error generating executive summary: {e}"
