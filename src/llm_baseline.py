# src/llm_baseline.py

import os
import json
from openai import OpenAI

# Create OpenAI client (uses OPENAI_API_KEY from environment)
client = OpenAI()

SYSTEM_PROMPT = """
You are an expert governance & compliance analyst for IDB projects.
Given a project paragraph and (optionally) model predictions / rule hits,
you must return a structured JSON assessment.

Your job is to:
- Identify the most relevant policy / thematic labels.
- Assess overall risk level (High / Medium / Low).
- Explain WHY in clear, concise language.

You MUST return a valid JSON object with this exact structure:
{
  "labels": [ "label_1", "label_2", ... ],
  "risk_level": "High | Medium | Low",
  "explanation": "short paragraph explanation"
}
"""

def generate_llm_explanation(
    paragraph: str,
    sentence_baseline_labels=None,
    transformer_labels=None,
    rule_hits=None,
    model_name: str = "gpt-4o-mini",
):
    """
    Call the OpenAI Chat Completions API (new v1 client) to get a JSON explanation.

    sentence_baseline_labels, transformer_labels, rule_hits are optional lists.
    """

    if sentence_baseline_labels is None:
        sentence_baseline_labels = []
    if transformer_labels is None:
        transformer_labels = []
    if rule_hits is None:
        rule_hits = []

    user_prompt = f"""
Paragraph:
\"\"\"{paragraph}\"\"\"

Sentence encoder baseline labels:
{sentence_baseline_labels}

Fine-tuned transformer labels:
{transformer_labels}

Rule-engine hits:
{rule_hits}

Using all of this, produce the final JSON as described.
Remember: respond with ONLY a JSON object, no extra text.
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0.0,
    )

    content = response.choices[0].message.content.strip()

    # Try to parse JSON; if it fails, wrap raw text
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = {
            "labels": [],
            "risk_level": "Unknown",
            "explanation": "LLM did not return valid JSON.",
            "raw_response": content,
        }

    return data


if __name__ == "__main__":
    # Simple test paragraph (you can change this)
    paragraph = """
    The program aims to improve urban water and drainage systems,
    strengthen institutional capacity for utility management,
    and support climate resilience and flood risk management in mid-sized cities.
    """

    # Fake predictions to show how it will be used later with your ML models
    sentence_labels = ["urban_water_and_drainage_systems"]
    transformer_labels = [
        "urban_water_and_drainage_systems",
        "climate_resilience_and_flood_drought",
        "utility_management_and_capacity",
    ]
    rule_hits = ["flood_risk", "critical_infrastructure"]

    result = generate_llm_explanation(
        paragraph=paragraph,
        sentence_baseline_labels=sentence_labels,
        transformer_labels=transformer_labels,
        rule_hits=rule_hits,
    )

    print(json.dumps(result, indent=2))
