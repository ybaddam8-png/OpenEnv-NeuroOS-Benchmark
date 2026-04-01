import os
import json
from openai import OpenAI
from person_a.environment import NeuroInclusiveEnv

# CRITICAL FIX: Use the exact environment variables required by the judges
client = OpenAI(
    base_url=os.getenv("API_BASE_URL"),
    api_key=os.getenv("HF_TOKEN")
)
model_name = os.getenv("MODEL_NAME")

def get_llm_action(observation):
    """Asks the LLM to act as a Neuro-Inclusive Design Expert."""
    prompt = f"""
    You are an expert in Neuro-Inclusive UI Design. 
    Current UI State (JSON): {json.dumps(observation['dom'])}
    User Biometric Stress: {json.dumps(observation['biometrics'])}
    Instructions: {observation['instructions']}

    Goal: Mutate the DOM to reduce sensory load and improve accessibility.
    Output ONLY a JSON list of commands. Example:
    [{{"op": "set_contrast", "node_id": "btn_1", "value": 7.0}}]
    """
    
    # CRITICAL FIX: Pass the MODEL_NAME variable here
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        response_format={ "type": "json_object" }
    )
    return json.loads(response.choices[0].message.content)

def main():
    env = NeuroInclusiveEnv(seed=42)
    obs, info = env.reset()
    
    for _ in range(3): # Max steps
        actions = get_llm_action(obs)
        obs, reward, term, trunc, info = env.step(actions)
        print(f"Step Reward: {reward} | Current Score: {info['grade']['score']}")
        if term or trunc: break

if __name__ == "__main__":
    main()