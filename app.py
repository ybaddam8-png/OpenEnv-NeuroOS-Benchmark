import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from person_a.environment import NeuroInclusiveEnv

app = FastAPI()
env = NeuroInclusiveEnv(seed=42)

# The bot will send a list of commands here
class StepAction(BaseModel):
    actions: list

# 1. The Ping Requirement (Must return 200)
@app.get("/")
def health_check():
    return {"status": "ok", "message": "NEXUS NeuroOS OpenEnv API is running"}

# 2. The Reset Endpoint
@app.post("/reset")
@app.get("/reset")
def reset_env():
    obs, info = env.reset()
    return {"observation": obs, "info": info}

# 3. The Step Endpoint
@app.post("/step")
def step_env(payload: StepAction):
    obs, reward, term, trunc, info = env.step(payload.actions)
    return {
        "observation": obs, 
        "reward": float(reward), 
        "terminated": term, 
        "truncated": trunc, 
        "info": info
    }

# 4. The State Endpoint
@app.get("/state")
def get_state():
    # Assuming obs is easily retrievable, or we just reset to get state
    return {"status": "active"}

if __name__ == "__main__":
    # Hugging Face Spaces strictly require port 7860
    uvicorn.run(app, host="0.0.0.0", port=7860)