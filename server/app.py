import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from person_a.environment import NeuroInclusiveEnv

app = FastAPI()
env = NeuroInclusiveEnv(seed=42)

# The bot will send a list of commands here
class StepAction(BaseModel):
    actions: list

class ResetOptions(BaseModel):
    task_name: str | None = None
    difficulty: str | None = None

# 1. The Ping Requirement (Must return 200)
@app.get("/")
def health_check():
    return {"status": "ok", "message": "NEXUS NeuroOS OpenEnv API is running"}

# 2. The Reset Endpoint
@app.post("/reset")
def reset_env_post(payload: ResetOptions = None):
    opts = payload.model_dump(exclude_unset=True) if payload else {}
    obs, info = env.reset(options=opts)
    return {"observation": obs, "info": info}

@app.get("/reset")
def reset_env_get():
    obs, info = env.reset(options={})
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
    return env.state()

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()