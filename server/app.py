import os
import uvicorn
import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from openenv.env import SupportEnv
from openenv.models import Action

app = FastAPI(title="OpenEnv Support Agent API")

# Global environment instance for the UI/API
env = SupportEnv(task_name="easy")
env.reset()

# --- FastAPI Routes ---

class ActionRequest(BaseModel):
    action_type: str
    team: str = None
    response: str = None

@app.get("/api/state")
@app.get("/state")
def get_state():
    return env.state()

@app.post("/api/reset")
@app.post("/reset")
def reset_env(task_name: str = "easy"):
    res = env.reset(task_name=task_name)
    return res

@app.post("/api/step")
@app.post("/step")
def step_env(action_req: ActionRequest):
    act = Action(**action_req.model_dump())
    res = env.step(act)
    return res

# --- Gradio UI ---

def ui_reset(task_choice):
    mapping = {"Easy (Classification)": "easy", "Medium (Assignment)": "medium", "Hard (Resolution)": "hard"}
    task_name = mapping.get(task_choice, "easy")
    res = env.reset(task_name)
    return f"Environment reset to {task_name}.", res.observation.model_dump_json(indent=2)

def ui_step(a_type, t_name, resp_text):
    if env.done:
        return "Episode is already done! Please reset.", env.state().model_dump_json(indent=2), ""

    try:
        act = Action(action_type=a_type, team=t_name if t_name else None, response=resp_text if resp_text else None)
        res = env.step(act)
        feedback = f"Reward: {res.reward}\nDone: {res.done}\nInfo: {res.info}"
        return feedback, res.observation.model_dump_json(indent=2), "-> " + "\n-> ".join(env.history)
    except Exception as e:
        obs_json = env.get_current_observation().model_dump_json(indent=2) if env.current_scenario else "{}"
        return f"Error: {e}", obs_json, ""

with gr.Blocks(title="OpenEnv Support Agent") as demo:
    gr.Markdown("# OpenEnv-Based Autonomous Customer Support Agent")
    gr.Markdown("This interface lets you manually step the environment or view the state. For agent evaluation, use `inference.py`.")

    with gr.Row():
        task_dropdown = gr.Dropdown(
            choices=["Easy (Classification)", "Medium (Assignment)", "Hard (Resolution)"],
            value="Easy (Classification)", label="Select Task Level"
        )
        reset_btn = gr.Button("Reset Environment")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Observation (Current State)")
            obs_box = gr.Code(language="json", label="Observation JSON")
            history_box = gr.Textbox(label="Action History", lines=5, interactive=False)

        with gr.Column():
            gr.Markdown("### Take Action")
            act_type = gr.Dropdown(choices=["classify", "assign", "respond", "refund", "escalate"], value="classify", label="Action Type")
            act_team = gr.Textbox(label="Team (optional)")
            act_resp = gr.Textbox(label="Response text (optional)")
            step_btn = gr.Button("Step Environment", variant="primary")
            feedback_box = gr.Textbox(label="Reward / Step Info", interactive=False)

    reset_btn.click(ui_reset, inputs=[task_dropdown], outputs=[feedback_box, obs_box])
    step_btn.click(ui_step, inputs=[act_type, act_team, act_resp], outputs=[feedback_box, obs_box, history_box])

demo.queue()
app = gr.mount_gradio_app(app, demo, path="/")

def main():
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=True)

if __name__ == "__main__":
    main()
