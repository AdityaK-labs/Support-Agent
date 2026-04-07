import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from openenv.models import Action, Observation
import os
from dotenv import load_dotenv

load_dotenv()

class AgentPolicy:
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        
    def act(self, obs: Observation) -> Action:
        sys_prompt = (
            "You are an AI BPO agent fixing customer issues. "
            "Output JSON with `action_type`, `team` (opt), `response` (opt). "
            "action_type is one of: [classify, assign, respond, refund, escalate]."
        )
        
        context = {
            "ticket": obs.model_dump(exclude={"history"}),
            "history": obs.history
        }
        
        user_prompt = f"Ticket Details:\n{json.dumps(context, indent=2)}\n\nWhat is your action JSON?"
        
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        try:
            outputs = self.model.generate(**inputs, max_new_tokens=250, temperature=0.2, do_sample=True)
            text = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            text = text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            raw = json.loads(text.strip())
            return Action(**raw)
        except Exception as e:
            print("Fallback act due to:", e)
            return Action(action_type="classify")
