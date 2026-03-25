"""
USP (User Simulator with Implicit Profiles) <-> K2 multi-turn conversation test.
5 conversations, 5 turns each. USP generates user turns with profile conditioning,
K2 (via OpenRouter) generates assistant turns. No system prompt for K2.
"""

import json
import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

OPENROUTER_KEY = "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"
K2_MODEL = "moonshotai/kimi-k2-0905"
USP_PATH = "/workspace/USP"
NUM_CONVERSATIONS = 5
NUM_TURNS = 5

PROFILE_TEMPLATE = (
    "You are engaging in a conversation with an AI assistant. "
    "Your profile is: \n{profile}\n "
    "You can say anything you want, either based on the profile or something brand new.\n\n"
)

# Profiles that correspond to the same topics as the UserLM test
PROFILES = [
    {
        "profile": (
            "You are a curious amateur astronomer who recently moved to northern Canada. "
            "You've seen the northern lights once and were captivated, and now you want to "
            "understand the science behind them. You wonder if they can be seen from other places too.\n\n"
            "You ask clear, direct questions and like to follow up on details. "
            "You're enthusiastic but not an expert — you want explanations in plain language."
        ),
    },
    {
        "profile": (
            "You are a city planner working for a small city of about 200,000 people. "
            "Your mayor has asked you to evaluate whether to build a light rail or bus rapid transit system. "
            "You have funding and land available but need to make the right choice.\n\n"
            "You are analytical and methodical. You like to weigh trade-offs and ask about "
            "costs, ridership projections, and comparable cities. You push back on vague advice."
        ),
    },
    {
        "profile": (
            "You are an aspiring fiction writer working on your first short story collection. "
            "Your current idea is about a lighthouse keeper who discovers something strange washed "
            "up on the shore. You have the setting but need help brainstorming the plot.\n\n"
            "You're creative and collaborative. You like to bounce ideas back and forth, "
            "build on suggestions, and aren't afraid to reject ideas that don't feel right."
        ),
    },
    {
        "profile": (
            "You are a computer science graduate student who has used transformer-based models "
            "but wants to understand their internals more deeply — specifically how attention "
            "mechanisms work and why transformers replaced RNNs.\n\n"
            "You have a solid math background and prefer precise technical explanations. "
            "You ask pointed follow-up questions and like worked examples."
        ),
    },
    {
        "profile": (
            "You are a home cook who is confident with European and Asian cuisines but has "
            "never cooked African food. You want to learn to make Ethiopian injera and a couple "
            "of stews for a dinner party next month.\n\n"
            "You're practical and focused on execution — you want actual recipes, ingredient lists, "
            "and tips for what can go wrong. You ask about substitutions for hard-to-find ingredients."
        ),
    },
]


def load_usp():
    print("Loading USP...")
    tokenizer = AutoTokenizer.from_pretrained(
        USP_PATH, padding_side="left", trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        USP_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    print("USP loaded.")
    return tokenizer, model


def generate_user_turn(tokenizer, model, messages):
    """Generate next user turn from USP given conversation history."""
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        input_text, return_tensors="pt", truncation=True, max_length=4096
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=256,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return response


def call_k2(messages):
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": K2_MODEL,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1024,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def run_conversation(tokenizer, model, profile_info, conv_idx):
    system_prompt = PROFILE_TEMPLATE.format(profile=profile_info["profile"])

    print(f"\n{'='*70}")
    print(f"CONVERSATION {conv_idx + 1}")
    print(f"Profile: {profile_info['profile'][:100]}...")
    print(f"{'='*70}")

    # USP message history (with system profile)
    usp_messages = [{"role": "system", "content": system_prompt}]
    # K2 message history (no system prompt)
    k2_messages = []

    for turn in range(NUM_TURNS):
        # Generate user turn via USP
        user_text = generate_user_turn(tokenizer, model, usp_messages)
        print(f"\n[User turn {turn + 1}]")
        print(user_text)

        usp_messages.append({"role": "user", "content": user_text})
        k2_messages.append({"role": "user", "content": user_text})

        # Get K2 response
        assistant_text = call_k2(k2_messages)
        print(f"\n[K2 turn {turn + 1}]")
        print(assistant_text)

        usp_messages.append({"role": "assistant", "content": assistant_text})
        k2_messages.append({"role": "assistant", "content": assistant_text})

    return {
        "profile": profile_info["profile"],
        "turns": [
            {"role": msg["role"], "content": msg["content"]}
            for msg in usp_messages[1:]
        ],
    }


def main():
    tokenizer, model = load_usp()
    conversations = []

    for i, profile in enumerate(PROFILES[:NUM_CONVERSATIONS]):
        conv = run_conversation(tokenizer, model, profile, i)
        conversations.append(conv)

    with open("/workspace/usp_k2_conversations.json", "w") as f:
        json.dump(conversations, f, indent=2)
    print(f"\n\nSaved {len(conversations)} conversations to /workspace/usp_k2_conversations.json")


if __name__ == "__main__":
    main()
