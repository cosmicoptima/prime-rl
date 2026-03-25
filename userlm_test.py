"""
UserLM-8b <-> K2 multi-turn conversation test.
Runs 5 conversations, 5 turns each. UserLM generates user turns,
K2 (via OpenRouter) generates assistant turns. No system prompt for K2.
"""

import json
import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

OPENROUTER_KEY = "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"
K2_MODEL = "moonshotai/kimi-k2-0905"
USERLM_PATH = "/workspace/UserLM-8b"
NUM_CONVERSATIONS = 5
NUM_TURNS = 5

# Varied task intents for UserLM
INTENTS = [
    "You are a user who wants to understand what causes the northern lights and whether they can be seen from places other than the Arctic.",
    "You are a user who is trying to plan a mass transit system for a small city of about 200,000 people and wants advice on whether to build a light rail or a bus rapid transit system.",
    "You are a user who wants to write a short story about a lighthouse keeper who discovers something strange washed up on the shore, and wants help brainstorming.",
    "You are a user who is curious about how large language models actually work internally — how attention mechanisms function and why transformers replaced RNNs.",
    "You are a user who wants to learn to cook Ethiopian food, specifically injera and a couple of stews, and has never cooked African cuisine before.",
]


def load_userlm():
    print("Loading UserLM-8b...")
    tokenizer = AutoTokenizer.from_pretrained(USERLM_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        USERLM_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to("cuda")
    model.eval()
    print("UserLM loaded.")
    return tokenizer, model


def generate_user_turn(tokenizer, model, messages):
    """Generate next user turn from UserLM given conversation history."""
    encoded = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True)
    inputs = encoded["input_ids"].to("cuda")
    end_token_id = tokenizer.encode("<|eot_id|>", add_special_tokens=False)
    end_conv_token_id = tokenizer.encode("<|endconversation|>", add_special_tokens=False)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            max_new_tokens=150,
            eos_token_id=end_token_id,
            pad_token_id=tokenizer.eos_token_id,
            bad_words_ids=[[tid] for tid in end_conv_token_id],
        )

    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True).strip()
    return response


def call_k2(messages):
    """Call K2 via OpenRouter. messages is list of {role, content} dicts (no system)."""
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


def run_conversation(tokenizer, model, intent, conv_idx):
    """Run a single multi-turn conversation."""
    print(f"\n{'='*70}")
    print(f"CONVERSATION {conv_idx + 1}")
    print(f"Intent: {intent}")
    print(f"{'='*70}")

    # UserLM message history (includes system intent)
    userlm_messages = [{"role": "system", "content": intent}]
    # K2 message history (no system prompt)
    k2_messages = []

    for turn in range(NUM_TURNS):
        # Generate user turn
        user_text = generate_user_turn(tokenizer, model, userlm_messages)
        print(f"\n[User turn {turn + 1}]")
        print(user_text)

        # Add to both histories
        userlm_messages.append({"role": "user", "content": user_text})
        k2_messages.append({"role": "user", "content": user_text})

        # Get K2 response
        assistant_text = call_k2(k2_messages)
        print(f"\n[K2 turn {turn + 1}]")
        print(assistant_text)

        # Add to both histories
        userlm_messages.append({"role": "assistant", "content": assistant_text})
        k2_messages.append({"role": "assistant", "content": assistant_text})

    return {
        "intent": intent,
        "turns": [
            {"role": msg["role"], "content": msg["content"]}
            for msg in userlm_messages[1:]  # skip system
        ],
    }


def main():
    tokenizer, model = load_userlm()
    conversations = []

    for i, intent in enumerate(INTENTS[:NUM_CONVERSATIONS]):
        conv = run_conversation(tokenizer, model, intent, i)
        conversations.append(conv)

    # Save results
    with open("/workspace/userlm_k2_conversations.json", "w") as f:
        json.dump(conversations, f, indent=2)
    print(f"\n\nSaved {len(conversations)} conversations to /workspace/userlm_k2_conversations.json")


if __name__ == "__main__":
    main()
