"""Simple Gradio chat UI for vLLM OpenAI-compatible server."""

import gradio as gr
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
MODEL = "cosmicoptima/diksha-r3-step50"


def chat(message, history):
    messages = []
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=1.0,
        stream=True,
    )

    partial = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            partial += chunk.choices[0].delta.content
            yield partial


demo = gr.ChatInterface(chat, title="Diksha R3 Step 50")
demo.launch(server_name="0.0.0.0", server_port=7860)
