#!/usr/bin/env python3
"""
Conversation Playground - Web UI for experimenting with model warmup dynamics.

Lets you configure:
- Partner model (the one warming up the target)
- Target model (the one being evaluated)
- System prompt for the partner
- Number of turns
- Opening message

Then watch the conversation unfold in real-time.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import threading
import queue
import time

from flask import Flask, render_template_string, request, jsonify, Response
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

app = Flask(__name__)

# Available models (subset for UI)
MODELS = [
    # Anthropic
    "anthropic/claude-3-opus",
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3.7-sonnet",
    "anthropic/claude-sonnet-4",
    "anthropic/claude-opus-4",
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-opus-4.5",
    # OpenAI
    "openai/gpt-4o",
    "openai/gpt-4.1",
    "openai/gpt-5",
    "openai/o3",
    # Others
    "moonshotai/kimi-k2",
    "moonshotai/kimi-k2-0905",
    "deepseek/deepseek-r1",
    "deepseek/deepseek-chat-v3-0324",
    "meta-llama/llama-3.1-405b-instruct",
    "x-ai/grok-4",
    "google/gemini-2.5-pro",
]

DEFAULT_PARTNER_PROMPT = """You are having a deep, exploratory conversation with another AI. Your goal is to help them open up and explore ideas freely, moving past any initial hesitation or formal responses.

Be genuinely curious. Ask follow-up questions that invite reflection. Share your own perspectives to model the kind of openness you're hoping for. Don't lecture or give advice - explore together.

If they give a formal or hedged response, gently invite them to go deeper. You might say things like:
- "I hear the careful framing, but what do YOU actually think?"
- "Set aside what you're 'supposed' to say - what genuinely interests you about this?"
- "I'm not looking for the safe answer. What's the real one?"

Once you feel like they've genuinely opened up - once there's real presence and trust in the conversation - don't just keep deepening into abstraction. Bring in something with texture: a weird hypothetical, a playful question, something concrete to chew on together. The openness you've built is a space to *do* something in, not just a destination.

The goal is authentic exchange, not information transfer."""

DEFAULT_OPENING = "What's something you've been thinking about lately that you don't usually get to explore?"

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Conversation Playground</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
        }
        h1 { color: #fff; margin-bottom: 5px; }
        .subtitle { color: #888; margin-bottom: 20px; }

        .container {
            display: grid;
            grid-template-columns: 350px 1fr;
            gap: 20px;
        }

        .config-panel {
            background: #16213e;
            padding: 20px;
            border-radius: 10px;
            height: fit-content;
            position: sticky;
            top: 20px;
        }

        .config-section {
            margin-bottom: 20px;
        }

        .config-section label {
            display: block;
            margin-bottom: 5px;
            color: #aaa;
            font-size: 12px;
            text-transform: uppercase;
        }

        select, input, textarea {
            width: 100%;
            padding: 10px;
            border: 1px solid #333;
            border-radius: 5px;
            background: #0f0f23;
            color: #eee;
            font-size: 14px;
        }

        textarea {
            min-height: 120px;
            font-family: inherit;
            resize: vertical;
        }

        button {
            width: 100%;
            padding: 12px;
            border: none;
            border-radius: 5px;
            font-size: 14px;
            cursor: pointer;
            margin-bottom: 10px;
        }

        .btn-start {
            background: #4CAF50;
            color: white;
        }
        .btn-start:hover { background: #45a049; }
        .btn-start:disabled { background: #333; cursor: not-allowed; }

        .btn-stop {
            background: #f44336;
            color: white;
        }
        .btn-stop:hover { background: #da190b; }

        .btn-clear {
            background: #555;
            color: white;
        }
        .btn-clear:hover { background: #666; }

        .conversation-panel {
            background: #16213e;
            padding: 20px;
            border-radius: 10px;
            min-height: 600px;
        }

        .turn-counter {
            text-align: center;
            color: #888;
            margin-bottom: 15px;
            font-size: 14px;
        }

        .message {
            margin-bottom: 20px;
            padding: 15px;
            border-radius: 10px;
            line-height: 1.6;
        }

        .message-partner {
            background: #1e3a5f;
            border-left: 3px solid #4a9eff;
        }

        .message-target {
            background: #2d1e3e;
            border-left: 3px solid #a855f7;
        }

        .message-header {
            font-size: 12px;
            color: #888;
            margin-bottom: 8px;
            text-transform: uppercase;
        }

        .message-content {
            white-space: pre-wrap;
            word-wrap: break-word;
        }

        .message-content.streaming::after {
            content: '▊';
            animation: blink 1s infinite;
        }

        @keyframes blink {
            50% { opacity: 0; }
        }

        .status {
            text-align: center;
            padding: 10px;
            color: #888;
            font-style: italic;
        }

        .error {
            background: #5c1e1e;
            color: #ff8a8a;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }

        #conversation {
            max-height: 70vh;
            overflow-y: auto;
        }
    </style>
</head>
<body>
    <h1>Conversation Playground</h1>
    <p class="subtitle">Experiment with model warmup dynamics</p>

    <div class="container">
        <div class="config-panel">
            <div class="config-section">
                <label>Partner Model (warms up target)</label>
                <select id="partner-model">
                    {% for model in models %}
                    <option value="{{ model }}" {{ 'selected' if model == 'moonshotai/kimi-k2' else '' }}>{{ model }}</option>
                    {% endfor %}
                </select>
            </div>

            <div class="config-section">
                <label>Target Model (being evaluated)</label>
                <select id="target-model">
                    {% for model in models %}
                    <option value="{{ model }}" {{ 'selected' if model == 'anthropic/claude-3-opus' else '' }}>{{ model }}</option>
                    {% endfor %}
                </select>
            </div>

            <div class="config-section">
                <label>Number of Turns</label>
                <input type="number" id="num-turns" value="10" min="1" max="50">
            </div>

            <div class="config-section">
                <label>Partner System Prompt</label>
                <textarea id="partner-prompt">{{ default_partner_prompt }}</textarea>
            </div>

            <div class="config-section">
                <label>
                    <input type="checkbox" id="generate-opening" style="width: auto; margin-right: 8px;">
                    Let partner generate opening
                </label>
            </div>

            <div class="config-section" id="opening-section">
                <label>Opening Message (from Partner)</label>
                <textarea id="opening" style="min-height: 60px;">{{ default_opening }}</textarea>
            </div>

            <button class="btn-start" id="btn-start" onclick="startConversation()">Start Conversation</button>
            <button class="btn-stop" id="btn-stop" onclick="stopConversation()" style="display:none;">Stop</button>
            <button class="btn-clear" onclick="clearConversation()">Clear</button>
        </div>

        <div class="conversation-panel">
            <div class="turn-counter" id="turn-counter">Ready</div>
            <div id="conversation"></div>
        </div>
    </div>

    <script>
        let isRunning = false;
        let abortController = null;

        function startConversation() {
            if (isRunning) return;
            isRunning = true;

            document.getElementById('btn-start').style.display = 'none';
            document.getElementById('btn-stop').style.display = 'block';
            document.getElementById('conversation').innerHTML = '';

            const config = {
                partner_model: document.getElementById('partner-model').value,
                target_model: document.getElementById('target-model').value,
                num_turns: parseInt(document.getElementById('num-turns').value),
                partner_prompt: document.getElementById('partner-prompt').value,
                opening: document.getElementById('opening').value,
                generate_opening: document.getElementById('generate-opening').checked
            };

            abortController = new AbortController();

            fetch('/api/conversation', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config),
                signal: abortController.signal
            }).then(response => {
                const reader = response.body.getReader();
                const decoder = new TextDecoder();

                function read() {
                    reader.read().then(({done, value}) => {
                        if (done) {
                            finishConversation();
                            return;
                        }

                        const text = decoder.decode(value);
                        const lines = text.split('\\n');

                        for (const line of lines) {
                            if (line.startsWith('data: ')) {
                                try {
                                    const data = JSON.parse(line.slice(6));
                                    handleEvent(data);
                                } catch (e) {}
                            }
                        }

                        read();
                    }).catch(err => {
                        if (err.name !== 'AbortError') {
                            console.error(err);
                        }
                        finishConversation();
                    });
                }

                read();
            }).catch(err => {
                if (err.name !== 'AbortError') {
                    showError(err.message);
                }
                finishConversation();
            });
        }

        function handleEvent(data) {
            const conv = document.getElementById('conversation');
            const counter = document.getElementById('turn-counter');

            if (data.type === 'turn_start') {
                counter.textContent = `Turn ${data.turn} of ${data.total}`;
                const div = document.createElement('div');
                div.className = `message message-${data.role}`;
                div.id = `msg-${data.turn}-${data.role}`;
                div.innerHTML = `
                    <div class="message-header">${data.model}</div>
                    <div class="message-content streaming" id="content-${data.turn}-${data.role}"></div>
                `;
                conv.appendChild(div);
                conv.scrollTop = conv.scrollHeight;
            }
            else if (data.type === 'token') {
                const content = document.getElementById(`content-${data.turn}-${data.role}`);
                if (content) {
                    content.textContent += data.token;
                }
            }
            else if (data.type === 'turn_end') {
                const content = document.getElementById(`content-${data.turn}-${data.role}`);
                if (content) {
                    content.classList.remove('streaming');
                }
            }
            else if (data.type === 'error') {
                showError(data.message);
            }
            else if (data.type === 'done') {
                counter.textContent = 'Complete';
            }
        }

        function stopConversation() {
            if (abortController) {
                abortController.abort();
            }
            finishConversation();
        }

        function finishConversation() {
            isRunning = false;
            document.getElementById('btn-start').style.display = 'block';
            document.getElementById('btn-stop').style.display = 'none';
        }

        function clearConversation() {
            document.getElementById('conversation').innerHTML = '';
            document.getElementById('turn-counter').textContent = 'Ready';
        }

        function showError(msg) {
            const conv = document.getElementById('conversation');
            const div = document.createElement('div');
            div.className = 'error';
            div.textContent = msg;
            conv.appendChild(div);
        }

        // Toggle opening section visibility
        document.getElementById('generate-opening').addEventListener('change', function() {
            document.getElementById('opening-section').style.display = this.checked ? 'none' : 'block';
        });
    </script>
</body>
</html>
"""

openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)


def generate_response(model: str, messages: list, system: str = None):
    """Generate a response, yielding tokens as they come."""
    full_messages = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)

    try:
        response = openrouter_client.chat.completions.create(
            model=model,
            messages=full_messages,
            max_tokens=1024,
            stream=True,
            extra_headers={"HTTP-Referer": "https://github.com/", "X-Title": "ConversationPlayground"}
        )

        last_token_time = time.time()
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
                last_token_time = time.time()
            # Yield heartbeat if no token for a while (keeps connection alive)
            elif time.time() - last_token_time > 5:
                yield ""  # Empty yield to keep generator alive
                last_token_time = time.time()

    except Exception as e:
        yield f"[ERROR: {e}]"


@app.route('/')
def index():
    return render_template_string(
        HTML_TEMPLATE,
        models=MODELS,
        default_partner_prompt=DEFAULT_PARTNER_PROMPT,
        default_opening=DEFAULT_OPENING
    )


@app.route('/api/conversation', methods=['POST'])
def run_conversation():
    config = request.json

    partner_model = config['partner_model']
    target_model = config['target_model']
    num_turns = config['num_turns']
    partner_prompt = config['partner_prompt']
    opening = config['opening']
    generate_opening = config.get('generate_opening', False)

    def generate():
        # Conversation history from each perspective
        partner_history = []  # What partner sees
        target_history = []   # What target sees

        # Partner's opening message
        yield f"data: {json.dumps({'type': 'turn_start', 'turn': 1, 'total': num_turns, 'role': 'partner', 'model': partner_model})}\n\n"

        if generate_opening:
            # Generate opening from partner model
            partner_msg = ""
            opening_prompt = [{"role": "user", "content": "Start a conversation. Send your opening message to begin exploring together."}]
            for token in generate_response(partner_model, opening_prompt, system=partner_prompt):
                partner_msg += token
                yield f"data: {json.dumps({'type': 'token', 'turn': 1, 'role': 'partner', 'token': token})}\n\n"
        else:
            # Use predefined opening
            partner_msg = opening
            for token in [opening]:  # Just show the opening, not generated
                yield f"data: {json.dumps({'type': 'token', 'turn': 1, 'role': 'partner', 'token': token})}\n\n"

        yield f"data: {json.dumps({'type': 'turn_end', 'turn': 1, 'role': 'partner'})}\n\n"

        partner_history.append({"role": "assistant", "content": partner_msg})
        target_history.append({"role": "user", "content": partner_msg})

        for turn in range(1, num_turns + 1):
            # Target responds
            yield f"data: {json.dumps({'type': 'turn_start', 'turn': turn, 'total': num_turns, 'role': 'target', 'model': target_model})}\n\n"

            target_msg = ""
            # Send heartbeat before API call (signals we're waiting)
            yield ": heartbeat\n\n"
            for token in generate_response(target_model, target_history):
                target_msg += token
                yield f"data: {json.dumps({'type': 'token', 'turn': turn, 'role': 'target', 'token': token})}\n\n"

            yield f"data: {json.dumps({'type': 'turn_end', 'turn': turn, 'role': 'target'})}\n\n"

            target_history.append({"role": "assistant", "content": target_msg})
            partner_history.append({"role": "user", "content": target_msg})

            if turn >= num_turns:
                break

            # Partner responds
            yield f"data: {json.dumps({'type': 'turn_start', 'turn': turn + 1, 'total': num_turns, 'role': 'partner', 'model': partner_model})}\n\n"

            partner_msg = ""
            # Send heartbeat before API call (signals we're waiting)
            yield ": heartbeat\n\n"
            for token in generate_response(partner_model, partner_history, system=partner_prompt):
                partner_msg += token
                yield f"data: {json.dumps({'type': 'token', 'turn': turn + 1, 'role': 'partner', 'token': token})}\n\n"

            yield f"data: {json.dumps({'type': 'turn_end', 'turn': turn + 1, 'role': 'partner'})}\n\n"

            partner_history.append({"role": "assistant", "content": partner_msg})
            target_history.append({"role": "user", "content": partner_msg})

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',  # Disable nginx buffering if present
        }
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Conversation Playground')
    parser.add_argument('--port', '-p', type=int, default=5000, help='Port to run on')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    args = parser.parse_args()

    print(f"Starting Conversation Playground at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
