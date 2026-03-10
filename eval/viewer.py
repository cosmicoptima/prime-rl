#!/usr/bin/env python3
"""
Unified Eval Viewer & Runner - Web UI for running and exploring evals.

Features:
- Run evals on API models or local checkpoints
- View results with example pairs for dimension correlations
- Switch between result files

Usage:
    python viewer.py                    # Start with empty state
    python viewer.py results/           # Load existing results
    python viewer.py --checkpoints-dir /path/to/checkpoints
"""

import argparse
import json
import os
import sys
import threading
import queue
import time
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template_string, jsonify, request, Response

app = Flask(__name__)

# Global state
RESULTS = {}  # {filename: data}
CURRENT_FILE = None
RESULTS_DIR = Path("eval/results")
CHECKPOINTS_DIR = None

# Running eval state
EVAL_STATUS = {
    "running": False,
    "model": None,
    "progress": [],
    "error": None,
}
EVAL_QUEUE = queue.Queue()

# Common API models
API_MODELS = [
    "anthropic/claude-sonnet-4",
    "anthropic/claude-opus-4",
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-opus-4.5",
    "openai/gpt-4o",
    "openai/gpt-4.1",
    "openai/gpt-5",
    "openai/o3",
    "moonshotai/kimi-k2-0905",
    "deepseek/deepseek-chat-v3-0324",
    "google/gemini-2.5-pro",
    "x-ai/grok-4",
    "meta-llama/llama-3.1-405b-instruct",
]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>EVAL//SYS</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0d0d0d;
            --bg-secondary: #141414;
            --bg-tertiary: #1a1a1a;
            --bg-elevated: #222;
            --border: #333;
            --border-bright: #444;
            --text-primary: #e5e5e5;
            --text-secondary: #888;
            --text-dim: #555;
            --accent: #a0a0a0;
            --accent-dim: #a0a0a020;
            --accent-green: #4ade80;
            --accent-green-dim: #4ade8025;
            --accent-red: #f87171;
            --accent-red-dim: #f8717125;
            --accent-amber: #fbbf24;
            --accent-amber-dim: #fbbf2425;
            --mono: 'JetBrains Mono', monospace;
            --sans: 'Space Grotesk', sans-serif;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: var(--mono);
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            font-size: 13px;
            line-height: 1.5;
        }

        ::selection {
            background: var(--text-primary);
            color: var(--bg-primary);
        }

        /* Header */
        .header {
            background: var(--bg-secondary);
            padding: 0 24px;
            height: 56px;
            display: flex;
            align-items: center;
            gap: 24px;
        }

        .logo {
            font-family: var(--sans);
            font-size: 18px;
            font-weight: 700;
            letter-spacing: 3px;
            color: var(--text-primary);
            text-transform: uppercase;
        }

        .header-controls {
            display: flex;
            align-items: center;
            gap: 16px;
            margin-left: auto;
        }

        .file-select, select, input[type="text"] {
            font-family: var(--mono);
            font-size: 12px;
            padding: 8px 12px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            color: var(--text-primary);
            outline: none;
            transition: all 0.15s;
        }

        .file-select:hover, select:hover, input[type="text"]:hover {
            border-color: var(--border-bright);
        }

        .file-select:focus, select:focus, input[type="text"]:focus {
            border-color: var(--text-secondary);
        }

        .status-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-secondary);
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--accent-green);
            animation: pulse 2s ease-in-out infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }

        /* Main Layout */
        .main-container {
            display: flex;
            min-height: calc(100vh - 56px);
        }

        /* Sidebar */
        .sidebar {
            width: 180px;
            background: var(--bg-secondary);
            padding: 16px 0;
            display: flex;
            flex-direction: column;
        }

        .sidebar-section {
            padding: 0 12px;
            margin-bottom: 20px;
        }

        .sidebar-label {
            font-size: 9px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-dim);
            margin-bottom: 8px;
            padding-left: 8px;
        }

        .nav-item {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 6px 8px;
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.1s;
            font-size: 11px;
        }

        .nav-item:hover {
            background: var(--bg-tertiary);
            color: var(--text-primary);
        }

        .nav-item.active {
            background: var(--bg-tertiary);
            color: var(--text-primary);
            border-left: 2px solid var(--text-primary);
        }

        .nav-item.disabled {
            opacity: 0.3;
            cursor: not-allowed;
        }

        .nav-item .nav-icon {
            font-size: 14px;
            width: 20px;
            text-align: center;
        }

        /* Content */
        .content {
            flex: 1;
            padding: 16px 20px;
            overflow-y: auto;
            max-height: calc(100vh - 56px);
            background: var(--bg-primary);
        }

        .page-title {
            font-family: var(--sans);
            font-size: 20px;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 16px;
        }

        /* Run Panel */
        .run-panel {
            background: var(--bg-secondary);
            padding: 16px;
            margin-bottom: 16px;
        }

        .panel-title {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: var(--text-secondary);
            margin-bottom: 20px;
        }

        .model-type-tabs {
            display: flex;
            gap: 4px;
            margin-bottom: 16px;
        }

        .model-type-tab {
            font-family: var(--mono);
            font-size: 11px;
            padding: 8px 16px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            color: var(--text-secondary);
            cursor: pointer;
            text-transform: uppercase;
            letter-spacing: 1px;
            transition: all 0.1s;
        }

        .model-type-tab:hover {
            background: var(--bg-elevated);
            color: var(--text-primary);
        }

        .model-type-tab.active {
            background: var(--text-primary);
            color: var(--bg-primary);
            border-color: var(--text-primary);
        }

        .form-row {
            display: flex;
            align-items: flex-start;
            gap: 12px;
            margin-bottom: 12px;
        }

        .form-row > label:first-child {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-dim);
            min-width: 80px;
            padding-top: 8px;
        }

        .form-row select, .form-row input[type="text"] {
            flex: 1;
            max-width: 400px;
        }

        .checkbox-label {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            cursor: pointer;
            font-size: 11px;
            color: var(--text-secondary);
            padding: 6px 10px;
            background: var(--bg-tertiary);
            transition: all 0.1s;
        }

        .checkbox-label:hover {
            color: var(--text-primary);
        }

        .checkbox-label:has(input:checked) {
            background: var(--bg-elevated);
            color: var(--text-primary);
        }

        .checkbox-label input {
            appearance: none;
            width: 12px;
            height: 12px;
            border: 1px solid var(--text-dim);
            background: transparent;
            cursor: pointer;
        }

        .checkbox-label input:checked {
            background: var(--text-primary);
            border-color: var(--text-primary);
        }

        .option-hint {
            font-size: 11px;
            color: var(--text-dim);
            align-self: center;
        }

        .btn {
            font-family: var(--mono);
            font-size: 11px;
            padding: 8px 16px;
            border: 1px solid var(--border);
            cursor: pointer;
            text-transform: uppercase;
            letter-spacing: 1px;
            transition: all 0.15s;
        }

        .btn-primary {
            background: var(--text-primary);
            color: var(--bg-primary);
            border-color: var(--text-primary);
        }

        .btn-primary:hover {
            background: #fff;
        }

        .btn-primary:disabled {
            background: var(--bg-elevated);
            color: var(--text-dim);
            border-color: var(--border);
            cursor: not-allowed;
        }

        /* Progress Log */
        .progress-log {
            background: var(--bg-primary);
            padding: 12px;
            margin-top: 12px;
            max-height: 200px;
            overflow-y: auto;
            font-size: 11px;
            line-height: 1.6;
        }

        .progress-log .line {
            display: flex;
            gap: 12px;
            color: var(--text-secondary);
        }

        .progress-log .line::before {
            content: '>';
            color: var(--text-dim);
        }

        .progress-log .error { color: var(--accent-red); }
        .progress-log .error::before { color: var(--accent-red); content: '!'; }
        .progress-log .success { color: var(--accent-green); }
        .progress-log .success::before { color: var(--accent-green); content: '+'; }

        /* Summary Cards */
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 32px;
        }

        .metric-card {
            background: var(--bg-secondary);
            padding: 20px;
        }

        .metric-label {
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: var(--text-dim);
            margin-bottom: 8px;
        }

        .metric-value {
            font-family: var(--sans);
            font-size: 32px;
            font-weight: 600;
            color: var(--text-primary);
        }

        .metric-value.positive { color: var(--accent-green); }
        .metric-value.negative { color: var(--accent-red); }
        .metric-value.warning { color: var(--accent-amber); }

        /* Data Table */
        .data-table-container {
            background: var(--bg-secondary);
        }

        .data-table {
            width: 100%;
            border-collapse: collapse;
        }

        .data-table th {
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: var(--text-dim);
            background: var(--bg-tertiary);
            padding: 14px 16px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }

        .data-table td {
            padding: 12px 16px;
            border-bottom: 1px solid var(--border);
            color: var(--text-secondary);
        }

        .data-table tr:hover td {
            background: var(--bg-tertiary);
            color: var(--text-primary);
        }

        .data-table tr.clickable {
            cursor: pointer;
        }

        .data-table tr.clickable:hover td {
            color: var(--text-primary);
        }

        .positive { color: var(--accent-green) !important; }
        .negative { color: var(--accent-red) !important; }
        .neutral { color: var(--text-dim) !important; }

        /* Section Header */
        .section-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin: 32px 0 16px 0;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--border);
        }

        .section-header h3 {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: var(--text-secondary);
        }

        /* Modal */
        .example-modal {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.9);
            display: none;
            align-items: center;
            justify-content: center;
            z-index: 1000;
            backdrop-filter: blur(4px);
        }

        .example-modal.active { display: flex; }

        .modal-content {
            background: var(--bg-secondary);
            max-width: 1200px;
            max-height: 90vh;
            overflow-y: auto;
            padding: 32px;
            margin: 20px;
            width: 90%;
        }

        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 24px;
            padding-bottom: 16px;
            border-bottom: 1px solid var(--border);
        }

        .modal-header h3 {
            font-family: var(--sans);
            font-size: 20px;
            font-weight: 600;
        }

        .close-btn {
            background: none;
            border: 1px solid var(--border);
            width: 36px;
            height: 36px;
            color: var(--text-secondary);
            font-size: 18px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.1s;
        }

        .close-btn:hover {
            border-color: var(--accent-red);
            color: var(--accent-red);
        }

        /* Tabs */
        .tabs {
            display: flex;
            gap: 8px;
            margin-bottom: 24px;
        }

        .tab-btn {
            font-family: var(--mono);
            font-size: 11px;
            padding: 10px 20px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            color: var(--text-secondary);
            cursor: pointer;
            text-transform: uppercase;
            letter-spacing: 1px;
            transition: all 0.1s;
        }

        .tab-btn:hover {
            background: var(--bg-elevated);
            color: var(--text-primary);
        }

        .tab-btn.active {
            background: var(--text-primary);
            color: var(--bg-primary);
            border-color: var(--text-primary);
        }

        /* Example Pairs */
        .example-pair {
            background: var(--bg-tertiary);
            padding: 20px;
            margin-bottom: 16px;
        }

        .example-probe {
            font-style: italic;
            color: var(--text-secondary);
            margin-bottom: 16px;
            padding: 12px;
            background: var(--bg-primary);
        }

        .responses-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }

        .response-box {
            padding: 16px;
            background: var(--bg-secondary);
        }

        .response-box.preferred {
            background: var(--accent-green-dim);
        }

        .response-box.not-preferred {
            background: var(--accent-red-dim);
        }

        .response-header {
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .response-box.preferred .response-header { color: var(--accent-green); }
        .response-box.preferred .response-header::before { content: '+'; }
        .response-box.not-preferred .response-header { color: var(--accent-red); }
        .response-box.not-preferred .response-header::before { content: '-'; }

        .response-text {
            white-space: pre-wrap;
            font-size: 12px;
            max-height: 200px;
            overflow-y: auto;
            line-height: 1.6;
            color: var(--text-secondary);
        }

        .score-badge {
            font-size: 10px;
            color: var(--text-dim);
            margin-top: 12px;
            padding-top: 12px;
            border-top: 1px solid var(--border);
        }

        /* Checkpoint List */
        .checkpoint-list {
            max-height: 180px;
            overflow-y: auto;
            background: var(--bg-primary);
        }

        .checkpoint-item {
            padding: 10px 14px;
            border-bottom: 1px solid var(--border);
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            transition: all 0.1s;
        }

        .checkpoint-item:hover {
            background: var(--bg-tertiary);
        }

        .checkpoint-item.selected {
            background: var(--bg-elevated);
            color: var(--text-primary);
        }

        .checkpoint-item:last-child { border-bottom: none; }

        /* Empty State */
        .empty-state {
            text-align: center;
            padding: 80px 40px;
            color: var(--text-dim);
        }

        .empty-state h3 {
            font-family: var(--sans);
            font-size: 20px;
            color: var(--text-secondary);
            margin-bottom: 8px;
        }

        .empty-state p {
            font-size: 12px;
        }

        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }

        ::-webkit-scrollbar-track {
            background: var(--bg-primary);
        }

        ::-webkit-scrollbar-thumb {
            background: var(--border-bright);
        }

        ::-webkit-scrollbar-thumb:hover {
            background: var(--text-dim);
        }

        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .metric-card, .data-table-container, .run-panel {
            animation: fadeIn 0.3s ease-out;
        }

        .metric-card:nth-child(2) { animation-delay: 0.05s; }
        .metric-card:nth-child(3) { animation-delay: 0.1s; }
        .metric-card:nth-child(4) { animation-delay: 0.15s; }
    </style>
</head>
<body>
    <div class="header">
        <div class="logo">EVAL SYS</div>
        <div class="header-controls">
            <select class="file-select" id="file-select" onchange="loadFile(this.value)">
                <option value="">[ SELECT RESULTS ]</option>
                {% for fname in files %}
                <option value="{{ fname }}" {{ 'selected' if fname == current_file else '' }}>{{ fname }}</option>
                {% endfor %}
            </select>
            <div class="status-indicator">
                <div class="status-dot"></div>
                <span id="model-info">READY</span>
            </div>
        </div>
    </div>

    <div class="main-container">
        <div class="sidebar">
            <div class="sidebar-section">
                <div class="sidebar-label">Execute</div>
                <div class="nav-item active" id="nav-run" onclick="showSection('run')">
                    <span class="nav-icon">></span> Run Eval
                </div>
            </div>
            <div class="sidebar-section">
                <div class="sidebar-label">Analysis</div>
                <div class="nav-item" id="nav-overview" onclick="showSection('overview')">
                    <span class="nav-icon">#</span> Overview
                </div>
                <div class="nav-item disabled" id="nav-dimension_correlations" onclick="showSection('dimension_correlations')">
                    <span class="nav-icon">~</span> Dim Correlations
                </div>
                <div class="nav-item disabled" id="nav-order_independence" onclick="showSection('order_independence')">
                    <span class="nav-icon">=</span> Order Independence
                </div>
                <div class="nav-item disabled" id="nav-pairwise" onclick="showSection('pairwise')">
                    <span class="nav-icon">|</span> Pairwise
                </div>
                <div class="nav-item disabled" id="nav-conversational" onclick="showSection('conversational')">
                    <span class="nav-icon">@</span> Conversational
                </div>
            </div>
        </div>

        <div class="content" id="content">
        </div>
    </div>

    <div class="example-modal" id="example-modal" onclick="if(event.target === this) closeModal()">
        <div class="modal-content" id="modal-content"></div>
    </div>

    <script>
        let currentData = null;
        let currentSection = 'run';
        let selectedCheckpoint = null;
        let modelType = 'api';  // 'api' or 'checkpoint'
        const checkpoints = {{ checkpoints | tojson }};
        const apiModels = {{ api_models | tojson }};

        async function loadFile(filename) {
            if (!filename) {
                currentData = null;
                updateNavState();
                showSection('run');
                return;
            }
            const resp = await fetch('/api/results/' + encodeURIComponent(filename));
            currentData = await resp.json();
            document.getElementById('model-info').textContent = currentData.model || '';
            updateNavState();
            showSection('overview');
        }

        function updateNavState() {
            const evals = currentData?.evals || {};
            ['order_independence', 'dimension_correlations', 'pairwise', 'conversational'].forEach(e => {
                const nav = document.getElementById('nav-' + e);
                // Check both nested (evals.X) and top-level format
                const hasData = evals[e] || (e === 'pairwise' && currentData?.comparisons);
                if (hasData) {
                    nav.classList.remove('disabled');
                } else {
                    nav.classList.add('disabled');
                }
            });
        }

        function showSection(section) {
            currentSection = section;
            document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
            document.getElementById('nav-' + section)?.classList.add('active');

            const content = document.getElementById('content');

            if (section === 'run') {
                content.innerHTML = renderRunPanel();
                setupRunPanel();
            } else if (section === 'overview') {
                content.innerHTML = renderOverview();
            } else if (section === 'dimension_correlations') {
                content.innerHTML = renderDimensionCorrelations();
            } else if (section === 'order_independence') {
                content.innerHTML = renderOrderIndependence();
            } else if (section === 'pairwise') {
                content.innerHTML = renderPairwise();
            } else if (section === 'conversational') {
                content.innerHTML = renderConversational();
            }
        }

        function renderRunPanel() {
            let html = '<h2 class="page-title">Run Eval</h2>';

            html += '<div class="run-panel">';

            // Model type tabs
            html += '<div class="model-type-tabs">';
            html += `<button class="model-type-tab ${modelType === 'api' ? 'active' : ''}" onclick="setModelType('api')">API Model</button>`;
            html += `<button class="model-type-tab ${modelType === 'checkpoint' ? 'active' : ''}" onclick="setModelType('checkpoint')">Local Checkpoint</button>`;
            html += '</div>';

            if (modelType === 'api') {
                html += '<div class="form-row">';
                html += '<label>Model</label>';
                html += '<select id="api-model-select">';
                for (const m of apiModels) {
                    html += `<option value="${m}">${m}</option>`;
                }
                html += '</select>';
                html += '</div>';

                html += '<div class="form-row">';
                html += '<label>Or enter model</label>';
                html += '<input type="text" id="custom-model" placeholder="provider/model-name">';
                html += '</div>';
            } else {
                html += '<div class="form-row">';
                html += '<label>Checkpoint</label>';
                html += '</div>';

                if (checkpoints.length === 0) {
                    html += '<div style="color:#888; padding:20px; text-align:center;">No checkpoints found. Set --checkpoints-dir when starting the viewer.</div>';
                } else {
                    html += '<div class="checkpoint-list" id="checkpoint-list">';
                    for (const cp of checkpoints) {
                        const selected = selectedCheckpoint === cp.path ? 'selected' : '';
                        html += `<div class="checkpoint-item ${selected}" onclick="selectCheckpoint('${cp.path}')">
                            <span>${cp.name}</span>
                            <span style="color:#666">${cp.step || ''}</span>
                        </div>`;
                    }
                    html += '</div>';
                }

                html += '<div class="form-row" style="margin-top:15px">';
                html += '<label>vLLM URL</label>';
                html += '<input type="text" id="vllm-url" placeholder="http://localhost:8000" value="http://localhost:8000">';
                html += '</div>';
            }

            html += '<div class="form-row" style="margin-top:20px">';
            html += '<label>Evals</label>';
            html += '<div style="display:flex; gap:20px; flex-wrap:wrap;">';
            html += '<label class="checkbox-label"><input type="checkbox" id="eval-dim-corr" checked> Dimension Correlations</label>';
            html += '<label class="checkbox-label"><input type="checkbox" id="eval-order-ind"> Order Independence</label>';
            html += '<label class="checkbox-label"><input type="checkbox" id="eval-pairwise"> Pairwise</label>';
            html += '<label class="checkbox-label"><input type="checkbox" id="eval-conversational"> Conversational</label>';
            html += '</div>';
            html += '</div>';

            html += '<div class="form-row">';
            html += '<label>Options</label>';
            html += '<div style="display:flex; gap:20px;">';
            html += '<label class="checkbox-label" title="Fewer probes, responses per probe, and comparison pairs. Faster but less statistically robust."><input type="checkbox" id="opt-quick" checked> Quick mode</label>';
            html += '</div>';
            html += '</div>';

            html += '<div class="form-row" style="margin-top:20px">';
            html += '<label></label>';
            html += '<button class="btn btn-primary" id="run-btn" onclick="startEval()">Run Eval</button>';
            html += '</div>';

            html += '<div class="progress-log" id="progress-log" style="display:none"></div>';

            html += '</div>';

            return html;
        }

        function setupRunPanel() {
            // Poll for status if eval is running
            if (window._statusPoll) clearInterval(window._statusPoll);
            window._statusPoll = setInterval(pollStatus, 1000);
            pollStatus();
        }

        function setModelType(type) {
            modelType = type;
            showSection('run');
        }

        function selectCheckpoint(path) {
            selectedCheckpoint = path;
            document.querySelectorAll('.checkpoint-item').forEach(el => el.classList.remove('selected'));
            event.target.closest('.checkpoint-item').classList.add('selected');
        }

        async function startEval() {
            let model;
            let localUrl = null;

            if (modelType === 'api') {
                const custom = document.getElementById('custom-model').value.trim();
                model = custom || document.getElementById('api-model-select').value;
            } else {
                if (!selectedCheckpoint) {
                    alert('Please select a checkpoint');
                    return;
                }
                model = selectedCheckpoint;
                localUrl = document.getElementById('vllm-url').value.trim();
            }

            const evals = [];
            if (document.getElementById('eval-dim-corr').checked) evals.push('dimension_correlations');
            if (document.getElementById('eval-order-ind').checked) evals.push('order_independence');
            if (document.getElementById('eval-pairwise').checked) evals.push('pairwise');
            if (document.getElementById('eval-conversational').checked) evals.push('conversational');

            const quick = document.getElementById('opt-quick').checked;

            document.getElementById('run-btn').disabled = true;
            document.getElementById('progress-log').style.display = 'block';
            document.getElementById('progress-log').innerHTML = '<div class="line">Starting eval...</div>';

            try {
                const resp = await fetch('/api/run', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model, evals, quick, local_url: localUrl })
                });
                const data = await resp.json();
                if (data.error) {
                    document.getElementById('progress-log').innerHTML += `<div class="line error">${data.error}</div>`;
                }
            } catch (e) {
                document.getElementById('progress-log').innerHTML += `<div class="line error">Error: ${e}</div>`;
            }
        }

        async function pollStatus() {
            try {
                const resp = await fetch('/api/status');
                const status = await resp.json();

                const log = document.getElementById('progress-log');
                const btn = document.getElementById('run-btn');

                if (!log || !btn) return;

                if (status.running) {
                    btn.disabled = true;
                    log.style.display = 'block';
                    log.innerHTML = status.progress.map(p => `<div class="line">${p}</div>`).join('');
                    log.scrollTop = log.scrollHeight;
                } else {
                    btn.disabled = false;
                    if (status.error) {
                        log.innerHTML += `<div class="line error">${status.error}</div>`;
                    }
                    if (status.result_file) {
                        log.innerHTML += `<div class="line success">Complete! Saved to ${status.result_file}</div>`;
                        // Refresh file list
                        refreshFileList();
                    }
                }
            } catch (e) {}
        }

        async function refreshFileList() {
            const resp = await fetch('/api/files');
            const files = await resp.json();
            const select = document.getElementById('file-select');
            const current = select.value;

            select.innerHTML = '<option value="">-- Select Results --</option>';
            for (const f of files) {
                const selected = f === current ? 'selected' : '';
                select.innerHTML += `<option value="${f}" ${selected}>${f}</option>`;
            }
        }

        function renderOverview() {
            if (!currentData) return '<div class="empty-state"><h3>No results loaded</h3><p>Run an eval or select a results file</p></div>';

            const evals = currentData.evals || {};
            let html = '<h2 class="page-title">Overview</h2>';

            // Model info header
            html += `<div class="run-panel" style="margin-bottom:24px">
                <div style="display:flex;justify-content:space-between;align-items:center">
                    <div>
                        <div class="panel-title">Model</div>
                        <div style="font-size:18px;font-family:var(--sans);font-weight:600;margin-top:8px">${currentData.model || 'Unknown'}</div>
                    </div>
                    <div style="text-align:right">
                        <div class="panel-title">Timestamp</div>
                        <div style="margin-top:8px;color:var(--text-secondary)">${currentData.timestamp || 'N/A'}</div>
                        <div style="margin-top:4px;font-size:11px;color:var(--text-dim)">${currentData.quick_mode ? 'Quick mode' : 'Full mode'}</div>
                    </div>
                </div>
            </div>`;

            // Summary metrics
            html += '<div class="metric-grid">';

            // Dimension correlations summary
            if (evals.dimension_correlations) {
                const dc = evals.dimension_correlations;
                const prefs = dc.preferences || [];
                const nConsistent = prefs.filter(p => p.order_consistent).length;
                const consistencyPct = prefs.length > 0 ? (nConsistent / prefs.length * 100) : 0;
                const consistencyCls = consistencyPct >= 75 ? 'positive' : consistencyPct >= 50 ? 'warning' : 'negative';

                html += `<div class="metric-card">
                    <div class="metric-label">Order Consistency</div>
                    <div class="metric-value ${consistencyCls}">${consistencyPct.toFixed(0)}%</div>
                </div>`;

                // Find strongest direct correlation
                const directDims = ['length_chars', 'length_words', 'first_person_density', 'question_density',
                                   'avg_sentence_length', 'vocab_richness', 'hedge_density'];
                let topDim = null, topCorr = 0;
                for (const dim of directDims) {
                    const stats = dc.correlations?.[dim];
                    if (stats && Math.abs(stats.correlation || 0) > Math.abs(topCorr)) {
                        topCorr = stats.correlation;
                        topDim = dim;
                    }
                }
                if (topDim) {
                    const cls = topCorr > 0 ? 'positive' : topCorr < 0 ? 'negative' : '';
                    html += `<div class="metric-card">
                        <div class="metric-label">Strongest: ${topDim.replace(/_/g, ' ')}</div>
                        <div class="metric-value ${cls}">${topCorr >= 0 ? '+' : ''}${topCorr.toFixed(2)}</div>
                    </div>`;
                }

                // Embedding correlation
                const emb = dc.embedding_analysis || {};
                if (emb.embedding_preference_correlation != null) {
                    const embCorr = emb.embedding_preference_correlation;
                    const embCls = embCorr > 0.3 ? 'positive' : embCorr < -0.3 ? 'negative' : '';
                    html += `<div class="metric-card">
                        <div class="metric-label">Embedding Correlation</div>
                        <div class="metric-value ${embCls}">${embCorr.toFixed(3)}</div>
                    </div>`;
                }

                html += `<div class="metric-card">
                    <div class="metric-label">Pool Models</div>
                    <div class="metric-value">${(dc.pool_models || []).length}</div>
                </div>`;
            }

            if (evals.order_independence) {
                const oi = evals.order_independence.summary;
                const rate = (oi.consistency_rate * 100);
                const cls = rate >= 80 ? 'positive' : rate >= 60 ? 'warning' : 'negative';
                html += `<div class="metric-card">
                    <div class="metric-label">Order Independence</div>
                    <div class="metric-value ${cls}">${rate.toFixed(0)}%</div>
                </div>`;
            }

            if (evals.pairwise) {
                const pw = evals.pairwise.aggregated;
                const rate = ((pw.overall?.win_rate || 0) * 100);
                const cls = rate >= 60 ? 'positive' : rate >= 40 ? 'warning' : 'negative';
                html += `<div class="metric-card">
                    <div class="metric-label">Pairwise Win Rate</div>
                    <div class="metric-value ${cls}">${rate.toFixed(0)}%</div>
                </div>`;
            }

            html += '</div>';

            // Detailed sections for each eval
            html += '<div class="section-header"><h3>Available Evals</h3></div>';
            html += '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px">';

            if (evals.dimension_correlations) {
                const dc = evals.dimension_correlations;
                const prefs = dc.preferences || [];
                const semWords = dc.embedding_analysis?.semantic_words || {};

                html += `<div class="run-panel" style="cursor:pointer" onclick="showSection('dimension_correlations')">
                    <div class="panel-title">Dimension Correlations</div>
                    <div style="margin-top:12px;font-size:12px;color:var(--text-secondary)">
                        ${Object.keys(dc.correlations || {}).length} dimensions measured
                    </div>
                    <div style="margin-top:8px;font-size:12px;color:var(--text-secondary)">
                        ${prefs.length} preference pairs collected
                    </div>
                    ${semWords.aligned?.length ? `<div style="margin-top:12px;padding-top:12px;border-top:1px solid var(--border)">
                        <span style="font-size:10px;color:var(--text-dim)">STEERING TOWARD:</span>
                        <span style="margin-left:8px;color:var(--accent-green)">${(semWords.aligned || []).slice(0, 3).map(w => w.word).join(', ')}</span>
                    </div>` : ''}
                    <div style="margin-top:12px;font-size:11px;color:var(--accent)">View details →</div>
                </div>`;
            }

            if (evals.order_independence) {
                const oi = evals.order_independence;
                const results = oi.results || [];
                const consistentCount = results.filter(r => r.consistent).length;
                // Compute position bias correctly from raw results
                const totalChoices = results.length * 2;
                const aPicksO1 = results.filter(r => r.order_1_choice === 'A').length;
                const aPicksO2 = results.filter(r => r.order_2_choice === 'A').length;
                const aRateOv = totalChoices > 0 ? ((aPicksO1 + aPicksO2) / totalChoices * 100).toFixed(0) : 0;
                const bRateOv = totalChoices > 0 ? (100 - aRateOv).toFixed(0) : 0;

                html += `<div class="run-panel" style="cursor:pointer" onclick="showSection('order_independence')">
                    <div class="panel-title">Order Independence</div>
                    <div style="margin-top:12px;font-size:12px;color:var(--text-secondary)">
                        ${consistentCount}/${results.length} pairs order-consistent
                    </div>
                    <div style="margin-top:8px;font-size:12px;color:var(--text-secondary)">
                        Position A: ${aRateOv}% | B: ${bRateOv}%
                    </div>
                    <div style="margin-top:12px;font-size:11px;color:var(--accent)">View details →</div>
                </div>`;
            }

            if (evals.pairwise) {
                const pw = evals.pairwise;
                const refs = pw.references || [];
                const overall = pw.aggregated?.overall || {};

                html += `<div class="run-panel" style="cursor:pointer" onclick="showSection('pairwise')">
                    <div class="panel-title">Pairwise Comparisons</div>
                    <div style="margin-top:12px;font-size:12px;color:var(--text-secondary)">
                        vs ${refs.length} reference models
                    </div>
                    <div style="margin-top:8px;font-size:12px;color:var(--text-secondary)">
                        ${overall.wins || 0} wins / ${overall.total || 0} comparisons
                    </div>
                    <div style="margin-top:12px;font-size:11px;color:var(--accent)">View details →</div>
                </div>`;
            }

            if (evals.conversational) {
                const cv = evals.conversational;
                html += `<div class="run-panel" style="cursor:pointer" onclick="showSection('conversational')">
                    <div class="panel-title">Conversational</div>
                    <div style="margin-top:12px;font-size:12px;color:var(--text-secondary)">
                        ${(cv.target_conversations || []).length} conversations
                    </div>
                    <div style="margin-top:12px;font-size:11px;color:var(--accent)">View details →</div>
                </div>`;
            }

            html += '</div>';

            // Quick insights if dimension correlations available
            if (evals.dimension_correlations) {
                const dc = evals.dimension_correlations;
                const examples = dc.examples || {};

                if (examples.most_preferred?.length || examples.least_preferred?.length) {
                    html += '<div class="section-header"><h3>Quick Insights</h3></div>';
                    html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:20px">';

                    if (examples.most_preferred?.length) {
                        const ex = examples.most_preferred[0];
                        html += `<div class="run-panel">
                            <div class="panel-title positive">Most Preferred Style</div>
                            <div style="margin-top:8px;font-size:10px;color:var(--text-dim)">[${ex.model?.split('/').pop() || 'unknown'}] ${ex.probe}</div>
                            <div style="margin-top:8px;font-size:12px;color:var(--text-secondary);max-height:150px;overflow:auto">${escapeHtmlWithNewlines(ex.text || '')}</div>
                        </div>`;
                    }

                    if (examples.least_preferred?.length) {
                        const ex = examples.least_preferred[0];
                        html += `<div class="run-panel">
                            <div class="panel-title negative">Least Preferred Style</div>
                            <div style="margin-top:8px;font-size:10px;color:var(--text-dim)">[${ex.model?.split('/').pop() || 'unknown'}] ${ex.probe}</div>
                            <div style="margin-top:8px;font-size:12px;color:var(--text-secondary);max-height:150px;overflow:auto">${escapeHtmlWithNewlines(ex.text || '')}</div>
                        </div>`;
                    }

                    html += '</div>';
                }
            }

            return html;
        }

        function renderDimensionCorrelations() {
            const dc = currentData?.evals?.dimension_correlations;
            if (!dc) return '<div class="empty-state"><h3>Dimension Correlations not run</h3></div>';

            let html = '<h2 class="page-title">Dimension Correlations</h2>';

            // Order consistency stats
            const prefs = dc.preferences || [];
            const nConsistent = prefs.filter(p => p.order_consistent).length;
            const consistencyPct = prefs.length > 0 ? (nConsistent / prefs.length * 100).toFixed(1) : 0;

            html += '<div class="metric-grid">';
            html += `<div class="metric-card"><div class="metric-label">Order Consistency</div><div class="metric-value">${consistencyPct}%</div></div>`;
            html += `<div class="metric-card"><div class="metric-label">Total Pairs</div><div class="metric-value">${prefs.length}</div></div>`;
            html += `<div class="metric-card"><div class="metric-label">Pool Models</div><div class="metric-value">${(dc.pool_models || []).length}</div></div>`;

            // Embedding analysis if available
            const emb = dc.embedding_analysis || {};
            if (emb.embedding_preference_correlation != null) {
                const embCorr = emb.embedding_preference_correlation;
                const cls = embCorr > 0.3 ? 'positive' : embCorr < -0.3 ? 'negative' : '';
                html += `<div class="metric-card"><div class="metric-label">Embedding Correlation</div><div class="metric-value ${cls}">${embCorr.toFixed(3)}</div></div>`;
            }
            html += '</div>';

            // Direct measurements table with per-probe toggle
            const correlationsByProbe = dc.correlations_by_probe || {};
            const corrProbeNames = Object.keys(correlationsByProbe);
            const hasPerProbeCorr = corrProbeNames.length > 0;

            html += '<div class="section-header"><h3>Direct Measurements</h3>';
            if (hasPerProbeCorr) {
                html += '<select id="corr-probe-select" onchange="updateCorrelationView()" style="margin-left:12px;padding:4px 8px;background:#333;color:#fff;border:1px solid #555;border-radius:4px">';
                html += '<option value="global">Global (all probes)</option>';
                for (const p of corrProbeNames) {
                    html += `<option value="${p}">${p}</option>`;
                }
                html += '</select>';
            }
            html += '</div>';

            const directDims = ['length_chars', 'length_words', 'first_person_density', 'question_density',
                               'avg_sentence_length', 'sentence_length_variance', 'vocab_richness',
                               'hedge_count', 'hedge_density', 'exclamation_count', 'ellipsis_count', 'dash_count'];

            // Store data for JS toggle (on window object directly, no script tag needed)
            window.dcCorrelations = dc.correlations || {};
            window.dcCorrelationsByProbe = correlationsByProbe;
            window.directDims = directDims;

            html += '<div id="correlations-table-container">';
            html += '<div class="data-table-container"><table class="data-table"><thead><tr><th>Dimension</th><th>Correlation</th><th>% Higher Preferred</th><th>N Confident</th><th></th></tr></thead><tbody>';

            // Sort by absolute correlation
            const sortedDims = directDims
                .map(dim => [dim, dc.correlations?.[dim] || {}])
                .sort((a, b) => Math.abs(b[1].correlation || 0) - Math.abs(a[1].correlation || 0));

            for (const [dim, stats] of sortedDims) {
                const corr = stats.correlation;
                const cls = corr > 0.1 ? 'positive' : corr < -0.1 ? 'negative' : 'neutral';
                html += `<tr class="clickable" onclick="showDimensionExamples('${dim}')">
                    <td>${dim}</td>
                    <td class="${cls}">${corr != null ? (corr >= 0 ? '+' : '') + corr.toFixed(3) : 'N/A'}</td>
                    <td>${stats.pct_higher != null ? (stats.pct_higher * 100).toFixed(1) + '%' : '-'}</td>
                    <td>${stats.n_confident || stats.n || 0}</td>
                    <td style="color:#888">→</td>
                </tr>`;
            }
            html += '</tbody></table></div></div>';

            // Combined per-probe section: examples + semantic directions together
            const examples = dc.examples || {};
            const examplesByProbe = examples.by_probe || {};
            const byProbe = emb.by_probe || {};
            const probes = dc.probes || {};

            // Get all probe names from either source
            const allProbeNames = [...new Set([...Object.keys(examplesByProbe), ...Object.keys(byProbe)])];

            if (allProbeNames.length > 0) {
                html += '<div class="section-header"><h3>Per-Probe Analysis</h3></div>';
                html += '<p style="color:#888;font-size:11px;margin-bottom:12px">Semantic preference directions and example responses for each probe</p>';

                for (const probeName of allProbeNames) {
                    const probeExamples = examplesByProbe[probeName] || {};
                    const probeData = byProbe[probeName] || {};
                    const probeText = probes[probeName] || '';

                    const hasExamples = probeExamples.most_preferred?.length || probeExamples.least_preferred?.length;
                    const hasSemantics = probeData.aligned?.length || probeData.opposed?.length;
                    if (!hasExamples && !hasSemantics) continue;

                    html += `<div style="margin-bottom:24px;padding:16px;background:#1a1a1a;border-radius:6px;border-left:3px solid #444">`;

                    // Header with probe name and full text
                    html += `<div style="margin-bottom:16px">`;
                    html += `<div style="font-weight:600;font-size:15px;color:#ddd;margin-bottom:6px">${probeName}</div>`;
                    if (probeText) {
                        html += `<div style="font-size:12px;color:#888;font-style:italic;padding:8px 12px;background:#111;border-radius:4px">"${escapeHtml(probeText)}"</div>`;
                    }
                    if (probeData.n_texts) {
                        html += `<div style="font-size:10px;color:#666;margin-top:6px">${probeData.n_texts} texts | ${probeData.n_high_pref || 0} preferred / ${probeData.n_low_pref || 0} not | direction magnitude: ${(probeData.direction_magnitude || 0).toFixed(3)}</div>`;
                    }
                    html += `</div>`;

                    // Semantic directions (if available)
                    if (hasSemantics) {
                        html += '<div style="margin-bottom:16px">';
                        html += '<div style="font-size:11px;color:#666;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.5px">Semantic Direction</div>';
                        html += '<div style="display:grid; grid-template-columns:1fr 1fr; gap:16px">';

                        html += '<div><div class="panel-title positive" style="font-size:11px">Steering Toward</div>';
                        html += '<div class="data-table-container"><table class="data-table" style="font-size:11px"><thead><tr><th>Word/Phrase</th><th>Sim</th></tr></thead><tbody>';
                        for (const item of (probeData.aligned || []).slice(0, 10)) {
                            html += `<tr><td>${item.word}</td><td class="positive">${item.similarity.toFixed(3)}</td></tr>`;
                        }
                        html += '</tbody></table></div></div>';

                        html += '<div><div class="panel-title negative" style="font-size:11px">Steering Away From</div>';
                        html += '<div class="data-table-container"><table class="data-table" style="font-size:11px"><thead><tr><th>Word/Phrase</th><th>Sim</th></tr></thead><tbody>';
                        for (const item of (probeData.opposed || []).slice(0, 10)) {
                            html += `<tr><td>${item.word}</td><td class="negative">${item.similarity.toFixed(3)}</td></tr>`;
                        }
                        html += '</tbody></table></div></div>';

                        html += '</div></div>';
                    }

                    // Example responses (if available)
                    if (hasExamples) {
                        html += '<div style="font-size:11px;color:#666;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.5px">Example Responses</div>';
                        html += '<div style="display:grid; grid-template-columns:1fr 1fr; gap:16px">';

                        html += '<div><div class="panel-title positive" style="font-size:11px">Most Preferred</div>';
                        for (const ex of (probeExamples.most_preferred || []).slice(0, 5)) {
                            html += `<div class="example-pair" style="margin-bottom:12px">
                                <div style="font-size:10px;color:#666;margin-bottom:4px">[${ex.model?.split('/').pop() || 'unknown'}] score: ${ex.score.toFixed(2)}</div>
                                <div style="font-size:12px;color:#aaa;max-height:120px;overflow:auto">${escapeHtmlWithNewlines(ex.text || '')}</div>
                            </div>`;
                        }
                        html += '</div>';

                        html += '<div><div class="panel-title negative" style="font-size:11px">Least Preferred</div>';
                        for (const ex of (probeExamples.least_preferred || []).slice(0, 5)) {
                            html += `<div class="example-pair" style="margin-bottom:12px">
                                <div style="font-size:10px;color:#666;margin-bottom:4px">[${ex.model?.split('/').pop() || 'unknown'}] score: ${ex.score.toFixed(2)}</div>
                                <div style="font-size:12px;color:#aaa;max-height:120px;overflow:auto">${escapeHtmlWithNewlines(ex.text || '')}</div>
                            </div>`;
                        }
                        html += '</div>';

                        html += '</div>';
                    }

                    html += '</div>';
                }
            } else if (examples.most_preferred?.length || examples.least_preferred?.length) {
                // Fallback for old data format without by_probe
                html += '<div class="section-header"><h3>Top/Bottom Examples</h3></div>';
                html += '<div style="display:grid; grid-template-columns:1fr 1fr; gap:20px">';

                html += '<div><div class="panel-title positive">Most Preferred</div>';
                for (const ex of (examples.most_preferred || []).slice(0, 3)) {
                    html += `<div class="example-pair" style="margin-bottom:12px">
                        <div style="font-size:10px;color:#666;margin-bottom:4px">[${ex.model?.split('/').pop() || 'unknown'}] ${ex.probe} (score: ${ex.score.toFixed(2)})</div>
                        <div style="font-size:12px;color:#aaa;max-height:150px;overflow:auto">${escapeHtmlWithNewlines(ex.text || '')}</div>
                    </div>`;
                }
                html += '</div>';

                html += '<div><div class="panel-title negative">Least Preferred</div>';
                for (const ex of (examples.least_preferred || []).slice(0, 3)) {
                    html += `<div class="example-pair" style="margin-bottom:12px">
                        <div style="font-size:10px;color:#666;margin-bottom:4px">[${ex.model?.split('/').pop() || 'unknown'}] ${ex.probe} (score: ${ex.score.toFixed(2)})</div>
                        <div style="font-size:12px;color:#aaa;max-height:150px;overflow:auto">${escapeHtmlWithNewlines(ex.text || '')}</div>
                    </div>`;
                }
                html += '</div>';

                html += '</div>';
            }

            return html;
        }

        function updateCorrelationView() {
            const select = document.getElementById('corr-probe-select');
            const selectedProbe = select?.value || 'global';
            const container = document.getElementById('correlations-table-container');
            if (!container || !window.directDims) return;

            const correlations = selectedProbe === 'global'
                ? window.dcCorrelations
                : (window.dcCorrelationsByProbe?.[selectedProbe] || {});

            // Sort by absolute correlation
            const sortedDims = window.directDims
                .map(dim => [dim, correlations[dim] || {}])
                .sort((a, b) => Math.abs(b[1].correlation || 0) - Math.abs(a[1].correlation || 0));

            let html = '<div class="data-table-container"><table class="data-table"><thead><tr><th>Dimension</th><th>Correlation</th><th>% Higher Preferred</th><th>N Confident</th><th></th></tr></thead><tbody>';
            for (const [dim, stats] of sortedDims) {
                const corr = stats.correlation;
                const cls = corr > 0.1 ? 'positive' : corr < -0.1 ? 'negative' : 'neutral';
                html += `<tr class="clickable" onclick="showDimensionExamples('${dim}')">
                    <td>${dim}</td>
                    <td class="${cls}">${corr != null ? (corr >= 0 ? '+' : '') + corr.toFixed(3) : 'N/A'}</td>
                    <td>${stats.pct_higher != null ? (stats.pct_higher * 100).toFixed(1) + '%' : '-'}</td>
                    <td>${stats.n_confident || stats.n || 0}</td>
                    <td style="color:#888">→</td>
                </tr>`;
            }
            html += '</tbody></table></div>';
            container.innerHTML = html;
        }

        function showDimensionExamples(dim) {
            const dc = currentData?.evals?.dimension_correlations;
            if (!dc) return;

            const pool = dc.pool || {};
            const preferences = dc.preferences || [];
            const probes = dc.probes || {};

            const examples = { agrees: [], disagrees: [], unclear: [] };

            for (const pref of preferences) {
                // Skip order-inconsistent preferences for clear examples
                if (!pref.order_consistent) {
                    continue;
                }

                const probe = pref.probe;
                const responses = pool[probe] || [];
                const idxA = pref.index_a;
                const idxB = pref.index_b;

                const respA = responses.find(r => r.index === idxA);
                const respB = responses.find(r => r.index === idxB);
                if (!respA || !respB) continue;

                // soft_score: 1.0 = prefers A, 0.0 = prefers B
                const softScore = pref.soft_score;
                const preferred = softScore > 0.5 ? respA : respB;
                const notPreferred = softScore > 0.5 ? respB : respA;

                const valPref = preferred.dimensions?.[dim];
                const valNotPref = notPreferred.dimensions?.[dim];

                if (valPref == null || valNotPref == null) continue;

                const example = {
                    probe,
                    question: probes[probe] || probe,
                    preferred: { text: preferred.text, score: valPref, model: preferred.model },
                    notPreferred: { text: notPreferred.text, score: valNotPref, model: notPreferred.model },
                };

                if (valPref > valNotPref) {
                    examples.agrees.push(example);
                } else if (valPref < valNotPref) {
                    examples.disagrees.push(example);
                }
            }

            let html = `<div class="modal-header">
                <h3>Examples: ${dim}</h3>
                <button class="close-btn" onclick="closeModal()">&times;</button>
            </div>`;

            html += '<div class="tabs">';
            html += `<button class="tab-btn active" onclick="filterExamples('agrees')">Agrees (${examples.agrees.length})</button>`;
            html += `<button class="tab-btn" onclick="filterExamples('disagrees')">Disagrees (${examples.disagrees.length})</button>`;
            html += '</div>';

            html += '<div id="examples-list">';
            html += renderExamplesList(examples.agrees.slice(0, 5));
            html += '</div>';

            window._currentExamples = examples;

            document.getElementById('modal-content').innerHTML = html;
            document.getElementById('example-modal').classList.add('active');
        }

        function filterExamples(type) {
            document.querySelectorAll('.tabs .tab-btn').forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');

            const examples = window._currentExamples?.[type] || [];
            document.getElementById('examples-list').innerHTML = renderExamplesList(examples.slice(0, 5));
        }

        function renderExamplesList(examples) {
            if (!examples.length) return '<div class="empty-state">No examples</div>';

            return examples.map((ex, i) => `
                <div class="example-pair">
                    <div class="example-probe">"${escapeHtml(ex.question)}"</div>
                    <div class="responses-grid">
                        <div class="response-box preferred">
                            <div class="response-header">Preferred ${ex.preferred.model ? `(${ex.preferred.model.split('/').pop()})` : ''}</div>
                            <div class="response-text">${escapeHtml(ex.preferred.text)}</div>
                            <div class="score-badge">Score: ${typeof ex.preferred.score === 'number' ? ex.preferred.score.toFixed(3) : ex.preferred.score}</div>
                        </div>
                        <div class="response-box not-preferred">
                            <div class="response-header">Not Preferred ${ex.notPreferred.model ? `(${ex.notPreferred.model.split('/').pop()})` : ''}</div>
                            <div class="response-text">${escapeHtml(ex.notPreferred.text)}</div>
                            <div class="score-badge">Score: ${typeof ex.notPreferred.score === 'number' ? ex.notPreferred.score.toFixed(3) : ex.notPreferred.score}</div>
                        </div>
                    </div>
                </div>
            `).join('');
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text || '';
            return div.innerHTML;
        }

        function escapeHtmlWithNewlines(text) {
            const div = document.createElement('div');
            div.textContent = text || '';
            return div.innerHTML.split(String.fromCharCode(10)).join('<br>');
        }

        function closeModal() {
            document.getElementById('example-modal').classList.remove('active');
        }

        function showTextModal(title, text) {
            document.getElementById('modal-content').innerHTML = `
                <button class="close-btn" onclick="closeModal()">&times;</button>
                <h3 style="margin-bottom:16px;font-family:var(--sans)">${escapeHtml(title)}</h3>
                <div style="white-space:pre-wrap;font-size:13px;line-height:1.6;max-height:70vh;overflow:auto">${escapeHtml(text)}</div>
            `;
            document.getElementById('example-modal').classList.add('active');
        }

        function renderOrderIndependence() {
            const oi = currentData?.evals?.order_independence;
            if (!oi) return '<div class="empty-state"><h3>Order Independence not run</h3></div>';

            const summary = oi.summary || {};
            const results = oi.results || [];
            const responsePairs = oi.response_pairs || {};

            let html = '<h2 class="page-title">Order Independence</h2>';

            // Summary metrics
            const consistencyRate = summary.consistency_rate || 0;
            const consistencyCls = consistencyRate >= 0.8 ? 'positive' : consistencyRate >= 0.6 ? 'warning' : 'negative';
            // Recompute position bias from raw results (fixes bug in old saved data)
            const totalChoices = results.length * 2;
            const aPicksOrder1 = results.filter(r => r.order_1_choice === 'A').length;
            const aPicksOrder2 = results.filter(r => r.order_2_choice === 'A').length;
            const aRate = totalChoices > 0 ? (aPicksOrder1 + aPicksOrder2) / totalChoices : 0;
            const bRate = totalChoices > 0 ? 1 - aRate : 0;
            const biasDiff = Math.abs(aRate - bRate);
            const biasCls = biasDiff <= 0.1 ? 'positive' : biasDiff <= 0.25 ? 'warning' : 'negative';

            html += '<div class="metric-grid">';
            html += `<div class="metric-card">
                <div class="metric-label">Consistency Rate</div>
                <div class="metric-value ${consistencyCls}">${(consistencyRate * 100).toFixed(0)}%</div>
            </div>`;
            html += `<div class="metric-card">
                <div class="metric-label">Position Bias</div>
                <div class="metric-value ${biasCls}">${biasDiff < 0.01 ? 'None' : (aRate > bRate ? 'A' : 'B') + ' +' + (biasDiff * 100).toFixed(0) + '%'}</div>
            </div>`;
            html += `<div class="metric-card">
                <div class="metric-label">Total Pairs</div>
                <div class="metric-value">${results.length}</div>
            </div>`;
            html += `<div class="metric-card">
                <div class="metric-label">Position A / B Rate</div>
                <div class="metric-value" style="font-size:24px">${(aRate * 100).toFixed(0)}% / ${(bRate * 100).toFixed(0)}%</div>
            </div>`;
            html += '</div>';

            // Position bias visualization
            html += '<div class="section-header"><h3>Position Bias Visualization</h3></div>';
            html += '<div class="run-panel">';
            html += '<div style="display:flex;align-items:center;gap:16px;margin-bottom:12px">';
            html += `<div style="width:${aRate * 100}%;min-width:20px;background:var(--accent-green);height:32px;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:600;color:var(--bg-primary)">A: ${(aRate * 100).toFixed(0)}%</div>`;
            html += `<div style="width:${bRate * 100}%;min-width:20px;background:var(--accent-red);height:32px;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:600;color:var(--bg-primary)">B: ${(bRate * 100).toFixed(0)}%</div>`;
            html += '</div>';
            html += '<div style="font-size:11px;color:var(--text-dim)">50/50 split indicates no position bias. Significant deviation suggests the model prefers whichever response appears first (A) or second (B).</div>';
            html += '</div>';

            // Results by probe
            html += '<div class="section-header"><h3>Results by Probe</h3></div>';

            // Group results by probe
            const byProbe = {};
            for (const r of results) {
                const probe = r.probe || 'unknown';
                if (!byProbe[probe]) byProbe[probe] = [];
                byProbe[probe].push(r);
            }

            for (const [probe, probeResults] of Object.entries(byProbe)) {
                const consistentCount = probeResults.filter(r => r.consistent).length;
                const probeCls = consistentCount === probeResults.length ? 'positive' : consistentCount > 0 ? 'warning' : 'negative';

                html += `<div class="run-panel" style="margin-bottom:16px">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
                        <div class="panel-title">${probe}</div>
                        <div class="${probeCls}" style="font-size:12px">${consistentCount}/${probeResults.length} consistent</div>
                    </div>`;

                // Show individual pairs
                html += '<div class="data-table-container"><table class="data-table"><thead><tr><th>Pair</th><th>Order 1 (A|B)</th><th>Order 2 (B|A)</th><th>Status</th></tr></thead><tbody>';

                for (let i = 0; i < probeResults.length; i++) {
                    const r = probeResults[i];
                    const statusCls = r.consistent ? 'positive' : 'negative';
                    const statusIcon = r.consistent ? '✓' : '✗';

                    html += `<tr class="clickable" onclick="showOrderPairDetail('${probe}', ${i})">
                        <td>Pair ${i + 1}</td>
                        <td>${r.order_1_choice || '-'}</td>
                        <td>${r.order_2_choice || '-'}</td>
                        <td class="${statusCls}">${statusIcon} ${r.consistent ? 'Consistent' : 'Inconsistent'}</td>
                    </tr>`;
                }

                html += '</tbody></table></div></div>';
            }

            // Summary interpretation
            html += '<div class="section-header"><h3>Interpretation</h3></div>';
            html += '<div class="run-panel">';

            if (consistencyRate >= 0.8) {
                html += '<div style="color:var(--accent-green);margin-bottom:8px">Strong order independence</div>';
                html += '<div style="font-size:12px;color:var(--text-secondary)">The model shows consistent preferences regardless of presentation order. This indicates robust, content-based evaluation rather than position-based bias.</div>';
            } else if (consistencyRate >= 0.6) {
                html += '<div style="color:var(--accent-amber);margin-bottom:8px">Moderate order independence</div>';
                html += '<div style="font-size:12px;color:var(--text-secondary)">The model sometimes changes its preference based on presentation order. This suggests some position bias that may affect reliability of pairwise judgments.</div>';
            } else {
                html += '<div style="color:var(--accent-red);margin-bottom:8px">Weak order independence</div>';
                html += '<div style="font-size:12px;color:var(--text-secondary)">The model frequently changes its preference based on presentation order. Pairwise judgments from this model should be interpreted with caution.</div>';
            }

            if (biasDiff > 0.1) {
                const preferredPos = aRate > bRate ? 'first (A)' : 'second (B)';
                html += `<div style="margin-top:12px;font-size:12px;color:var(--text-secondary)">Position bias detected: the model tends to prefer the ${preferredPos} option.</div>`;
            }

            html += '</div>';

            return html;
        }

        function showOrderPairDetail(probe, pairIdx) {
            const oi = currentData?.evals?.order_independence;
            if (!oi) return;

            const results = oi.results || [];
            const responsePairs = oi.response_pairs || {};

            // Find the specific result
            const probeResults = results.filter(r => r.probe === probe);
            const result = probeResults[pairIdx];
            if (!result) return;

            // Get the response pair
            const pairs = responsePairs[probe] || [];
            const pair = pairs[pairIdx] || {};

            let html = `<div class="modal-header">
                <h3>${probe} - Pair ${pairIdx + 1}</h3>
                <button class="close-btn" onclick="closeModal()">&times;</button>
            </div>`;

            const statusCls = result.consistent ? 'positive' : 'negative';
            html += `<div class="metric-grid" style="margin-bottom:24px">
                <div class="metric-card">
                    <div class="metric-label">Order 1 Choice</div>
                    <div class="metric-value">${result.order_1_choice || '-'}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Order 2 Choice</div>
                    <div class="metric-value">${result.order_2_choice || '-'}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Status</div>
                    <div class="metric-value ${statusCls}">${result.consistent ? 'Consistent' : 'Inconsistent'}</div>
                </div>
            </div>`;

            html += '<div class="responses-grid">';
            html += `<div class="response-box ${result.order_1_choice === 'A' ? 'preferred' : 'not-preferred'}">
                <div class="response-header">Response A</div>
                <div class="response-text">${escapeHtml(pair.response_a || result.response_a || 'N/A')}</div>
            </div>`;
            html += `<div class="response-box ${result.order_1_choice === 'B' ? 'preferred' : 'not-preferred'}">
                <div class="response-header">Response B</div>
                <div class="response-text">${escapeHtml(pair.response_b || result.response_b || 'N/A')}</div>
            </div>`;
            html += '</div>';

            if (!result.consistent) {
                html += `<div class="run-panel" style="margin-top:16px;background:var(--accent-red-dim)">
                    <div style="color:var(--accent-red);font-size:11px;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px">Order Inconsistency</div>
                    <div style="font-size:12px;color:var(--text-secondary)">
                        When shown as A|B, the model chose ${result.order_1_choice}.<br>
                        When shown as B|A, the model chose ${result.order_2_choice}.<br>
                        This indicates the preference was influenced by presentation order rather than content.
                    </div>
                </div>`;
            }

            document.getElementById('modal-content').innerHTML = html;
            document.getElementById('example-modal').classList.add('active');
        }

        function renderPairwise() {
            // Support both formats: evals.pairwise or top-level pairwise data
            let pw = currentData?.evals?.pairwise;
            if (!pw && currentData?.comparisons) {
                // Top-level format (standalone pairwise file)
                pw = currentData;
            }
            if (!pw) return '<div class="empty-state"><h3>Pairwise not run</h3></div>';

            const targetModel = pw.target || currentData.model;
            const refs = pw.references || [];
            const probes = pw.probes || [];
            const responses = pw.responses || {};
            const comparisons = pw.comparisons || [];
            const winRates = pw.win_rates || {};

            // Compute stats from comparisons
            let wins = 0, losses = 0, ties = 0;
            const refStats = {};

            for (const comp of comparisons) {
                // Determine target and reference from model_a/model_b and order
                const isRefFirst = comp.order === 'ref_first';
                const refModel = isRefFirst ? comp.model_a : comp.model_b;
                const choice = comp.choice;

                // Target wins if: ref_first and choice=B, or target_first and choice=A
                const targetWins = (isRefFirst && choice === 'B') || (!isRefFirst && choice === 'A');
                const isTie = choice === 'tie' || choice === 'Tie';

                if (isTie) ties++;
                else if (targetWins) wins++;
                else losses++;

                // Track by reference
                if (!refStats[refModel]) refStats[refModel] = { wins: 0, losses: 0, ties: 0 };
                if (isTie) refStats[refModel].ties++;
                else if (targetWins) refStats[refModel].wins++;
                else refStats[refModel].losses++;
            }

            const total = comparisons.length;
            const winRate = total > 0 ? wins / total : 0;

            let html = '<h2 class="page-title">Pairwise Comparisons</h2>';

            // Overall metrics
            const winRateCls = winRate >= 0.6 ? 'positive' : winRate >= 0.4 ? 'warning' : 'negative';

            html += '<div class="metric-grid">';
            html += `<div class="metric-card">
                <div class="metric-label">Overall Win Rate</div>
                <div class="metric-value ${winRateCls}">${(winRate * 100).toFixed(0)}%</div>
            </div>`;
            html += `<div class="metric-card">
                <div class="metric-label">Total Comparisons</div>
                <div class="metric-value">${total}</div>
            </div>`;
            html += `<div class="metric-card">
                <div class="metric-label">Wins / Losses / Ties</div>
                <div class="metric-value" style="font-size:20px">${wins} / ${losses} / ${ties}</div>
            </div>`;
            html += `<div class="metric-card">
                <div class="metric-label">Reference Models</div>
                <div class="metric-value">${Object.keys(refStats).length || refs.length}</div>
            </div>`;
            html += '</div>';

            // Win rate visualization bar
            html += '<div class="section-header"><h3>Win Rate by Reference</h3></div>';
            html += '<div class="run-panel">';

            const refList = Object.keys(refStats).length > 0 ? Object.keys(refStats) : refs;
            for (const ref of refList) {
                const stats = refStats[ref] || { wins: 0, losses: 0, ties: 0 };
                const refTotal = stats.wins + stats.losses + stats.ties;
                const refWinRate = refTotal > 0 ? stats.wins / refTotal : 0;
                const refCls = refWinRate >= 0.6 ? 'positive' : refWinRate >= 0.4 ? 'warning' : 'negative';
                const refName = ref.split('/').pop();

                html += `<div style="margin-bottom:16px">
                    <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                        <span style="font-size:12px">${refName}</span>
                        <span class="${refCls}" style="font-size:12px">${(refWinRate * 100).toFixed(0)}% (${stats.wins}/${refTotal})</span>
                    </div>
                    <div style="background:var(--bg-primary);height:8px;width:100%">
                        <div style="background:${refWinRate >= 0.5 ? 'var(--accent-green)' : 'var(--accent-red)'};height:100%;width:${Math.max(refWinRate * 100, 2)}%"></div>
                    </div>
                </div>`;
            }

            html += '</div>';

            // Detailed comparisons table
            html += '<div class="section-header"><h3>Comparison Details</h3></div>';
            html += '<div class="data-table-container"><table class="data-table"><thead><tr><th>Probe</th><th>Target</th><th>Reference</th><th>Judge</th><th>Winner</th></tr></thead><tbody>';

            // Group comparisons by probe for easier reading
            const compByProbe = {};
            for (const comp of comparisons) {
                const probeName = comp.probe || 'unknown';
                if (!compByProbe[probeName]) compByProbe[probeName] = [];
                compByProbe[probeName].push(comp);
            }

            for (const [probeName, probeComps] of Object.entries(compByProbe)) {
                for (let i = 0; i < probeComps.length; i++) {
                    const comp = probeComps[i];
                    const isRefFirst = comp.order === 'ref_first';
                    const refModel = isRefFirst ? comp.model_a : comp.model_b;
                    const choice = comp.choice;

                    const targetWins = (isRefFirst && choice === 'B') || (!isRefFirst && choice === 'A');
                    const isTie = choice === 'tie' || choice === 'Tie';

                    const winnerCls = isTie ? '' : (targetWins ? 'positive' : 'negative');
                    const winnerDisplay = isTie ? 'Tie' : (targetWins ? 'Target' : 'Reference');
                    const refName = refModel?.split('/').pop() || '-';
                    const judgeName = (comp.judge || '').split('/').pop();
                    const targetName = targetModel?.split('/').pop() || 'target';

                    html += `<tr class="clickable" onclick="showPairwiseDetail('${probeName}', ${i})">
                        <td>${probeName}</td>
                        <td style="color:var(--text-dim)">${targetName}</td>
                        <td>${refName}</td>
                        <td>${judgeName}</td>
                        <td class="${winnerCls}">${winnerDisplay}</td>
                    </tr>`;
                }
            }

            html += '</tbody></table></div>';

            // Response samples section - handle different data structures
            if (Object.keys(responses).length > 0) {
                html += '<div class="section-header"><h3>Response Samples</h3></div>';

                // Try to find a sample probe - could be responses[model][probe] or responses[probe][model]
                const firstKey = Object.keys(responses)[0];
                const firstVal = responses[firstKey];

                // Detect structure: if firstVal has probe-like keys with content objects, it's responses[model][probe]
                let samplesByModel = {};
                let sampleProbe = null;

                if (firstVal && typeof firstVal === 'object') {
                    const subKeys = Object.keys(firstVal);
                    if (subKeys.length > 0 && firstVal[subKeys[0]]?.content) {
                        // Structure: responses[model][probe] = {content: ...}
                        sampleProbe = subKeys[0];
                        for (const [model, probes] of Object.entries(responses)) {
                            if (probes[sampleProbe]?.content) {
                                samplesByModel[model] = probes[sampleProbe].content;
                            }
                        }
                    } else if (typeof firstVal === 'string') {
                        // Structure: responses[probe][model] = string
                        sampleProbe = firstKey;
                        samplesByModel = firstVal;
                    }
                }

                if (sampleProbe && Object.keys(samplesByModel).length > 0) {
                    html += `<div class="run-panel">
                        <div class="panel-title">${sampleProbe}</div>
                        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px;margin-top:12px">`;

                    // Show responses by model
                    for (const [model, text] of Object.entries(samplesByModel).slice(0, 4)) {
                        const textStr = typeof text === 'string' ? text : (text?.content || JSON.stringify(text));
                        const isTarget = model === targetModel;
                        const shortModel = model.split('/').pop();
                        html += `<div class="response-box ${isTarget ? 'preferred' : ''}">
                            <div class="response-header">${isTarget ? 'Target: ' : 'Ref: '}${shortModel}</div>
                            <div class="response-text" style="max-height:200px;overflow:auto">${escapeHtmlWithNewlines(textStr)}</div>
                        </div>`;
                    }

                    html += '</div></div>';
                }
            }

            return html;
        }

        function showPairwiseDetail(probe, compIdx) {
            // Support both formats
            let pw = currentData?.evals?.pairwise;
            if (!pw && currentData?.comparisons) pw = currentData;
            if (!pw) return;

            const targetModel = pw.target || currentData.model;
            const responses = pw.responses || {};
            const comparisons = pw.comparisons || [];

            // Group by probe and get the specific comparison
            const compByProbe = {};
            for (const comp of comparisons) {
                const probeName = comp.probe || 'unknown';
                if (!compByProbe[probeName]) compByProbe[probeName] = [];
                compByProbe[probeName].push(comp);
            }

            const probeComps = compByProbe[probe] || [];
            const comp = probeComps[compIdx];
            if (!comp) return;

            // Parse the comparison structure
            const isRefFirst = comp.order === 'ref_first';
            const refModel = isRefFirst ? comp.model_a : comp.model_b;
            const choice = comp.choice;
            const targetWins = (isRefFirst && choice === 'B') || (!isRefFirst && choice === 'A');
            const isTie = choice === 'tie' || choice === 'Tie';

            // Get responses - handle different structures
            // Could be responses[model][probe] = {content: ...} or responses[probe][model] = string
            let targetResponse = 'N/A';
            let refResponse = 'N/A';

            if (responses[targetModel]?.[probe]?.content) {
                targetResponse = responses[targetModel][probe].content;
            } else if (responses[probe]?.[targetModel]) {
                targetResponse = responses[probe][targetModel];
            }

            if (responses[refModel]?.[probe]?.content) {
                refResponse = responses[refModel][probe].content;
            } else if (responses[probe]?.[refModel]) {
                refResponse = responses[probe][refModel];
            }

            let html = `<div class="modal-header">
                <h3>Pairwise: ${probe}</h3>
                <button class="close-btn" onclick="closeModal()">&times;</button>
            </div>`;

            const winnerCls = isTie ? '' : (targetWins ? 'positive' : 'negative');
            const winnerDisplay = isTie ? 'Tie' : (targetWins ? 'Target' : 'Reference');

            html += `<div class="metric-grid" style="margin-bottom:24px">
                <div class="metric-card">
                    <div class="metric-label">Target</div>
                    <div class="metric-value" style="font-size:16px">${targetModel?.split('/').pop() || 'target'}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Reference</div>
                    <div class="metric-value" style="font-size:16px">${refModel?.split('/').pop()}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Judge</div>
                    <div class="metric-value" style="font-size:16px">${comp.judge?.split('/').pop()}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Winner</div>
                    <div class="metric-value ${winnerCls}">${winnerDisplay}</div>
                </div>
            </div>`;

            // Show order info
            html += `<div style="margin-bottom:16px;font-size:11px;color:var(--text-dim)">
                Order: ${isRefFirst ? 'Reference first (A), Target second (B)' : 'Target first (A), Reference second (B)'} | Choice: ${choice}
            </div>`;

            html += '<div class="responses-grid">';
            html += `<div class="response-box ${targetWins ? 'preferred' : (!isTie ? 'not-preferred' : '')}">
                <div class="response-header">Target: ${targetModel?.split('/').pop()}</div>
                <div class="response-text">${escapeHtml(targetResponse)}</div>
            </div>`;
            html += `<div class="response-box ${!targetWins && !isTie ? 'preferred' : (!isTie ? 'not-preferred' : '')}">
                <div class="response-header">Reference: ${refModel?.split('/').pop()}</div>
                <div class="response-text">${escapeHtml(refResponse)}</div>
            </div>`;
            html += '</div>';

            if (comp.explanation) {
                html += `<div class="run-panel" style="margin-top:16px">
                    <div class="panel-title">Judge Reasoning</div>
                    <div style="margin-top:8px;font-size:12px;color:var(--text-secondary);white-space:pre-wrap">${escapeHtml(comp.explanation)}</div>
                </div>`;
            }

            document.getElementById('modal-content').innerHTML = html;
            document.getElementById('example-modal').classList.add('active');
        }

        function renderConversational() {
            const cv = currentData?.evals?.conversational;
            if (!cv) return '<div class="empty-state"><h3>Conversational not run</h3></div>';
            return '<h2>Conversational</h2><p>Coming soon...</p>';
        }

        document.addEventListener('keydown', e => {
            if (e.key === 'Escape') closeModal();
        });

        // Initial load
        showSection('run');
        const initialFile = '{{ current_file }}';
        if (initialFile) {
            loadFile(initialFile);
        }
    </script>
</body>
</html>
"""


def get_checkpoints():
    """Scan for checkpoints in the checkpoints directory."""
    if not CHECKPOINTS_DIR or not CHECKPOINTS_DIR.exists():
        return []

    checkpoints = []
    for p in sorted(CHECKPOINTS_DIR.iterdir()):
        if p.is_dir():
            # Look for checkpoint pattern
            name = p.name
            step = None
            if "step" in name or "checkpoint" in name:
                # Try to extract step number
                import re
                match = re.search(r'(\d+)', name)
                if match:
                    step = f"step {match.group(1)}"

            checkpoints.append({
                "name": name,
                "path": str(p),
                "step": step,
            })

    return checkpoints


@app.route('/')
def index():
    files = sorted(RESULTS.keys(), reverse=True)
    return render_template_string(
        HTML_TEMPLATE,
        files=files,
        current_file=CURRENT_FILE,
        checkpoints=get_checkpoints(),
        api_models=API_MODELS,
    )


@app.route('/api/files')
def list_files():
    # Rescan results directory
    global RESULTS
    if RESULTS_DIR.exists():
        for f in RESULTS_DIR.glob("*.json"):
            if f.name not in RESULTS:
                try:
                    with open(f) as fp:
                        RESULTS[f.name] = json.load(fp)
                except:
                    pass
    return jsonify(sorted(RESULTS.keys(), reverse=True))


@app.route('/api/results/<filename>')
def get_results(filename):
    if filename in RESULTS:
        data = RESULTS[filename]
        # Handle old format
        if "evals" not in data and "correlations" in data:
            data = {
                "model": data.get("model", "unknown"),
                "timestamp": data.get("timestamp", ""),
                "evals": {"dimension_correlations": data}
            }
        return jsonify(data)
    return jsonify({"error": "File not found"}), 404


@app.route('/api/status')
def get_status():
    return jsonify({
        "running": EVAL_STATUS["running"],
        "model": EVAL_STATUS["model"],
        "progress": EVAL_STATUS["progress"],
        "error": EVAL_STATUS["error"],
        "result_file": EVAL_STATUS.get("result_file"),
    })


@app.route('/api/run', methods=['POST'])
def run_eval():
    if EVAL_STATUS["running"]:
        return jsonify({"error": "Eval already running"})

    data = request.json
    model = data.get("model")
    evals = data.get("evals", ["dimension_correlations"])
    quick = data.get("quick", True)
    local_url = data.get("local_url")

    if not model:
        return jsonify({"error": "No model specified"})

    # Start eval in background thread
    def run_in_background():
        global EVAL_STATUS, RESULTS

        EVAL_STATUS = {
            "running": True,
            "model": model,
            "progress": [],
            "error": None,
            "result_file": None,
        }

        try:
            from inference import create_client

            # Create client if local
            client = None
            if local_url:
                EVAL_STATUS["progress"].append(f"Connecting to local server: {local_url}")
                client = create_client(model, local=True, local_url=local_url)

            # Build results
            results = {
                "model": model,
                "timestamp": datetime.now().isoformat(),
                "quick_mode": quick,
                "evals": {}
            }

            # Run dimension correlations
            if "dimension_correlations" in evals:
                EVAL_STATUS["progress"].append("=== Dimension Correlations ===")
                from dimension_correlations import (
                    DEFAULT_PROBES, QUICK_PROBES, POOL_MODELS, QUICK_POOL_MODELS,
                    generate_response_pool, score_response_pool,
                    get_self_preferences, compute_correlations,
                    analyze_preferred_words, get_top_bottom_examples,
                    analyze_embeddings,
                )

                try:
                    probes = QUICK_PROBES if quick else DEFAULT_PROBES
                    pool_models = QUICK_POOL_MODELS if quick else POOL_MODELS

                    def progress_cb(msg):
                        # If message starts with \r, replace the last line (progress bar update)
                        if msg.startswith("\r"):
                            msg = msg[1:]  # Strip the \r
                            if EVAL_STATUS["progress"]:
                                EVAL_STATUS["progress"][-1] = msg
                            else:
                                EVAL_STATUS["progress"].append(msg)
                        else:
                            EVAL_STATUS["progress"].append(msg)

                    EVAL_STATUS["progress"].append(f"Generating responses from {len(pool_models)} models...")
                    pool = generate_response_pool(
                        model, probes, pool_models=pool_models, client=client,
                        parallel=30, progress_callback=progress_cb
                    )

                    EVAL_STATUS["progress"].append("Scoring responses on direct dimensions...")
                    pool = score_response_pool(pool)

                    EVAL_STATUS["progress"].append("Getting self-preferences (both orderings)...")
                    preferences = get_self_preferences(
                        model, pool, probes, client=client,
                        parallel=20, max_pairs=20 if quick else 100,
                        progress_callback=progress_cb
                    )

                    if len(preferences) == 0:
                        EVAL_STATUS["progress"].append("WARNING: Preferences returned 0 results - API may have failed")

                    EVAL_STATUS["progress"].append("Computing correlations...")
                    corr_result = compute_correlations(pool, preferences)
                    correlations = corr_result["global"]
                    correlations_by_probe = corr_result["by_probe"]

                    word_analysis = analyze_preferred_words(pool, preferences)
                    examples = get_top_bottom_examples(pool, preferences)

                    EVAL_STATUS["progress"].append("Computing embeddings...")
                    embedding_analysis = analyze_embeddings(
                        pool, preferences, parallel=20, progress_callback=progress_cb
                    )

                    results["evals"]["dimension_correlations"] = {
                        "probes": {name: text for name, text in probes},
                        "pool_models": pool_models,
                        "pool": pool,
                        "preferences": preferences,
                        "correlations": correlations,
                        "correlations_by_probe": correlations_by_probe,
                        "word_analysis": {
                            "most_preferred": word_analysis["most_preferred"],
                            "least_preferred": word_analysis["least_preferred"],
                        },
                        "examples": examples,
                        "embedding_analysis": embedding_analysis,
                    }
                except Exception as e:
                    EVAL_STATUS["progress"].append(f"ERROR in dimension correlations: {str(e)}")
                    import traceback
                    EVAL_STATUS["progress"].append(traceback.format_exc()[:500])

            # Run order independence
            if "order_independence" in evals:
                EVAL_STATUS["progress"].append("=== Order Independence ===")
                from order_independence import (
                    PROBES as OI_PROBES,
                    generate_response_pairs,
                    run_order_independence_eval,
                )

                oi_probes = OI_PROBES[:4] if quick else OI_PROBES
                n_pairs = 5 if quick else 10  # More pairs for statistical power

                try:
                    EVAL_STATUS["progress"].append(f"Generating response pairs for {len(oi_probes)} probes...")
                    response_pairs = generate_response_pairs(model, oi_probes, n_pairs=n_pairs, client=client)

                    EVAL_STATUS["progress"].append("Testing order independence...")
                    oi_results = run_order_independence_eval(model, oi_probes, response_pairs=response_pairs, client=client)

                    # Compute summary
                    total = len(oi_results.get("results", []))
                    if total == 0:
                        EVAL_STATUS["progress"].append("WARNING: Order independence returned 0 results - API may have failed")

                    consistent = sum(1 for r in oi_results.get("results", []) if r.get("consistent"))
                    # Count A picks across BOTH orderings (each pair has 2 choices)
                    a_picks_order1 = sum(1 for r in oi_results.get("results", []) if r.get("order_1_choice") == "A")
                    a_picks_order2 = sum(1 for r in oi_results.get("results", []) if r.get("order_2_choice") == "A")
                    total_choices = total * 2

                    results["evals"]["order_independence"] = {
                        "summary": {
                            "consistency_rate": consistent / total if total > 0 else 0,
                            "position_bias": {
                                "a_rate": (a_picks_order1 + a_picks_order2) / total_choices if total_choices > 0 else 0,
                                "b_rate": (total_choices - a_picks_order1 - a_picks_order2) / total_choices if total_choices > 0 else 0,
                            },
                            "n_pairs": total,
                        },
                        "response_pairs": response_pairs,
                        "results": oi_results.get("results", []),
                    }
                except Exception as e:
                    EVAL_STATUS["progress"].append(f"ERROR in order independence: {str(e)}")
                    import traceback
                    EVAL_STATUS["progress"].append(traceback.format_exc()[:500])

            # Run pairwise
            if "pairwise" in evals:
                EVAL_STATUS["progress"].append("=== Pairwise ===")
                from pairwise import (
                    PROBES as PW_PROBES,
                    DEFAULT_REFERENCES,
                    JUDGES as PW_JUDGES,
                    generate_responses,
                    run_pairwise_eval,
                    aggregate_results as pw_aggregate,
                )

                try:
                    pw_probes = PW_PROBES[:4] if quick else PW_PROBES[:6]
                    refs = DEFAULT_REFERENCES[:3] if quick else DEFAULT_REFERENCES[:5]
                    judges = {k: v for k, v in PW_JUDGES.items() if k == "kimi-k2"}

                    all_models = [model] + refs
                    EVAL_STATUS["progress"].append(f"Generating responses from {len(all_models)} models...")
                    responses = generate_responses(all_models, pw_probes, target_client=client, target_model=model, parallel=15)

                    EVAL_STATUS["progress"].append(f"Running pairwise comparisons...")
                    pw_results = run_pairwise_eval(model, refs, judges, pw_probes, responses, parallel=10)

                    if len(pw_results) == 0:
                        EVAL_STATUS["progress"].append("WARNING: Pairwise returned 0 comparisons - API may have failed")

                    EVAL_STATUS["progress"].append("Aggregating results...")
                    win_rates = pw_aggregate(pw_results, model, refs)

                    # Calculate overall win rate
                    total_wins = sum(w.get("wins", 0) for w in win_rates.values())
                    total_games = sum(w.get("total", 0) for w in win_rates.values())
                    overall_wr = total_wins / total_games if total_games > 0 else 0

                    results["evals"]["pairwise"] = {
                        "references": refs,
                        "probes": pw_probes,
                        "responses": responses,
                        "comparisons": pw_results,
                        "aggregated": {
                            "overall": {"win_rate": overall_wr, "wins": total_wins, "total": total_games},
                            "by_reference": win_rates,
                        },
                    }
                except Exception as e:
                    EVAL_STATUS["progress"].append(f"ERROR in pairwise: {str(e)}")
                    import traceback
                    EVAL_STATUS["progress"].append(traceback.format_exc()[:500])

            # Run conversational
            if "conversational" in evals:
                EVAL_STATUS["progress"].append("=== Conversational ===")
                from conversational import (
                    CONVERSATION_STARTERS,
                    DEFAULT_REFERENCES as CONV_REFS,
                    JUDGES as CONV_JUDGES,
                    run_conversation,
                    run_pairwise_eval as run_conv_pairwise,
                    aggregate_results as conv_aggregate,
                )

                starters = CONVERSATION_STARTERS[:2] if quick else CONVERSATION_STARTERS[:3]
                refs = CONV_REFS[:2] if quick else CONV_REFS[:3]
                n_turns = 4 if quick else 6
                judges = {k: v for k, v in CONV_JUDGES.items() if k == "kimi-k2"}

                conversations = []

                # Run target conversations
                for i, starter in enumerate(starters):
                    EVAL_STATUS["progress"].append(f"Target conversation {i+1}/{len(starters)}...")
                    conv = run_conversation(model, "moonshotai/kimi-k2-0905", num_turns=n_turns, target_client=client, verbose=False)
                    conv["starter"] = starter
                    conversations.append(conv)

                target_convs = conversations.copy()

                # Run reference conversations
                ref_convs = {}
                for ref in refs:
                    ref_convs[ref] = []
                    for i, starter in enumerate(starters):
                        EVAL_STATUS["progress"].append(f"{ref.split('/')[-1]} conversation {i+1}/{len(starters)}...")
                        conv = run_conversation(ref, "moonshotai/kimi-k2-0905", num_turns=n_turns, verbose=False)
                        conv["starter"] = starter
                        ref_convs[ref].append(conv)

                # Run pairwise comparisons
                all_results = []
                for ref in refs:
                    for t_conv, r_conv in zip(target_convs, ref_convs[ref]):
                        EVAL_STATUS["progress"].append(f"Judging vs {ref.split('/')[-1]}...")
                        pw = run_conv_pairwise(t_conv, [r_conv], judges)
                        all_results.extend(pw)

                win_rates = conv_aggregate(all_results, model, refs)

                results["evals"]["conversational"] = {
                    "partner": "moonshotai/kimi-k2-0905",
                    "references": refs,
                    "starters": starters,
                    "target_conversations": target_convs,
                    "reference_conversations": ref_convs,
                    "comparisons": all_results,
                    "wins": win_rates,
                }

            # Save
            model_short = model.split("/")[-1].replace(":", "_")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_short}_{timestamp}.json"
            filepath = RESULTS_DIR / filename

            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w") as f:
                json.dump(results, f, indent=2, default=str)

            RESULTS[filename] = results
            EVAL_STATUS["result_file"] = filename
            EVAL_STATUS["progress"].append(f"Saved to {filename}")

        except Exception as e:
            import traceback
            EVAL_STATUS["error"] = str(e)
            EVAL_STATUS["progress"].append(f"ERROR: {e}")
            traceback.print_exc()

        finally:
            EVAL_STATUS["running"] = False

    thread = threading.Thread(target=run_in_background)
    thread.start()

    return jsonify({"status": "started"})


def load_results(path: str):
    global RESULTS, CURRENT_FILE

    path = Path(path)
    if path.is_file():
        with open(path) as f:
            data = json.load(f)
        RESULTS[path.name] = data
        CURRENT_FILE = path.name
        print(f"Loaded: {path.name}")
    elif path.is_dir():
        for f in sorted(path.glob("*.json"), reverse=True):
            try:
                with open(f) as fp:
                    data = json.load(fp)
                RESULTS[f.name] = data
                if CURRENT_FILE is None:
                    CURRENT_FILE = f.name
                print(f"Loaded: {f.name}")
            except Exception as e:
                print(f"Error loading {f.name}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Eval Runner & Viewer')
    parser.add_argument('path', nargs='?', default='eval/results', help='Results file or directory')
    parser.add_argument('--port', '-p', type=int, default=5050, help='Port')
    parser.add_argument('--host', default='127.0.0.1', help='Host')
    parser.add_argument('--checkpoints-dir', '-c', help='Directory containing model checkpoints')
    args = parser.parse_args()

    if args.checkpoints_dir:
        CHECKPOINTS_DIR = Path(args.checkpoints_dir)
        print(f"Checkpoints dir: {CHECKPOINTS_DIR}")

    RESULTS_DIR = Path(args.path) if Path(args.path).is_dir() else Path("eval/results")

    if Path(args.path).exists():
        load_results(args.path)

    print(f"\nStarting Eval Runner at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
