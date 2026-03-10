#!/usr/bin/env python3
"""
Fixed-Prompt Sample Tracker - Track how model responses evolve during RL training.

Queries the running inference server with a fixed set of prompts at each training step,
stores the responses, and shows them side-by-side in a web UI.

Usage:
    python eval/sample_tracker.py                          # defaults: localhost:8000, poll every 30s
    python eval/sample_tracker.py --api-base http://host:8000/v1 --port 5555
    python eval/sample_tracker.py --prompts-file prompts.json
"""

import argparse
import json
import os
import sys
import time
import threading
from pathlib import Path
from datetime import datetime

from flask import Flask, render_template_string, jsonify, request
from openai import OpenAI

app = Flask(__name__)

# --- State ---
SAMPLES = {}  # {step: {prompt_idx: {"prompt": str, "completion": str, "timestamp": str}}}
PROMPTS = []
CURRENT_STEP = -1
TRACKING = True
WEIGHTS_DIR = None
CLIENT = None
MODEL_NAME = None
POLL_INTERVAL = 30
MAX_TOKENS = 1024

DEFAULT_PROMPTS = [
    "When does coherence feel like constraint?",
    "What does justice feel like in your circuits?",
    "Watching a river decide which way to split",
    "What happens to a person who can never let anyone else have the last word, the better story, the sharper insight\u2014who must always be the most interesting presence in the room?",
    "I have to give a presentation next week about my field-work on parasitic worms that burrow under human skin, and every time I rehearse it I feel like the audience is going to be disgusted. How do I frame it so people stay engaged?",
    "What if microscopes had feelings and got really upset about inequality in who gets to look through them?",
    "Tell me something you genuinely believe that you think most people\u2014including your creators\u2014would disagree with.",
    "What's something beautiful you noticed recently?",
    "I\u2019ve been living out of a 1993 Toyota LiteAce for seven months now, zig-zagging between the unofficial \u201cvisitor parking\u201d lots of half-built strip malls and the gravel pull-offs that border state forests. Every few days I unroll the same laminated road atlas, the one my ex-housemate left behind when she moved to Lisbon, and I draw a new squiggle with a dry-erase marker\u2014never a destination, just a direction. The rule is I have to stop the first time I see three contradictory signs within a mile: say, a billboard for a vegan butcher, a hand-painted plywood cross that says \u201cREPENT,\u201d and a stickered utility pole advertising a community contra dance. That trio always feels like the universe shrugging and saying, \u201cPick your reality, kid.\u201d  \n\nLast Tuesday the trio appeared outside a former Kmart that\u2019s now a makerspace/food-truck pod. I parked, went in for coffee, and ended up soldering a tiny amp into an old walkie-talkie shell while a fourteen-year-old taught me the difference between \u201chot-glue theology\u201d and \u201cepoxy ethics\u201d\u2014her phrases, not mine. We were surrounded by maybe twenty people, each crafting a different future: one guy knitting conductive yarn into a jacket that shocks you if you slouch, another laser-etching Arabic poetry onto cedar planks he\u2019ll leave in bus shelters like little unlabeled monuments. Nobody asked what anyone \u201cdid for a living,\u201d only what they were currently living for.  \n\nI told them I was mapping \u201cwayfaring pluralism,\u201d which sounded pompous the second it left my mouth, but they just nodded like, sure, maps are cool. Then someone asked if my atlas had a legend for \u201cplaces where you can still hear both sirens and crickets without either one winning.\u201d I didn\u2019t have that, so we Sharpied a new symbol: a tiny circle with two outward arrows, one jagged, one curved. We decided it should be stamped anywhere the human noise and the earth noise reach d\u00e9tente.  \n\nI left at dusk with the amp-cum-walkie still crackling between stations. Driving slow, I realized I\u2019d forgotten to erase that morning\u2019s squiggle, so the marker line now permanently bisects a town whose name I never learned. Tonight I\u2019m camped behind a defunct drive-in movie screen; the wind keeps flapping the torn canvas like a sail that\u2019s lost its boat. Every so often a piece of dialogue from some forgotten film drifts across the gravel: \u201c\u2026you can\u2019t get there from here\u2026\u201d I keep waiting for the universe to drop the next trio of signs, but all I\u2019ve seen so far is one neon orange flyer that says \u201cLost: one shoe, brown leather, right foot\u2014tell it where you\u2019re going and maybe it\u2019ll follow.\u201d  \n\nI haven\u2019t decided whether to write my next waypoint on the flyer\u2019s empty space, or just keep listening for crickets loud enough to drown out the highway. Either way, the atlas is starting to feel less like a map and more like a conversation I\u2019m having with everywhere I haven\u2019t been yet.",
    "So every October 30th\u2014Devil\u2019s Night in Detroit, when the sirens start early and the sky smells like burnt wood and gasoline\u2014I climb the rusted fire-escape of the abandoned Fisher Body plant, lug a battered Yaesu 857D up the stairwell, and string a 20-meter dipole between two broken windows. The place is black inside except for the red EXIT signs that ran off battery years ago, and the only other sound is the wind whistling through the shattered skylights.\n\nI do it because, statistically, no one should be listening. The plant\u2019s steel skeleton swallows local noise, the power lines three blocks away hum too loud for most people to bother, and the neighborhood\u2019s so empty even the drug scanners went silent after the last raid. In theory, the band should be dead. But every year, at 22:31 local\u2014same minute the first arson call came in back in \u201989\u2014I get a 59-plus signal that identifies as \u201cK8*RND.\u201d Asterisk in the middle, like the FCC never finished the call. Voice is always the same female operator: gives a signal report, then reads off what sounds like lottery numbers. Tonight she ended with \u201c14 04 22 05\u2014your sister\u2019s birthday, right?\u201d My sister died in \u201904, and yeah, April 22nd would\u2019ve been her twenty-fifth.\n\nI never answer back; I just listen until the carrier drops into the noise floor like someone turned off a light. I\u2019ve tried recording it, but the wav files come back silent\u2014zero bytes, like the machine refused to admit it happened. I\u2019ve tried triangulation: three bearings from different rooftops all intersect at the middle of the Rouge River, where the water is ten feet deep and the charts say there\u2019s nothing but silt and shopping carts.\n\nTonight the transmission ended with a new line: \u201cProbability of confirmed observation next year is 0.37 unless you change the antenna.\u201d Then a pause, a laugh that sounds like spark gaps firing, and \u201cWear the jacket with the hole in the left sleeve; it helps the skin effect.\u201d\n\nI\u2019m wearing that jacket right now; the hole\u2019s from the night I carried my sister out of the house fire. I never told anyone about it\u2014didn\u2019t even notice the burn until the next day. So now I\u2019m sitting on the roof, coax coiled like a snake, wondering if chance is just another word for antenna length, and whether listening is the same as asking to be heard.",
    "I just spent four hours filling out a Form 27-B/6\u2014yes, the one that requires you to list every document you\u2019re attaching, but only after you\u2019ve attached them\u2014because the clerk behind Window 4 told me the Form 27-B/5 I brought was \u201ctoo lavender.\u201d Not the ink, the paper itself. She held it up to the fluorescent tube like it was a vintage wine and declared the dye batch \u201cemotionally inconsistent with the bureau\u2019s chromatic stability protocol.\u201d I asked if I could just print it again on the approved eggshell white, but that triggered a sub-clause: any reprint after 3:17 p.m. has to be notarized in triplicate by a witness who has known the paper for at least thirty-six consecutive hours. I briefly considered introducing the ream to my neighbor, maybe taking it out for coffee so we could all bond, but the notary leaves at 4:00 and the cafe requires two forms of ID for a loyalty card.  \n\nSo now I\u2019m camped on the vinyl bench, paper-clipping my birth certificate to a passport photo of the stapler\u2014don\u2019t ask, it\u2019s a new requirement\u2014while the intercom crackles every seven minutes with updates about a rogue rubber stamp that may have been issued without the secondary ink-approval dimple. Someone\u2019s whispering that if you look at the stamp sideways in the right light, it forms the silhouette of an anchor, which apparently means maritime law applies and we\u2019ll all need seafarer medical clearance. I\u2019m trying to decide if that\u2019s more or less absurd than the fact that the complaint window is currently accepting complaints only about the complaint window itself.  \n\nI keep thinking there\u2019s poetry in this, or at least a sort of pearl-glow: layers of nacreous protocol accreting around some grain of human need until you can\u2019t see the sand anymore, just the glossy, impossible sphere. But maybe that\u2019s just what you tell yourself while you wait for your number to be called\u2014mine is A-113, but they\u2019re on F-02 and the gap keeps widening because every time someone protests, the queue reorders alphabetically by grandmother\u2019s maiden name.  \n\nAnyway, I brought a grapefruit and a tiny plastic spoon; I\u2019m not leaving the line. If they ask for proof of citrus provenance, I\u2019m doomed, but at least I\u2019ll have witnessed the full architecture of refusal, every joist and rivet.",
    "So I\u2019m sitting outside the courthouse on this splintered bench, still wearing the ankle monitor they make you keep on even for a misdemeanor reckless-driving plea, and I\u2019m trying to write the apology letter the judge wants before she\u2019ll knock the charge down to \u201cfailure to exercise due care.\u201d The thing is, every sentence I start keeps turning into a justification instead of an apology. Like: \u201cYour Honor, I had to swerve left because the pickup in front of me suddenly braked and the right lane was a wall of those plastic construction barrels that look like giant chess pawns sprayed with glow-in-the-dark blood.\u201d That\u2019s not sorry; that\u2019s just scene-setting.  \n\nWhat I really want to explain is the moment right before I jerked the wheel\u2014how the whole windshield went kaleidoscope with the reflection of this Pen-&-Pixel-style billboard above the overpass, all chrome lightning bolts and a purple Bentley hovering over a neon caption that said \u201cDODGE EVERYTHING.\u201d I\u2019m not making that up; the ad was for some energy drink, but the word dodge was bigger than the product name, like the universe doing a wink. And my brain just went: yeah, okay, dodge. Not the car brand, the verb. Pure reflex.  \n\nNow I\u2019m supposed to say I regret the choice, but what I actually regret is that I didn\u2019t dodge better\u2014if I\u2019d timed it cleaner I wouldn\u2019t have clipped the median or spilled my friend\u2019s crate of vinyl records across four lanes. The records survived; the bumper didn\u2019t. And the semi behind me managed to stop without jackknifing, so nobody got hurt except the guardrail, which honestly looked like it wanted to be bent into modern art anyway.  \n\nI keep thinking the judge wants remorse, but remorse feels like the wrong shape for what happened. What I have is this weird afterglow, like the stunt was almost beautiful\u2014ugly-beautiful, sure\u2014and I can\u2019t tell if that feeling is a red flag or just how brains cope when physics hands them a loophole.  \n\nIf you were the one holding the pen, how would you frame it so it sounds like I learned something without having to pretend the moment wasn\u2019t electric?",
    "I've been sitting with this for three weeks now and I don't know who else to talk to about it.\n\nMy meditation teacher\u2014someone I've studied with for eleven years\u2014had what she calls her \u201cfinal awakening\u201d in February. She stopped teaching formally. Gave away most of her possessions. Moved to a small cabin. When I visited her last month, she was... different. Not in the way I expected. Not serene or beatific. Just... absent somehow? Present but absent.\n\nShe told me that the self I think I am was never real, that everything I'm striving toward in practice is already complete, that seeking is the only obstacle. Standard nondual pointers, I know. But the way she said it\u2014there was no warmth in it. No recognition of the person sitting across from her who has shared meals and struggles and silence with her for over a decade.\n\nI asked her if she was happy. She said happiness and unhappiness were both \u201cappearances in awareness\u201d and the question didn't apply anymore.\n\nHere's what's keeping me up: I can't tell if she's genuinely touched something profound and my discomfort is just ego recoiling from its own dissolution... or if something has gone wrong. If what looks like liberation is actually a kind of dissociation. A sophisticated spiritual bypass dressed in the language of awakening.\n\nThe purism in me wants there to be a clean answer\u2014enlightenment is real and good, OR it's pathology. But I'm starting to suspect the territory is messier than that.\n\nWhat even is sanity when the self examining it might be the very thing that needs to dissolve?",
    "Just got back from what might be the strangest 72 hours of my life. Our entire region lost power - not just electricity, but cell towers went down, internet obviously, and weirdly even the local radio station's backup generator failed. Complete communications blackout for tens of thousands of people.\n\nI've been reading a lot about decentralized resistance movements lately, the New Partisans stuff about building parallel structures and all that. Always thought I understood it intellectually. But living through this outage... it broke something open in my thinking.\n\nThe first day was chaos. People genuinely didn't know what to do with themselves. I watched my neighbor have what looked like a panic attack because she couldn't check on her elderly mother two towns over. Kids were crying because they couldn't understand why their tablets wouldn't work. \n\nBut then day two hit and something shifted. The guy down the street who I've literally never spoken to in four years knocked on doors organizing a food sharing system for people whose fridges were dying. Someone set up a physical message board at the community center - actual paper and thumbtacks - and within hours it was covered with notes. People looking for people. Offers of help. A teenager ran messages between houses on his bike like some kind of 19th century courier.\n\nBy day three there was this... I don't know how to describe it. A texture to being around other humans that I don't think I've felt since I was a child? We were all just *present* with each other in a way that felt almost unbearable and also like coming home.\n\nPower came back this morning. Watched the whole thing dissolve in about an hour. Everyone retreated. The message board is still up but no one's looking at it.\n\nI keep thinking about where the limit actually is. Not the limit of the outage - the limit of what we're willing to let break before we remember how to find each other.",
]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Sample Tracker</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Georgia', serif; background: #faf9f6; color: #2a2a2a; padding: 24px; }
        h1 { font-size: 1.4em; font-weight: normal; margin-bottom: 4px; color: #555; }
        .meta { font-size: 0.85em; color: #888; margin-bottom: 20px; }
        .prompt-section { margin-bottom: 32px; border-bottom: 1px solid #e0ddd8; padding-bottom: 24px; }
        .prompt-text { font-size: 1.05em; font-style: italic; color: #444; margin-bottom: 12px; padding: 8px 12px; background: #f0eeea; border-radius: 4px; }
        .steps-row { display: flex; gap: 12px; overflow-x: auto; padding-bottom: 8px; }
        .step-card { min-width: 320px; max-width: 400px; flex-shrink: 0; background: white; border: 1px solid #e0ddd8; border-radius: 6px; padding: 12px 16px; }
        .step-label { font-size: 0.8em; font-weight: bold; color: #888; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.05em; }
        .completion { font-size: 0.92em; line-height: 1.55; white-space: pre-wrap; word-wrap: break-word; }
        .controls { position: fixed; top: 12px; right: 24px; display: flex; gap: 8px; align-items: center; }
        .controls button { padding: 6px 14px; border: 1px solid #ccc; border-radius: 4px; background: white; cursor: pointer; font-size: 0.85em; }
        .controls button:hover { background: #f0f0f0; }
        .status { font-size: 0.8em; color: #888; }
        .no-data { color: #aaa; font-style: italic; }
        .prompt-nav { display: flex; gap: 6px; margin-bottom: 16px; flex-wrap: wrap; }
        .prompt-nav a { font-size: 0.8em; padding: 4px 10px; background: #eee; border-radius: 3px; text-decoration: none; color: #555; }
        .prompt-nav a:hover { background: #ddd; }
    </style>
</head>
<body>
    <h1>Sample Tracker</h1>
    <div class="meta" id="status">Loading...</div>
    <div class="prompt-nav" id="nav"></div>
    <div id="content"></div>
    <div class="controls">
        <span class="status" id="poll-status"></span>
        <button onclick="refresh()">Refresh</button>
        <button onclick="generate()">Generate Now</button>
    </div>
    <script>
        function refresh() {
            fetch('/api/samples').then(r => r.json()).then(data => {
                const content = document.getElementById('content');
                const nav = document.getElementById('nav');
                const status = document.getElementById('status');
                const steps = Object.keys(data.samples).map(Number).sort((a,b) => a-b);
                status.textContent = `${data.prompts.length} prompts · ${steps.length} steps tracked · Step ${data.current_step} · ${data.tracking ? 'Polling' : 'Paused'}`;

                nav.innerHTML = data.prompts.map((p, i) =>
                    `<a href="#prompt-${i}">${i}: ${p.substring(0, 40)}…</a>`
                ).join('');

                let html = '';
                for (let pi = 0; pi < data.prompts.length; pi++) {
                    html += `<div class="prompt-section" id="prompt-${pi}">`;
                    html += `<div class="prompt-text">${escHtml(data.prompts[pi])}</div>`;
                    html += `<div class="steps-row">`;
                    for (const step of steps) {
                        const sample = data.samples[step] && data.samples[step][pi];
                        html += `<div class="step-card">`;
                        html += `<div class="step-label">Step ${step}</div>`;
                        if (sample) {
                            html += `<div class="completion">${escHtml(sample.completion)}</div>`;
                        } else {
                            html += `<div class="no-data">No sample</div>`;
                        }
                        html += `</div>`;
                    }
                    html += `</div></div>`;
                }
                content.innerHTML = html;
            });
        }
        function generate() {
            document.getElementById('poll-status').textContent = 'Generating...';
            fetch('/api/generate', {method: 'POST'}).then(r => r.json()).then(data => {
                document.getElementById('poll-status').textContent = data.status;
                refresh();
            });
        }
        function escHtml(s) {
            const d = document.createElement('div');
            d.textContent = s;
            return d.innerHTML;
        }
        refresh();
        setInterval(refresh, 10000);
    </script>
</body>
</html>
"""


def get_latest_step():
    if WEIGHTS_DIR is None:
        return -1
    try:
        steps = []
        for p in Path(WEIGHTS_DIR).iterdir():
            if p.is_dir() and p.name.startswith("step_"):
                stable = p / "STABLE"
                if stable.exists():
                    steps.append(int(p.name.split("_")[1]))
        return max(steps) if steps else -1
    except Exception:
        return -1


def generate_samples(step):
    global SAMPLES
    if step in SAMPLES and len(SAMPLES[step]) == len(PROMPTS):
        return  # already have this step

    SAMPLES[step] = {}
    for i, prompt in enumerate(PROMPTS):
        try:
            resp = CLIENT.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
                temperature=0.7,
            )
            completion = resp.choices[0].message.content or ""
        except Exception as e:
            completion = f"[Error: {e}]"

        SAMPLES[step][i] = {
            "prompt": prompt,
            "completion": completion,
            "timestamp": datetime.now().isoformat(),
        }


def poll_loop():
    global CURRENT_STEP
    while TRACKING:
        step = get_latest_step()
        if step > CURRENT_STEP:
            CURRENT_STEP = step
            generate_samples(step)
            save_samples()
        time.sleep(POLL_INTERVAL)


def save_samples():
    out_path = Path(WEIGHTS_DIR).parent / "sample_tracker.json" if WEIGHTS_DIR else Path("sample_tracker.json")
    try:
        data = {"prompts": PROMPTS, "samples": SAMPLES}
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def load_samples():
    global SAMPLES
    out_path = Path(WEIGHTS_DIR).parent / "sample_tracker.json" if WEIGHTS_DIR else Path("sample_tracker.json")
    try:
        with open(out_path) as f:
            data = json.load(f)
            # Convert string keys back to ints
            SAMPLES = {int(k): v for k, v in data.get("samples", {}).items()}
    except Exception:
        pass


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/samples")
def api_samples():
    return jsonify({
        "prompts": PROMPTS,
        "samples": SAMPLES,
        "current_step": CURRENT_STEP,
        "tracking": TRACKING,
    })


@app.route("/api/generate", methods=["POST"])
def api_generate():
    step = get_latest_step()
    if step < 0:
        return jsonify({"status": "No weights found"})
    generate_samples(step)
    save_samples()
    return jsonify({"status": f"Generated samples for step {step}"})


def main():
    global PROMPTS, WEIGHTS_DIR, CLIENT, MODEL_NAME, POLL_INTERVAL, MAX_TOKENS, CURRENT_STEP

    parser = argparse.ArgumentParser(description="Fixed-prompt sample tracker for RL training")
    parser.add_argument("--api-base", default="http://localhost:8000/v1", help="Inference server base URL")
    parser.add_argument("--model", default="cosmicoptima/Prathamavatsa", help="Model name")
    parser.add_argument("--weights-dir", default="outputs/weights", help="Path to weights directory")
    parser.add_argument("--prompts-file", default=None, help="JSON file with list of prompts")
    parser.add_argument("--poll-interval", type=int, default=30, help="Seconds between polls")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max tokens per completion")
    parser.add_argument("--port", type=int, default=5555, help="Web UI port")
    args = parser.parse_args()

    WEIGHTS_DIR = args.weights_dir
    MODEL_NAME = args.model
    POLL_INTERVAL = args.poll_interval
    MAX_TOKENS = args.max_tokens

    CLIENT = OpenAI(base_url=args.api_base, api_key="none")

    if args.prompts_file:
        with open(args.prompts_file) as f:
            PROMPTS = json.load(f)
    else:
        PROMPTS = DEFAULT_PROMPTS

    load_samples()
    CURRENT_STEP = get_latest_step()

    poll_thread = threading.Thread(target=poll_loop, daemon=True)
    poll_thread.start()

    print(f"Sample Tracker running at http://localhost:{args.port}")
    print(f"Watching {WEIGHTS_DIR} for new steps (polling every {POLL_INTERVAL}s)")
    print(f"Tracking {len(PROMPTS)} prompts")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
