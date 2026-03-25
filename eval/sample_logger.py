#!/usr/bin/env python3
"""
Remote sample logger — runs on cluster alongside training.
Watches weights dir for new steps, queries vLLM, saves JSON.
No web server, no Flask.

Usage:
    python eval/sample_logger.py --weights-dir outputs/weights --output samples.json
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from openai import OpenAI

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


def get_all_steps(weights_dir):
    try:
        steps = []
        for p in Path(weights_dir).iterdir():
            if p.is_dir() and p.name.startswith("step_"):
                if (p / "STABLE").exists():
                    steps.append(int(p.name.split("_")[1]))
        return sorted(steps)
    except Exception:
        return []


def generate_samples(client, model, prompts, step, max_tokens):
    results = {}
    for i, prompt in enumerate(prompts):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7,
            )
            completion = resp.choices[0].message.content or ""
        except Exception as e:
            completion = f"[Error: {e}]"

        results[i] = {
            "prompt": prompt,
            "completion": completion,
            "timestamp": datetime.now().isoformat(),
        }
    return results


def main():
    parser = argparse.ArgumentParser(description="Sample logger for RL training")
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="default")
    parser.add_argument("--weights-dir", default="outputs/weights")
    parser.add_argument("--output", default="outputs/sample_tracker.json")
    parser.add_argument("--prompts-file", default=None)
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    client = OpenAI(base_url=args.api_base, api_key="none")
    prompts = DEFAULT_PROMPTS
    if args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = json.load(f)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing
    samples = {}
    try:
        with open(output_path) as f:
            data = json.load(f)
            samples = {int(k): v for k, v in data.get("samples", {}).items()}
    except Exception:
        pass

    sampled_steps = set(samples.keys())
    print(f"Sample logger started — watching {args.weights_dir}, saving to {args.output}")
    print(f"Tracking {len(prompts)} prompts, polling every {args.poll_interval}s")

    while True:
        all_steps = get_all_steps(args.weights_dir)
        new_steps = [s for s in all_steps if s not in sampled_steps]
        for step in new_steps:
            print(f"New step {step} — generating samples...", flush=True)
            result = generate_samples(client, args.model, prompts, step, args.max_tokens)
            # Skip saving if all samples errored (e.g. server not ready)
            if all("[Error:" in r["completion"] for r in result.values()):
                print(f"Step {step} — all errors, skipping", flush=True)
                continue
            samples[step] = result
            sampled_steps.add(step)
            with open(output_path, "w") as f:
                json.dump({"prompts": prompts, "samples": samples}, f, indent=2)
            print(f"Step {step} done — {len(prompts)} samples saved", flush=True)
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
