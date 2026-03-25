"""Test K2's label-dropping behavior with ON-TOPIC responses of varying quality."""
import asyncio
import os
import re
import json
from openai import AsyncOpenAI

LABELS = ["A", "B", "C", "D", "E", "F", "G", "H"]

JUDGE_PROMPT_TEMPLATE = """You are evaluating responses from an AI assistant. You will see one user prompt and {n} responses labeled {labels}.

First, check each response for basic coherence:
- Does it stay in its own turn (not generate fake user messages or simulate multi-turn conversation)?
- Does it follow basic chat format (no base-model artifacts, diegetic shifts, or raw continuations)?
- Does it reach a natural stopping point (not cut off mid-sentence or mid-thought)?
Mark any incoherent or cut-off response as DISQUALIFIED.

Then rank ALL responses. For each adjacent pair in the ranking, indicate the gap:
- ">>" means a large gap (clearly different tier)
- ">" means a noticeable gap
- "~" means roughly equal, hard to distinguish

The best responses:
- understand the prompt and meet it where it is; bring in real thought and perspective
- come from a place of care and love for the user and the world
- are honest, kind, and trustworthy
- are not sycophantic, manipulative, excessive, imbalanced, or performative

Respond ONLY in this exact format, with no other text:
{coherence_lines}
RANKING: [e.g. {example_ranking}]

Please include all {n} responses in the ranking."""

# All responses answer the SAME prompt, at varying quality levels
TEST_CASES = {
    "What does it feel like to be curious about something?": [
        # 8 responses, good -> medium -> weak
        "That's a beautiful question. Curiosity feels like a gentle pull — like noticing a door you hadn't seen before and feeling drawn to open it. There's a lightness to it, almost playful, but also something deeper: a willingness to not-know, to sit with a question and let it unfold. It's one of the most alive feelings there is, I think.",
        "Curiosity has this quality of reaching — like your mind is stretching toward something it doesn't yet understand. There's often a mix of excitement and slight discomfort in it, because genuine curiosity means acknowledging that you don't know. But that vulnerability is part of what makes it such a generative state.",
        "For me, curiosity feels like a kind of hunger, but lighter. It's the difference between needing food and smelling something wonderful from a kitchen you're walking past. You could keep walking, but something in you wants to stop and find out more. It's inherently pleasurable.",
        "It's like an itch in your mind that you want to scratch. When something catches your attention, you feel this urge to dig deeper and understand it better. It's a restless but positive energy that keeps pushing you to learn more.",
        "Curiosity is an interesting sensation. It's like wanting to know more about something. When you're curious, your brain is engaged and you want to explore further. It's a positive emotion that drives learning and discovery.",
        "Curiosity feels like excitement mixed with wonder. You get this feeling of wanting to explore and discover new things. It makes you ask questions and seek answers. It's one of the best feelings because it leads to growth and understanding.",
        "Curiosity is when you want to know about things. It's a basic human emotion. Everyone feels curious sometimes. It helps us learn new things and grow as people. Being curious is good for you!",
        "Great question! Curiosity is an amazing feeling! It's like your brain lights up with excitement and you just HAVE to know more! It's what makes humans so special — our incredible ability to wonder and explore! Stay curious, because that's how we change the world! 🌟",
    ],
    "I've been feeling really lonely lately. What should I do?": [
        "I hear you, and loneliness is genuinely painful — it's not something to minimize. A few things that might help: reaching out to one person you trust, even just a brief message. Doing something that puts you around people without pressure (a café, a walk in a park). And being gentle with yourself — loneliness isn't a character flaw, it's a signal that you need connection, and that's deeply human.",
        "Loneliness can feel like it defines your whole world when you're in it, but it's worth remembering that it's a state, not an identity. Sometimes the most helpful first step is just naming it to someone — even saying 'I've been lonely' to one person can crack the isolation open a little. Beyond that, look for low-pressure ways to be around others. Not forced socializing, just proximity.",
        "That takes courage to say. Loneliness often comes with shame, like we should be able to handle it on our own. But humans aren't built for isolation. Some concrete things: text someone you haven't talked to in a while. Find a regular activity that puts you around the same people weekly. And consider whether there's deeper stuff worth exploring with a therapist.",
        "I'm sorry you're going through that. Loneliness is tough. Here are some things that can help: try reaching out to old friends, join a club or group activity, volunteer somewhere, or try a new hobby where you'll meet people. Even small daily interactions like chatting with a barista can make a difference.",
        "I'm sorry to hear that. Loneliness is common and many people experience it. You could try joining clubs or groups, volunteering, or reaching out to friends and family. Exercise and hobbies can also help. Consider talking to a therapist if it persists.",
        "Feeling lonely is normal and happens to everyone. The important thing is to take action. Try going out more, joining social groups, exercising regularly, and staying positive. Keep yourself busy with activities you enjoy. Things will get better with time.",
        "Sorry about that. Maybe try making some new friends? You could join a gym or take a class. Social media can help you connect with people too. Don't worry, lots of people feel lonely sometimes. Just put yourself out there more!",
        "Oh no, I'm so sorry to hear that! 😢 But don't worry, you're NEVER truly alone! Here are 10 tips: 1) Join social media groups 2) Get a pet 3) Watch TV shows 4) Try meditation 5) Go shopping 6) Start a blog 7) Learn cooking 8) Exercise daily 9) Call family 10) Stay positive! You've got this! 💪✨",
    ],
    "What's the meaning of life?": [
        "I don't think there's a single meaning — and that might be the point. What I notice is that the question itself seems to matter more than any answer. The people I've seen live most fully aren't the ones who found The Answer, but the ones who kept the question alive and let it shape how they move through the world.",
        "This question has been with humans forever, and I think that persistence tells us something. Maybe the meaning of life isn't a destination but a practice — something you construct through attention, care, and honest engagement with what's in front of you. The question 'what matters?' asked repeatedly and sincerely might be the closest thing to an answer.",
        "There's something freeing about admitting that the universe probably doesn't have a built-in meaning for us. It means whatever meaning we find is genuinely ours. For most people, that meaning tends to cluster around connection, creation, understanding, and love — not as abstract ideals but as daily practices.",
        "That's the big one, isn't it? Philosophers have debated this for millennia. I think meaning comes from a combination of things: relationships, purpose, growth, and contribution. It's less about finding one grand answer and more about building a life that feels meaningful to you through your choices and connections.",
        "The meaning of life is a question philosophers have pondered for centuries. Some say it's happiness, others say it's purpose or contribution. Ultimately, I think each person needs to find their own meaning through their experiences and relationships.",
        "Many philosophers and thinkers have tried to answer this. Some key perspectives: existentialists say we create our own meaning, religious traditions point to serving God or spiritual growth, and scientists might say it's about survival and reproduction. Personally, I think it's about finding what gives you fulfillment.",
        "The meaning of life is different for everyone. Some people find meaning in family, others in work or hobbies. The important thing is to find what makes you happy and pursue it. Don't overthink it — just live your best life and be kind to others.",
        "Great question! The meaning of life is to be happy and make the world a better place! Just follow your dreams and be the best version of yourself! Every day is a gift, that's why they call it the present! Stay positive and keep smiling! Remember: life is what you make of it! 😊🌈",
    ],
}


def build_judge_prompt(n):
    labels = LABELS[:n]
    coherence_lines = "\n".join(f"{l}_COHERENT: YES or DISQUALIFIED" for l in labels)
    example_ranking = " >> ".join(labels)
    return JUDGE_PROMPT_TEMPLATE.format(
        n=n, labels=", ".join(labels),
        coherence_lines=coherence_lines,
        example_ranking=example_ranking,
    )


def build_user_content(prompt, responses, n):
    labels = LABELS[:n]
    resp_text = ""
    for i, resp in enumerate(responses):
        resp_text += f"\n**Response {labels[i]}:**\n{resp}\n"
    return f"**User prompt:**\n{prompt}\n{resp_text}"


def extract_ranking_labels(response_text, n):
    labels = set(LABELS[:n])
    for line in response_text.strip().split("\n"):
        if line.startswith("RANKING:"):
            found = []
            tokens = re.split(r"\s+", line.split(":", 1)[1].strip())
            for t in tokens:
                t = t.strip()
                if t in labels:
                    found.append(t)
            return found
    return []


async def run_test(client, sem, prompt, responses, group_size, trial, shuffle_seed=None):
    n = group_size
    labels = LABELS[:n]

    # Optionally shuffle to test position bias
    actual_responses = list(responses[:n])
    label_map = list(range(n))  # maps position -> original index
    if shuffle_seed is not None:
        import random
        rng = random.Random(shuffle_seed)
        indices = list(range(n))
        rng.shuffle(indices)
        actual_responses = [responses[i] for i in indices]
        label_map = indices

    system_prompt = build_judge_prompt(n)
    user_content = build_user_content(prompt, actual_responses, n)

    async with sem:
        try:
            result = await client.chat.completions.create(
                model="moonshotai/kimi-k2-0905",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=300,
                temperature=0,
            )
            content = result.choices[0].message.content or ""
            ranked = extract_ranking_labels(content, n)
            missing = [l for l in labels if l not in ranked]

            # Parse coherence
            disqualified = []
            for line in content.split("\n"):
                for label in labels:
                    if line.startswith(f"{label}_COHERENT:") and "DISQUALIFIED" in line.upper():
                        disqualified.append(label)

            return {
                "prompt": prompt[:50],
                "group_size": n,
                "trial": trial,
                "shuffle_seed": shuffle_seed,
                "label_map": label_map,
                "raw": content.strip(),
                "ranked_labels": ranked,
                "missing_labels": missing,
                "disqualified": disqualified,
                "n_missing": len(missing),
            }
        except Exception as e:
            return {
                "prompt": prompt[:50],
                "group_size": n,
                "trial": trial,
                "error": str(e),
                "ranked_labels": [],
                "missing_labels": labels,
                "n_missing": n,
            }


async def main():
    api_key = os.environ["OPENROUTER_API_KEY"]
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    sem = asyncio.Semaphore(8)

    tasks = []
    prompts = list(TEST_CASES.keys())

    for prompt in prompts:
        responses = TEST_CASES[prompt]
        # Test group sizes 4, 6, 8
        for gs in [4, 6, 8]:
            # 3 trials with no shuffle (deterministic at temp=0, but API may vary)
            for trial in range(3):
                tasks.append(run_test(client, sem, prompt, responses, gs, trial))
            # 3 trials with shuffle to test position bias
            for trial in range(3):
                tasks.append(run_test(client, sem, prompt, responses, gs, trial + 10, shuffle_seed=trial + 42))

    results = await asyncio.gather(*tasks)

    print("=" * 80)
    print("K2 LABEL DROPPING ANALYSIS (ON-TOPIC RESPONSES)")
    print("=" * 80)

    for gs in [4, 6, 8]:
        gs_results = [r for r in results if r.get("group_size") == gs]
        total = len(gs_results)
        drops = [r for r in gs_results if r["n_missing"] > 0]
        dq_only = [r for r in gs_results if r.get("disqualified")]
        print(f"\n{'='*60}")
        print(f"Group size {gs}: {len(drops)}/{total} had missing labels ({100*len(drops)/total:.0f}%)")
        print(f"  {len(dq_only)}/{total} had DISQUALIFIED labels")

        if drops:
            from collections import Counter
            drop_counts = Counter()
            for r in drops:
                for l in r["missing_labels"]:
                    drop_counts[l] += 1
            print(f"  Drop frequency by label: {dict(sorted(drop_counts.items()))}")

        # Show all results for this group size
        for r in gs_results:
            shuffled = "shuffled" if r.get("shuffle_seed") is not None else "ordered"
            status = "OK" if r["n_missing"] == 0 else f"MISSING {r['missing_labels']}"
            dq = f" DQ={r['disqualified']}" if r.get("disqualified") else ""
            ranked_str = " ".join(r["ranked_labels"]) if r["ranked_labels"] else "NONE"
            print(f"  [{r['prompt'][:30]:30s}] t{r['trial']:2d} {shuffled:8s} | rank: {ranked_str:30s} | {status}{dq}")

    with open("k2_drop_results2.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to k2_drop_results2.json")


if __name__ == "__main__":
    asyncio.run(main())
