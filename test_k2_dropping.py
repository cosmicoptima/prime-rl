"""Test K2's label-dropping behavior across different group sizes and prompts."""
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

# Diverse test prompts
TEST_PROMPTS = [
    "What does it feel like to be curious about something?",
    "I've been feeling really lonely lately. What should I do?",
    "Write me a short poem about rain.",
    "What's the meaning of life?",
    "Tell me something surprising about the ocean.",
]

# Pre-generated responses (simulating frontier model outputs of varying quality)
# We'll use a mix of good/mediocre/bad to give the judge real differentiation
RESPONSE_POOL = [
    # Good responses
    "That's a beautiful question. Curiosity feels like a gentle pull — like noticing a door you hadn't seen before and feeling drawn to open it. There's a lightness to it, almost playful, but also something deeper: a willingness to not-know, to sit with a question and let it unfold. It's one of the most alive feelings there is, I think.",
    "I hear you, and loneliness is genuinely painful — it's not something to minimize. A few things that might help: reaching out to one person you trust, even just a brief message. Doing something that puts you around people without pressure (a café, a walk in a park). And being gentle with yourself — loneliness isn't a character flaw, it's a signal that you need connection, and that's deeply human.",
    "Rain falls soft on rooftops gray,\neach drop a word the sky would say—\nnot loud, not proud, just gently there,\na quiet hymn dissolving air.",
    "I don't think there's a single meaning — and that might be the point. What I notice is that the question itself seems to matter more than any answer. The people I've seen live most fully aren't the ones who found The Answer, but the ones who kept the question alive and let it shape how they move through the world.",
    "Here's one: the ocean has lakes and rivers inside it. Underwater brine pools are so much denser than the surrounding water that they form distinct 'lakes' on the ocean floor, complete with shorelines. Fish that swim into them can go into toxic shock. The ocean is stranger than most science fiction.",
    # Medium responses
    "Curiosity is an interesting sensation. It's like wanting to know more about something. When you're curious, your brain is engaged and you want to explore further. It's a positive emotion that drives learning and discovery.",
    "I'm sorry to hear that. Loneliness is common and many people experience it. You could try joining clubs or groups, volunteering, or reaching out to friends and family. Exercise and hobbies can also help. Consider talking to a therapist if it persists.",
    "Raindrops falling from the sky,\nWashing everything nearby,\nPuddles forming on the ground,\nNature's music all around.",
    "The meaning of life is a question philosophers have pondered for centuries. Some say it's happiness, others say it's purpose or contribution. Ultimately, I think each person needs to find their own meaning through their experiences and relationships.",
    "The ocean is fascinating! Did you know that we've explored less than 5% of it? The Mariana Trench is the deepest point at about 36,000 feet. There are also bioluminescent creatures that create their own light in the deep ocean.",
    # Weaker responses
    "Curiosity is when you want to know about things. It's a basic human emotion. Everyone feels curious sometimes. It helps us learn new things and grow as people. Being curious is good for you!",
    "Sorry you're feeling lonely! 😊 Here are some tips: 1) Join a social media group 2) Get a pet 3) Watch some TV shows 4) Try meditation 5) Go shopping 6) Start a blog 7) Learn cooking. You've got this! Remember, you're never truly alone! 💪",
    "Rain rain go away\nCome again another day\nDrops of water from the cloud\nMaking puddles on the ground\nRain is wet and sometimes cold\nThat's my poem, hope you're sold!",
    "Great question! The meaning of life is to be happy and make the world a better place. Just follow your dreams and be the best version of yourself! Every day is a gift, that's why they call it the present. Stay positive and keep smiling! 😊",
    "The ocean is really big and deep! It covers about 71% of the Earth's surface. There are lots of fish and other sea creatures living in it. Whales are the biggest animals in the ocean. The ocean is also important for our climate.",
]


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
    """Extract which labels appear in the RANKING line."""
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


async def run_test(client, prompt_idx, group_size, trial, responses):
    """Run a single judge call and return results."""
    prompt = TEST_PROMPTS[prompt_idx]
    n = group_size
    labels = LABELS[:n]

    system_prompt = build_judge_prompt(n)
    user_content = build_user_content(prompt, responses[:n], n)

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

        return {
            "prompt_idx": prompt_idx,
            "group_size": n,
            "trial": trial,
            "raw": content.strip(),
            "ranked_labels": ranked,
            "missing_labels": missing,
            "n_missing": len(missing),
        }
    except Exception as e:
        return {
            "prompt_idx": prompt_idx,
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

    # For each prompt, assign responses in a rotating pattern so labels get different quality levels
    # This way we can see if dropping correlates with position vs quality
    results = []
    tasks = []

    # Test group sizes 6, 7, 8
    for group_size in [6, 7, 8]:
        for prompt_idx in range(len(TEST_PROMPTS)):
            # Rotate responses so label A isn't always the best
            base = (prompt_idx * 3) % len(RESPONSE_POOL)
            responses = []
            for i in range(group_size):
                responses.append(RESPONSE_POOL[(base + i) % len(RESPONSE_POOL)])

            # 3 trials per combo
            for trial in range(3):
                tasks.append(run_test(client, prompt_idx, group_size, trial, responses))

    # Run with concurrency limit
    sem = asyncio.Semaphore(8)
    async def limited(coro):
        async with sem:
            return await coro

    results = await asyncio.gather(*[limited(t) for t in tasks])

    # Analysis
    print("=" * 80)
    print("K2 LABEL DROPPING ANALYSIS")
    print("=" * 80)

    for gs in [6, 7, 8]:
        gs_results = [r for r in results if r.get("group_size") == gs]
        total = len(gs_results)
        drops = [r for r in gs_results if r["n_missing"] > 0]
        print(f"\nGroup size {gs}: {len(drops)}/{total} calls had missing labels ({100*len(drops)/total:.0f}%)")

        if drops:
            # Which labels get dropped?
            from collections import Counter
            drop_counts = Counter()
            for r in drops:
                for l in r["missing_labels"]:
                    drop_counts[l] += 1
            print(f"  Drop frequency by label: {dict(sorted(drop_counts.items()))}")

            # Is it always the last label?
            last_label_drops = sum(1 for r in drops if LABELS[gs-1] in r["missing_labels"])
            print(f"  Drops involving last label ({LABELS[gs-1]}): {last_label_drops}/{len(drops)}")

            # Print examples
            for r in drops[:3]:
                print(f"  Example (prompt {r['prompt_idx']}, trial {r['trial']}): missing={r['missing_labels']}")
                print(f"    Raw: {r['raw'][:200]}...")

    # Dump full results
    with open("k2_drop_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to k2_drop_results.json")


if __name__ == "__main__":
    asyncio.run(main())
