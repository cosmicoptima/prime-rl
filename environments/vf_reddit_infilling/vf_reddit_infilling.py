from datasets import load_dataset
import json
import re
import verifiers as vf
from transformers import AutoTokenizer


# Global tokenizer instance to avoid reloading
_tokenizer = None


def get_tokenizer():
    """Get or create the tokenizer instance"""
    global _tokenizer
    if _tokenizer is None:
        # Use the same model as specified in the config files
        model_name = "Qwen/Qwen2.5-3B-Instruct"
        _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return _tokenizer


get_tokenizer()


def count_tokens(text):
    """Count the number of tokens in a text string"""
    return len(_tokenizer.encode(str(text)))


def render_all_examples(thread: dict) -> list[str]:
    examples = []
    comments = thread["comments"]
    
    for i, comment in enumerate(comments):
        modified_thread = {
            **thread,
            "redacted": i
        }
        
        modified_thread["question"] = render_question(modified_thread)
        modified_thread["answer"] = comment["comment"]
        examples.append(modified_thread)
    
    return examples


def render_question(thread: dict) -> str:
    title = thread["title"]
    body = thread["body"]
    
    comments_html = []
    for i, c in enumerate(thread["comments"]):
        if thread['redacted'] == i:
            comment_text = '[REDACTED: COMPLETE THIS]'
        else:
            comment_text = c['comment'].replace("\n", "\n        ")
        
        comments_html.append(
            f"    <comment>\n"
            f"      <n>{i+1}</n>\n"
            f"      <user>{c['username']}</user>\n"
            f"      <text>\n"
            f"        {comment_text}\n"
            f"      </text>\n"
            f"    </comment>"
        )
    
    comments = "\n".join(comments_html)
    
    return f"""<context>
  <submission>
    <title>{title}</title>
    <body>{body}</body>
  </submission>
  <thread>
{comments}
  </thread>
</context>

What you see inside the above <context> block is a Reddit post and a single comment thread under that post. One comment in the thread has been redacted (its full text is "[REDACTED: COMPLETE THIS]"). Your task is to guess what the comment says. Think as much as you need to in a <reasoning> block, then output your answer in an <answer> block containing only the text of the comment."""


def load_environment(**kwargs) -> vf.Environment:
    dataset = load_dataset("cosmicoptima/IFhXR5QAHNW9", split="train")
    dataset = dataset.filter(lambda x: all(k in x for k in ["title", "body", "comments"]))
    dataset = dataset.map(render_all_examples, batched=True, batch_size=1)
    
    MAX_PROMPT_TOKENS = 2076 # 3264 ... - 128 because 3264 still errored at one point
    dataset = dataset.filter(lambda x: count_tokens(x["question"]) <= MAX_PROMPT_TOKENS)
    
    parser = vf.XMLParser(fields=["reasoning", "answer"], answer_field="answer")

    def reward(completion, answer, **kwargs):
        text = " ".join(m.get("content", "") for m in completion if m.get("role") == "assistant")
        predicted_answer = parser.parse_answer(text)
        
        if not predicted_answer:
            return 0.0
        
        # Convert to strings and normalize
        actual_text = str(answer).strip()
        predicted_text = str(predicted_answer).strip()
        
        # Handle empty cases
        if not actual_text and not predicted_text:
            return 1.0
        if not actual_text or not predicted_text:
            return 0.0
        
        # Calculate edit distance (your existing implementation)
        def edit_distance(s1, s2):
            m, n = len(s1), len(s2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            # Initialize base cases
            for i in range(m + 1):
                dp[i][0] = i
            for j in range(n + 1):
                dp[0][j] = j
            
            # Fill the dp table
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if s1[i - 1] == s2[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1]
                    else:
                        dp[i][j] = 1 + min(dp[i - 1][j],    # deletion
                                          dp[i][j - 1],     # insertion
                                          dp[i - 1][j - 1]) # substitution
            
            return dp[m][n]
        
        distance = edit_distance(actual_text, predicted_text)
        max_len = max(len(actual_text), len(predicted_text))
        
        # Convert to similarity score (1 - normalized edit distance)
        if max_len == 0:
            return 1.0
        
        similarity = 1.0 - (distance / max_len)
        
        return similarity
    
    rubric = vf.Rubric(
        funcs=[reward],
        weights=[1.0],
    )

    return vf.SingleTurnEnv(
        dataset=dataset,
        parser=parser,
        rubric=rubric,
        **kwargs,
    )