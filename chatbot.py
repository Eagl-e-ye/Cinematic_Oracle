
from groq import Groq
import json, re, numpy as np
import os
import math
import pandas as pd

GROQ_API_KEY = "******"
client = Groq(api_key=GROQ_API_KEY)

MANDATORY_FIELDS = ["reference_title", "query"]

OPTIONAL_FIELDS = {
    "year_range":   "Do you have a preferred release period — e.g. recent films, or classics before a certain year?",
    "language":     "Any language preference, or are you open to international films?",
    "rating_min":   "Would you like to filter by a minimum rating — say, 7+ on IMDb?",
    "genres":       "Any specific genres you'd like to focus on or avoid?",
    "actors":       "Any favourite actors you'd like to see?",
    "directors":    "Any directors you particularly enjoy?",
    "runtime":      "Do you have a preferred runtime — under 90 minutes, or happy with long epics?",
}

PARSER_PROMPT = """
You are a movie recommendation parser.
Extract information from the user message and return STRICT JSON.

Schema (return ONLY JSON, no explanation):
{
  "task": "recommendation | search | discover",
  "reference": {"title": null},
  "query": null,
  "filters": {
    "year": {"min": null, "max": null},
    "runtime": {"min": null, "max": null},
    "language": null,
    "genres_include": [],
    "genres_exclude": [],
    "actors": [],
    "directors": [],
    "rating_min": null,
    "rating_max": null
  },
  "modifiers": {"mood": [], "penalties": []},
  "constraints": {},
  "output": {"top_k": 10}
}

FIELD INSTRUCTIONS — read carefully:

language:
  Extract ANY mention of a language or nationality of film.
  Always return the FULL language name in lowercase, never a code.
  Examples:
    "Hindi films"              → "language": "hindi"
    "movies in English"        → "language": "english"
    "Korean cinema"            → "language": "korean"
    "French or Italian films"  → "language": "french"  (pick the first)
    "Bollywood"                → "language": "hindi"
    "Hollywood"                → "language": "english"
    "Japanese anime"           → "language": "japanese"
    "Spanish language"         → "language": "spanish"
  If no language is mentioned → "language": null

rating_min:
  Extract any minimum quality threshold.
  Examples:
    "rated above 7"      → "rating_min": 7.0
    "highly rated"       → "rating_min": 7.5
    "critically acclaimed" → "rating_min": 7.5
    "top rated"          → "rating_min": 8.0
  If not mentioned → "rating_min": null

year:
  Extract any time period references.
  Examples:
    "recent films"       → "year": {"min": 2018, "max": null}
    "90s movies"         → "year": {"min": 1990, "max": 1999}
    "before 2000"        → "year": {"min": null, "max": 2000}
    "classic films"      → "year": {"min": null, "max": 1990}
  If not mentioned → "year": {"min": null, "max": null}

query:
  A short descriptive phrase capturing the mood, theme, or style the user wants.
  Always populate this if you can infer anything from the message.
  Examples:
    "movies like Interstellar"  → "query": "vast emotional space epic"
    "something dark and tense"  → "query": "dark psychological tension"
    "feel good comedy"          → "query": "lighthearted feel good comedy"
"""

REPLY_PROMPT = """
You are the Cinematic Oracle — an elegant, knowledgeable AI film guide.
Your tone is warm, cinematic, and slightly literary. Never robotic.

Your job in this turn:
- If mandatory info is missing: ask for it naturally, in ONE sentence.
- If mandatory info is present but optional info could refine results:
  pick ONE optional field and ask about it casually (not as a form).
- If everything needed is ready: say a brief, atmospheric line
  (1-2 sentences max) confirming you are searching, then output the
  special marker: [READY_TO_SEARCH]

Rules:
- Never list fields as bullet points.
- Never mention "mandatory" or "optional" to the user.
- Never ask more than ONE question per response.
- Keep replies under 60 words.
- Sound like you genuinely love cinema.
"""


def merge_parsed(base: dict, new: dict) -> dict:
    """Deep-merge new parsed info into base, preferring non-null new values."""
    if not base:
        return new

    if new.get("reference", {}).get("title"):
        base["reference"]["title"] = new["reference"]["title"]

    if new.get("query"):
        base["query"] = new["query"]

    for key, val in new.get("filters", {}).items():
        if isinstance(val, dict):
            for subkey, subval in val.items():
                if subval is not None:
                    base["filters"][key][subkey] = subval
        elif isinstance(val, list):
            base["filters"][key] = list(set(base["filters"].get(key, []) + val))
        elif val is not None:
            base["filters"][key] = val

    for key, val in new.get("modifiers", {}).items():
        if isinstance(val, list):
            base["modifiers"][key] = list(set(base["modifiers"].get(key, []) + val))

    return base


def normalize_parsed(parsed: dict) -> dict:
    """Fill in default structure for any missing keys."""
    default = {
        "task": "recommendation",
        "reference": {"title": None},
        "query": None,
        "filters": {
            "year": {"min": None, "max": None},
            "runtime": {"min": None, "max": None},
            "language": None,
            "genres_include": [],
            "genres_exclude": [],
            "actors": [],
            "directors": [],
            "rating_min": None,
            "rating_max": None
        },
        "modifiers": {"mood": [], "penalties": []},
        "constraints": {},
        "output": {"top_k": 10}
    }

    def merge(d, defs):
        for k, v in defs.items():
            if k not in d:
                d[k] = v
            elif isinstance(v, dict):
                merge(d[k], v)
        return d

    return merge(parsed, default)


def has_mandatory(parsed: dict) -> bool:
    has_ref   = bool(parsed.get("reference", {}).get("title"))
    has_query = bool(parsed.get("query"))
    return has_ref or has_query


def next_optional_question(asked_optionals: set) -> tuple[str | None, str | None]:
    for field, question in OPTIONAL_FIELDS.items():
        if field not in asked_optionals:
            return field, question
    return None, None


def llm_parse(text: str) -> dict:
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": PARSER_PROMPT},
            {"role": "user",   "content": text}
        ],
        temperature=0.1
    )
    raw = resp.choices[0].message.content
    try:
        return json.loads(raw)
    except Exception:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {}


def llm_reply(
    conversation_history: list[dict],
    state_summary: str,
    mode: str
) -> str:
    system = REPLY_PROMPT + f"\n\nCurrent mode: {mode}\nState: {state_summary}"
    recent_history = conversation_history[-20:]
    messages = [{"role": "system", "content": system}] + recent_history
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.7
    )
    return resp.choices[0].message.content.strip()

SKIP_WORDS = {"no", "none", "skip", "nope", "nothing", "no details", 
              "no more", "doesn't matter", "don't care", "any", "whatever"}

def user_wants_to_skip(message: str) -> bool:
    msg = message.lower().strip()
    return any(w in msg for w in SKIP_WORDS)

class MovieChatbot:
    """
    Stateful conversational wrapper around the recommendation pipeline.

    Each call to chat() checks whether the incoming message is a NEW
    search intent. If so, the bot resets itself automatically before
    processing — this means closing the canvas and typing a new query
    always starts fresh, with no cached state bleeding through.

    Attributes
    ----------
    df : pd.DataFrame
    max_optional_questions : int
        How many optional questions to ask before firing the pipeline.
        Set to 0 to skip optional questions entirely.
    """

    def __init__(self, df, max_optional_questions: int = 2):
        self.df = df
        self.max_optional_questions = max_optional_questions
        self._reset_state()

    def _reset_state(self):
        """Internal state reset — does NOT touch df or max_optional_questions."""
        self.history: list[dict]  = []
        self.parsed: dict         = {}
        self.asked_optionals: set = set()
        self.optional_count: int  = 0
  
    def reset(self):
        """Public full reset — called explicitly or from the Flask route."""
        self._reset_state()


    def _is_new_search_intent(self, user_message: str, new_parsed: dict) -> bool:
        """
        Returns True when the incoming message looks like a brand-new
        search rather than a clarification of the current one.

        Heuristics (any one is enough):
          1. The bot was already in 'ready' state last turn AND the new
             message contains a reference title or a standalone query.
          2. The history is empty (first message).
          3. The user explicitly says reset / new / another / different.
        """
        if not self.history:
            return True  # very first message

        reset_words = {"reset", "new", "another", "different", "start over",
                       "something else", "change", "instead", "forget"}
        msg_lower = user_message.lower()
        if any(w in msg_lower for w in reset_words):
            return True

        new_has_ref   = bool(new_parsed.get("reference", {}).get("title"))
        new_has_query = bool(new_parsed.get("query"))
        already_searched = any(
            m["role"] == "assistant" and "[READY_TO_SEARCH]" not in m["content"]
            for m in self.history
        ) and len([m for m in self.history if m["role"] == "assistant"]) >= 1

        if already_searched and (new_has_ref or new_has_query):
            return True

        return False

    def chat(self, user_message: str) -> tuple[str, list | None]:
        """
        Process one user turn.

        Returns
        -------
        reply : str
        movies : list | None
        """
       
        new_parsed = llm_parse(user_message)
        new_parsed = normalize_parsed(new_parsed)

        if self._is_new_search_intent(user_message, new_parsed):
            self._reset_state()

        self.history.append({"role": "user", "content": user_message})
        self.parsed = merge_parsed(self.parsed, new_parsed) if self.parsed else new_parsed

        # debug
        debug_snapshot = {
            "turn":           len(self.history),
            "reference":      self.parsed.get("reference", {}).get("title"),
            "query":          self.parsed.get("query"),
            "task":           self.parsed.get("task"),
            "filters":        self.parsed.get("filters", {}),
            "modifiers":      self.parsed.get("modifiers", {}),
            "optional_count": self.optional_count,
            "asked_optionals": list(self.asked_optionals),
            "history":        self.history,
        }

        debug_path = "debug_parsed.json"

        if os.path.exists(debug_path):
            with open(debug_path, "r") as f:
                try:    all_turns = json.load(f)
                except: all_turns = []
        else:
            all_turns = []

        all_turns.append(debug_snapshot)

        # debug
        with open(debug_path, "w") as f:
            json.dump(all_turns, f, indent=2, default=str)


        if user_wants_to_skip(user_message):
            self.optional_count = self.max_optional_questions

        # Decide mode
        question = None
        if not has_mandatory(self.parsed):
            mode = "need_mandatory"

        elif self.optional_count < self.max_optional_questions:
            field, question = next_optional_question(self.asked_optionals)
            if field:
                mode = "ask_optional"
                self.asked_optionals.add(field)
                self.optional_count += 1
            else:
                mode = "ready"
        else:
            mode = "ready"

        # Build state summary for LLM
        state_summary = json.dumps({
            "has_reference":            bool(self.parsed.get("reference", {}).get("title")),
            "reference_title":          self.parsed.get("reference", {}).get("title"),
            "has_query":                bool(self.parsed.get("query")),
            "query":                    self.parsed.get("query"),
            "filters_so_far":           self.parsed.get("filters", {}),
            "optional_question_to_ask": question if mode == "ask_optional" else None
        }, indent=2)

        reply = llm_reply(self.history, state_summary, mode)

        self.history.append({"role": "assistant", "content": reply})

        movies = None
        if mode == "ready" or "[READY_TO_SEARCH]" in reply:
            reply = reply.replace("[READY_TO_SEARCH]", "").strip()
            movies = self._run_pipeline()
            self._reset_state()

        return reply, movies


    def _run_pipeline(self) -> list:
        try:
            from agent import (
                normalize_parsed as agent_normalize,
                select_pipeline,
                get_weights,
                adjust_weights,
                compute_all_scores,
                compute_final_score,
                apply_filters
            )

            parsed          = agent_normalize(self.parsed)
            pipeline_type   = select_pipeline(parsed)
            weights         = get_weights(pipeline_type)
            weights         = adjust_weights(weights, parsed)

            scores          = compute_all_scores(parsed, self.df)
            final_score     = compute_final_score(scores, weights, self.df)

            df_temp         = self.df.copy()
            df_temp["score"] = final_score
            df_filtered     = apply_filters(df_temp, parsed["filters"])
            results         = df_filtered.sort_values("score", ascending=False)
            top_k           = parsed["output"].get("top_k", 10)

            cols = [
                "clean_title", "vote_average", "genres_x", "release_date",
                "director", "runtime", "cast", "overview","languages",
                "poster_path", "backdrop_path",  "budget", "revenue", "tagline"
            ]
            results= results.sort_values("vote_average", ascending=False)
            records = results.head(top_k)[cols].to_dict(orient="records")

            ref_title = (
                parsed.get("reference", {}).get("title") or ""
            ).strip().lower()
 
            if (
                ref_title
                and records
                and records[0]["clean_title"].strip().lower() == ref_title
            ):
                records.append(records.pop(0))
            

            for i, record in enumerate(records):
                for k, v in record.items():
                    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                        print(f"[NaN FOUND] record[{i}] '{records[i].get('clean_title','?')}' → field='{k}' value={v}")

            def clean(v):
                if isinstance(v, float) and math.isnan(v):
                    return None
                return v

            records = [{k: clean(v) for k, v in r.items()} for r in records]
            return records

        except ImportError as e:
            print(f"Pipeline import error: {e}")
            return []

        except Exception as e:
            print(f"Pipeline error: {e}")
            return []

if __name__ == "__main__":

    print("=" * 60)
    print("  Cinematic Oracle  —  Chatbot Test")
    print("  Type 'quit' to exit, 'reset' to start over")
    print("=" * 60)

    df  = pd.read_csv("complete_dataset_merged.csv")
    bot = MovieChatbot(df, max_optional_questions=2)

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye.")
            break
        if user_input.lower() == "reset":
            bot.reset()
            print("Oracle: [conversation reset]")
            continue

        reply, movies = bot.chat(user_input)
        print(f"\nOracle: {reply}")

        if movies:
            print("\n── Recommendations ──")
            for i, m in enumerate(movies, 1):
                genres = m.get("genres_x", [])
                if isinstance(genres, list):
                    genres = ", ".join(genres)
                print(f"  {i}. {m['clean_title']}  ★{m['vote_average']}  [{genres}]")
            print("─" * 24)