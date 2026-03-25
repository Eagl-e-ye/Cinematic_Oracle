import ast
import pandas as pd
import similarities as sml
import numpy as np

LANG_MAP = {
    "english":            "en",
    "hindi":              "hi",
    "french":             "fr",
    "spanish":            "es",
    "german":             "de",
    "italian":            "it",
    "portuguese":         "pt",
    "russian":            "ru",
    "japanese":           "ja",
    "korean":             "ko",
    "chinese":            "zh",
    "mandarin":           "zh",
    "cantonese":          "zh",
    "arabic":             "ar",
    "turkish":            "tr",
    "swedish":            "sv",
    "danish":             "da",
    "norwegian":          "no",
    "finnish":            "fi",
    "dutch":              "nl",
    "polish":             "pl",
    "czech":              "cs",
    "hungarian":          "hu",
    "romanian":           "ro",
    "greek":              "el",
    "hebrew":             "he",
    "thai":               "th",
    "indonesian":         "id",
    "malay":              "ms",
    "tamil":              "ta",
    "telugu":             "te",
    "malayalam":          "ml",
    "kannada":            "kn",
    "bengali":            "bn",
    "marathi":            "mr",
    "punjabi":            "pa",
    "urdu":               "ur",
    "persian":            "fa",
    "farsi":              "fa",
    "vietnamese":         "vi",
    "ukrainian":          "uk",
    "catalan":            "ca",
    "croatian":           "hr",
    "serbian":            "sr",
    "slovak":             "sk",
    "bulgarian":          "bg",
    "lithuanian":         "lt",
    "latvian":            "lv",
    "estonian":           "et",
    "slovenian":          "sl",
    "swahili":            "sw",
    "afrikaans":          "af",
    "albanian":           "sq",
    "armenian":           "hy",
    "azerbaijani":        "az",
    "basque":             "eu",
    "belarusian":         "be",
    "bosnian":            "bs",
    "georgian":           "ka",
    "icelandic":          "is",
    "irish":              "ga",
    "macedonian":         "mk",
    "maltese":            "mt",
    "mongolian":          "mn",
    "nepali":             "ne",
    "sinhala":            "si",
    "welsh":              "cy",
}


def normalize_language(value: str) -> str:

    if not isinstance(value, str):
        return value

    cleaned = value.strip().lower()

    if len(cleaned) == 2 and cleaned.isalpha():
        return cleaned

    return LANG_MAP.get(cleaned, cleaned)   

def apply_filters(df, filters):
    
    df_filtered = df.copy()
    
    if filters["year"]["max"]:
        df_filtered = df_filtered[df_filtered["year"] <= filters["year"]["max"]]
    
    if filters["year"]["min"]:
        df_filtered = df_filtered[df_filtered["year"] >= filters["year"]["min"]]
    if filters["runtime"]["max"]:
        df_filtered = df_filtered[df_filtered["runtime"] <= filters["runtime"]["max"]]

    if filters["runtime"]["min"]:
        df_filtered = df_filtered[df_filtered["runtime"] >= filters["runtime"]["min"]]

    if filters["language"]:
        lang = filters["language"].strip().lower()
        lang = normalize_language(lang)
        def lang_match(val):
            if not isinstance(val, str):
                return False
            try:
                parsed = ast.literal_eval(val)
                return lang in [v.strip().lower() for v in parsed]
            except Exception:
                print("language filter uns")
                return val.strip().lower() == lang

        df_filtered = df_filtered[df_filtered["languages"].apply(lang_match)]
    return df_filtered

def select_pipeline(parsed):
    
    if parsed["reference"]["title"]:
        return "reference"
    
    if parsed["query"]:
        return "search"
    
    if parsed["filters"]["actors"]:
        return "actor"

    if parsed["filters"]["directors"]:
        return "director"
        
    return "discover"

def get_weights(pipeline_type):
    
    if pipeline_type == "reference":
        return {
            "text": 0.7,
            "genre": 0.2,
            "cast": 0.05,
            "director": 0.03,
            "rating": 0.02,
            "popularity": 0.0,
            "underrated": 0.0
        }
    
    elif pipeline_type == "search":
        return {
            "text": 0.8,
            "genre": 0.15,
            "cast": 0.0,
            "director": 0.0,
            "rating": 0.05,
            "popularity": 0.0,
            "underrated": 0.0
        }
    
    elif pipeline_type == "actor":
        return {
            "text": 0.3,
            "genre": 0.1,
            "cast": 0.5,
            "director": 0.05,
            "rating": 0.05,
            "popularity": 0.0,
            "underrated": 0.0
        }
    
    elif pipeline_type == "director":
        return {
            "text": 0.3,
            "genre": 0.1,
            "cast": 0.1,
            "director": 0.4,
            "rating": 0.1,
            "popularity": 0.0,
            "underrated": 0.0
        }
    
    else:  # discover
        return {
            "text": 0.6,
            "genre": 0.2,
            "cast": 0.1,
            "director": 0.05,
            "rating": 0.05,
            "popularity": 0.0,
            "underrated": 0.0
        }


def find_reference(title, df):
    idx = df[df["clean_title"].str.lower() == title.lower()].index
    
    if len(idx) == 0:
        return None
    
    return idx[0]


def compute_all_scores(parsed, df):    
    # --- TEXT ---
    if parsed["reference"]["title"]:
        idx = find_reference(parsed["reference"]["title"], df)
        text_sim = sml.compute_text_similarity(idx=idx)
        genre_sim = sml.compute_genre_similarity(idx, df)
        cast_sim = sml.compute_cast_similarity(idx=idx)
        director_sim = sml.compute_director_similarity(idx=idx)
    
    else:
        text_sim = sml.compute_text_similarity(query=parsed["query"])
        genre_sim = np.zeros(len(df))
        cast_sim = np.zeros(len(df))
        director_sim = np.zeros(len(df))
    
    if parsed["filters"]["actors"]:
        actor_query = " ".join(parsed["filters"]["actors"])
        cast_sim = sml.compute_cast_similarity(cast=actor_query)
    
    if parsed["filters"]["directors"]:
        director_query = " ".join(parsed["filters"]["directors"])
        director_sim = sml.compute_director_similarity(director=director_query)
    
    return {
        "text": text_sim,
        "genre": genre_sim,
        "cast": cast_sim,
        "director": director_sim
    }


def compute_final_score(scores, weights, df):
    
    return (
        weights["text"] * scores["text"] +
        weights["genre"] * scores["genre"] +
        weights["cast"] * scores["cast"] +
        weights["director"] * scores["director"] +
        weights["rating"] * df["rating_norm"].values +
        weights["popularity"] * df["popularity_norm"].values +
        weights["underrated"] * df["underrated_score"].values
    )


def normalize_parsed(parsed):
    
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
        "modifiers": {
            "mood": []
        },
        "constraints": {},
        "output": {"top_k": 10}
    }
    
    def merge(d, default_d):
        for k, v in default_d.items():
            if k not in d:
                d[k] = v
            elif isinstance(v, dict):
                merge(d[k], v)
        return d
    
    return merge(parsed, default)


def adjust_weights(weights, parsed):
    
    if parsed["filters"]["genres_include"]:
        weights["genre"] += 0.1
    
    if parsed["modifiers"]["mood"]:
        weights["text"] += 0.1
    
    total = sum(weights.values())
    return {k: v/total for k, v in weights.items()}
