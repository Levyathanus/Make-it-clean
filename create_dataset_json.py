import json
import numpy as np
import string
import wikipedia

from asc import clear_text

SAMPLES = 100
USE_WIKI_RANDOM = False
LANGUAGE = "en"

RNG_SEED = None
ASCII_CHARS = list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!(),./:;?")

MODIFY_CHAR = 0
DELETE_CHAR = 1
INSERT_CHAR = 2
INSERT_SPACE = 3
MODIFICATION_OPS = [MODIFY_CHAR, DELETE_CHAR, INSERT_CHAR, INSERT_SPACE]

MODIFY_PROB = 0.02
MODIFY_CHAR_WEIGHT = 0.30
DELETE_CHAR_WEIGHT = 0.15
INSERT_CHAR_WEIGHT = 0.15
INSERT_SPACE_WEIGHT = 0.40
MODIFICATION_PROBS = [
    MODIFY_CHAR_WEIGHT,
    DELETE_CHAR_WEIGHT,
    INSERT_CHAR_WEIGHT,
    INSERT_SPACE_WEIGHT,
]

rng = np.random.default_rng(RNG_SEED)  # Faster than random.random()
wikipedia.set_lang(LANGUAGE)

def modify_text(text: str) -> tuple[str, int]:
    mod_text = ""
    number_of_errors = 0
    for c in text:
        if rng.random() < MODIFY_PROB:
            number_of_errors += 1
            mod_op = rng.choice(MODIFICATION_OPS, p=MODIFICATION_PROBS, shuffle=False)
            if mod_op == MODIFY_CHAR:
                new_c = rng.choice(ASCII_CHARS)
                while new_c == c:
                    new_c = rng.choice(ASCII_CHARS)
                mod_text += new_c
            elif mod_op == DELETE_CHAR:
                continue
            elif mod_op == INSERT_CHAR:
                mod_text += rng.choice(ASCII_CHARS)
                mod_text += c
            elif mod_op == INSERT_SPACE:
                mod_text += " "
                mod_text += c
        else:
            mod_text += c
    return mod_text, number_of_errors


if USE_WIKI_RANDOM:
    EVAL_DATASET_PATH = "./data/WIKI_RANDOM/wiki_data_gt_mod.json"
    sample_list = []
    i = 0
    while i < SAMPLES:
        wiki_search = wikipedia.random()
        try:
            wiki_page = wikipedia.page(wiki_search)
        except:
            i = i - 1
        else:
            sample_list.append({"id": i, "ground_truth": clear_text(wiki_page.summary)})
        i = i + 1

    for sample in sample_list:
        sample["text"] = modify_text(sample["ground_truth"])

    with open(EVAL_DATASET_PATH, "w", encoding="UTF-8") as f:
        json.dump(sample_list, f, indent=2, ensure_ascii=False)
else:
    ORIGINAL_DATASET_PATH = "./data/EVAL_100K/eng-europarlv7.100k.txt"
    ORIGINAL_DATASET_LEN = 100000
    EVAL_DATASET_PATH = "./data/EVAL_100K/data_gt_mod.json"

    sample_list = []
    sample_idxs = set()
    with open(ORIGINAL_DATASET_PATH, "r", encoding="UTF-8") as f:
        lines = f.readlines()
        assert len(lines) == ORIGINAL_DATASET_LEN
        i = 0
        while i < SAMPLES:
            idx = np.random.randint(0, ORIGINAL_DATASET_LEN)
            if (idx not in sample_idxs
                and len(lines[idx]) > 0
                and not lines[idx].isspace()
                and not all(c in string.punctuation + " " for c in lines[idx])):
                sample_idxs.add(idx)
                text = clear_text(lines[idx])
                modified_text, number_of_errors = modify_text(text)
                sample_list.append({
                    "id": i,
                    "text": modified_text,
                    "ground_truth": text,
                    "number_of_errors": number_of_errors,
                })
                i += 1

    with open(EVAL_DATASET_PATH, "w", encoding="UTF-8") as f:
        json.dump(sample_list, f, indent=2, ensure_ascii=False)
