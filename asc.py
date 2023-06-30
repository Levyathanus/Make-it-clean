# asc.py
"""Advanced Spelling Correction module."""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import evaluate
import pkg_resources
import pytesseract
import re
import string
import time
import torch
import warnings

warnings.filterwarnings("ignore")

from enum import Enum
from PIL import Image
from PyPDF2 import PdfReader
from spellchecker import SpellChecker
from symspellpy import SymSpell
from transformers import pipeline
from transformers.utils import logging

logging.set_verbosity(logging.ERROR)

__author__ = "Michele Zenoni"
__credits__ = ["Michele Zenoni"]
__maintainer__ = "Michele Zenoni"

DEFAULT_SPELLING_MODEL = "oliverguhr/spelling-correction-english-base"
DEFAULT_MAX_LEN = 128
DEFAULT_NUM_BEAMS = 4
DEFAULT_TEMPERATURE = 0.5
DEFAULT_MAX_EDIT_DISTANCE = 2


def normalize_spaces(text) -> str:
    """Replaces multiple white spaces with single ones.

    It works directly on strings as weel as on the spelling model output.
    """
    if isinstance(text, str):
        return re.sub(r"\s+", " ", text)
    return re.sub(r"\s+", " ", text[0]["generated_text"])

def clear_text(text: str, remove_punctuation=False) -> str:
    """Clears the input text from non-ASCII characters and special
    characters (\\n, \\t, \\r, \\x0b, \\x0c).
    It normalizes the spaces.

    If `remove_punctuation` is set to True, it also removes
    characters belonging to string.punctutation.
    """
    text = re.sub(f"[^{string.printable}]", " ", text)
    text = re.sub(r"[\n\t\r\x0b\x0c]", " ", text)
    if remove_punctuation:
        text = re.sub(f"[{string.punctuation}]", " ", text)
    clear_text = "".join([c for c in text.encode("ascii", "ignore").decode()])
    clear_text = normalize_spaces(clear_text)
    clear_text = clear_text.strip()

    if remove_punctuation:
        return clear_text
    
    return clear_text if clear_text[-1] in string.punctuation else clear_text + "."

def levenshtein_distance(s1: str, s2: str) -> int:
    """Computes the Levenshtein distance between two string."""
    n, m = len(s1), len(s2)
    # Create an array of size n x m
    dp = [[0 for _ in range(m + 1)] for _ in range(n + 1)]

    # Base case: when n = 0
    for j in range(m + 1):
        dp[0][j] = j
    # Base case: when m = 0
    for i in range(n + 1):
        dp[i][0] = i

    # Transitions/steps
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],  # Insertion
                    dp[i][j - 1],  # Deletion
                    dp[i - 1][j - 1],  # Replacement
                )

    return dp[n][m]


def compute_WER(reference: str, hypothesis: str, verbose=False) -> float:
    """Computes the Word Error Rate (WER) between two sentences based on the
    Levenshtein distance.

    Set `verbose` to True for debugging output.
    """
    hypothesis = clear_text(hypothesis, remove_punctuation=True)
    reference = clear_text(reference, remove_punctuation=True)

    pred_words = hypothesis.split()
    ground_truth_words = reference.split()

    if len(pred_words) < len(ground_truth_words):
        if verbose:
            print(
                f"""Extending pred_words:
                Original pred_words:
                {pred_words}
                ground_truth_words:
                {ground_truth_words}"""
            )
        pred_words.extend("" for _ in range(len(ground_truth_words) - len(pred_words)))
    elif len(ground_truth_words) < len(pred_words):
        if verbose:
            print(
                f"""Extending ground_truth_words:
                Original pred_words:
                {pred_words}
                ground_truth_words:
                {ground_truth_words}"""
            )
        ground_truth_words.extend(
            "" for _ in range(len(pred_words) - len(ground_truth_words))
        )

    assert len(pred_words) == len(ground_truth_words), (
        "ERROR:\nPREDICTED_WORDS = " + pred_words + "\nGT_WORDS = " + ground_truth_words
    )

    correct_score = 0
    error_score = 0
    for w1, w2 in zip(pred_words, ground_truth_words):
        if w1.lower() == w2.lower():
            correct_score += 1
        else:
            error_score += levenshtein_distance(w1.lower(), w2.lower())

    if correct_score + error_score != 0:
        return error_score / (correct_score + error_score)
    else:
        return -1.0


def compute_CER(reference: str, hypothesis: str) -> float:
    """Computes the Character Error Rate (CER) between two words based on the
    Levenshtein distance.
    """
    hypothesis = clear_text(hypothesis, remove_punctuation=True)
    reference = clear_text(reference, remove_punctuation=True)

    if len(reference) > len(hypothesis):
        hypothesis += "".join([" " for _ in range(len(reference) - len(hypothesis))])
    elif len(hypothesis) > len(reference):
        reference += "".join([" " for _ in range(len(hypothesis) - len(reference))])
    assert len(reference) == len(hypothesis), f"Couldn't extend strings: {reference}, {hypothesis}"
    ref_len = len(reference)
    cer = levenshtein_distance(reference, hypothesis) / ref_len if ref_len > 0 else 0
    return cer


def compute_avg_CER(reference: str, hypothesis: str) -> float:
    """Computes the average Character Error Rate (CER) between two sentences
    based on the single words' CER.
    """
    hypothesis = clear_text(hypothesis, remove_punctuation=True)
    reference = clear_text(reference, remove_punctuation=True)

    reference_words = reference.split()
    hypothesis_words = hypothesis.split()

    total_cer = sum(
        compute_CER(ref, hyp) for ref, hyp in zip(reference_words, hypothesis_words)
    )
    average_cer = total_cer / len(reference_words) if len(reference_words) > 0 else 0
    return average_cer


def compute_accuracy(reference: str, hypothesis: str, consider_punctuation=True) -> float:
    """Computes accuracy between two sentences based on equality
    between characters' codes.

    Set `consider_punctuation` to False to ignore punctuation characters.
    """
    metric = evaluate.load("accuracy")

    hypothesis = normalize_spaces(hypothesis)
    reference = normalize_spaces(reference)
    if not consider_punctuation:
        hypothesis = clear_text(hypothesis, remove_punctuation=True)
        reference = clear_text(reference, remove_punctuation=True)

    hypothesis = [t for t in hypothesis]
    reference = [t for t in reference]

    ord_predicted_chars = list(map(lambda c: ord(c), hypothesis))
    ord_gt_chars = list(map(lambda c: ord(c), reference))

    if len(ord_gt_chars) > len(ord_predicted_chars):
        ord_predicted_chars.extend([-1 for _ in range(len(ord_gt_chars) - len(ord_predicted_chars))])
    elif len(ord_gt_chars) < len(ord_predicted_chars):
        ord_gt_chars.extend([-1 for _ in range(len(ord_predicted_chars) - len(ord_gt_chars))])

    assert len(ord_predicted_chars) == len(ord_gt_chars), (
        "ERROR:\nPREDICTED_CHARS = "
        + ord_predicted_chars
        + "\nGT_CHARS = "
        + ord_gt_chars
    )

    return metric.compute(
        predictions=ord_predicted_chars, 
        references=ord_gt_chars)["accuracy"]


class CorrectionStrategy(Enum):
    """Correction Strategy:
    used in the Advanced Spell Checker and OCR Spell Checker to specify the 
    spell correction strategy. Options include:
    1.  NORVIG: exploit the Python Spell Checker 
        (https://github.com/barrust/pyspellchecker) module which is based
        on the Norvig's spelling corrector 
        (https://norvig.com/spell-correct.html); 
    2.  SYMSPELL: exploit the SymSpell spelling corrector 
        (https://github.com/wolfgarbe/SymSpell);
    3.  BART: exploit a fine tuned version of the Facebook AI BART model
        (https://huggingface.co/oliverguhr/spelling-correction-english-base);
    4.  BART_NORVIG: exploit a combination of the NORVIG strategy and
        a fine tuned version of the Facebook AI BART model
        (https://huggingface.co/oliverguhr/spelling-correction-english-base);
    5.  BART_SYMSPELL: exploit a combination of the SYMSPELL strategy and
        a fine tuned version of the Facebook AI BART model
        (https://huggingface.co/oliverguhr/spelling-correction-english-base);
    """
    NORVIG = 1,
    SYMSPELL = 2,
    BART = 3,
    BART_NORVIG = 4,
    BART_SYMSPELL = 5

class AdvancedSpellChecker:
    """Advanced Spell Checker:
    it corrects sentences based on the sorrounding context exploiting:
    1.  the pyspellchecker module (https://github.com/barrust/pyspellchecker),
        which provides an extended implementation of Peter Norvig's spelling
        corrector (https://norvig.com/spell-correct.html); 
    2.  the SymSpell spelling corrector (https://github.com/wolfgarbe/SymSpell);
    3.  an LLM, namely the Facebook AI BART model (https://arxiv.org/abs/1910.13461) 
        fine tuned for spelling correction
        (https://huggingface.co/oliverguhr/spelling-correction-english-base).
    Due to the model involved it works only for the English language.
    """

    EN_ONE_LETTER_WORDS = set(['i', 'a'])

    def __init__(
        self,
        max_new_tokens=DEFAULT_MAX_LEN,
        num_beams=DEFAULT_NUM_BEAMS,
        temperature=DEFAULT_TEMPERATURE,
        max_edit_distance=DEFAULT_MAX_EDIT_DISTANCE,
    ) -> None:
        self.spell = SpellChecker()
        self.sym_spell = SymSpell(
            max_dictionary_edit_distance=(
                max_edit_distance
                if max_edit_distance >= 2
                else DEFAULT_MAX_EDIT_DISTANCE
            ),
            prefix_length=10,
        )
        dictionary_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_dictionary_en_82_765.txt"
        )
        bigram_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_bigramdictionary_en_243_342.txt"
        )
        self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
        self.sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

        self.bart_spelling_model = pipeline(
            "text2text-generation",
            model=DEFAULT_SPELLING_MODEL,
            framework="pt",
            device=(torch.cuda.current_device() if torch.cuda.is_available() else "cpu"),
            max_new_tokens=(max_new_tokens if max_new_tokens > 20 else DEFAULT_MAX_LEN),
            num_beams=(num_beams if num_beams >= 1 else DEFAULT_NUM_BEAMS),
            temperature=temperature,
        )
        self.params = {
            "max_length": (max_new_tokens if max_new_tokens > 20 else DEFAULT_MAX_LEN),
            "max_new_tokens": (
                max_new_tokens if max_new_tokens > 20 else DEFAULT_MAX_LEN
            ),
            "num_beams": (num_beams if num_beams >= 1 else DEFAULT_NUM_BEAMS),
            "temperature": temperature,
        }

    def _batch(self, correction_function, text, *args, **kwargs):
        words = text.split()
        if len(words) < self.params["max_new_tokens"]:
            return correction_function(text, *args, **kwargs)
        
        batches = [" ".join(words[i:i + self.params["max_new_tokens"] - 1]) for i in range(0, len(text), self.params["max_new_tokens"] - 1)]
        batches = list(filter(lambda x: len(x) > 0, batches))
        return normalize_spaces(" ".join([correction_function(batch, *args, **kwargs) for batch in batches]))

    def _n_correction(self, text: str, verbose=False) -> str:
        corrected_text = []
        words = text.split()

        for i, word in enumerate(words):
            correction = normalize_spaces(word if self.spell.correction(word) is None else self.spell.correction(word))
            correction = re.sub(r"[.,;:]", "", correction)
            word = re.sub(r"[.,;:]", "", word)
            if (
                len(word) == 1
                and word.lower() in AdvancedSpellChecker.EN_ONE_LETTER_WORDS
            ) or (len(word) > 1 and correction.lower() == word.lower()):
                if verbose:
                    print("Already correct word:", word)
                corrected_text.append(word)
            else:
                # Try to correct the current word concatenating substrings of the next word
                if verbose:
                    print(
                        "Detected wrong word:",
                        word,
                        "| candidates corrections:",
                        self.spell.candidates(word),
                    )
                if i < len(words) - 1:
                    last_combined_word = (
                        self.spell.correction(word)
                        if self.spell.candidates(word) is not None
                        else word
                    )
                    next_word = words[i + 1]

                    for j in range(len(next_word) + 1):
                        combined_word = word + next_word[:j]
                        if self.spell.correction(combined_word) == combined_word:
                            last_combined_word = combined_word
                    corrected_text.append(last_combined_word)
                    if last_combined_word == word + next_word:
                        del words[i + 1]
                else:
                    corrected_text.append(
                        self.spell.correction(word)
                        if self.spell.candidates(word) is not None
                        else word
                    )
        corrected_text = " ".join(corrected_text)

        return corrected_text
    
    def n_correction(self, text: str, verbose=False) -> str:
        return self._batch(self._n_correction, text, verbose)

    def _n_correction_plus(self, text: str, verbose=False) -> str:
        corrected_text = []
        words = text.split()

        for i, word in enumerate(words):
            correction = normalize_spaces(self.bart_spelling_model(word, **self.params))
            correction = re.sub(r"[.,;:]", "", correction)
            word = re.sub(r"[.,;:]", "", word)
            if (
                len(word) == 1
                and word.lower() in AdvancedSpellChecker.EN_ONE_LETTER_WORDS
            ) or (len(word) > 1 and correction.lower() == word.lower()):
                if verbose:
                    print("Already correct word:", word)
                corrected_text.append(word)
            else:
                # Try to correct the current word concatenating substrings of the next word
                if verbose:
                    print(
                        "Detected wrong word:",
                        word,
                        "| candidates corrections:",
                        self.spell.candidates(word),
                    )
                if i < len(words) - 1:
                    last_combined_word = (
                        self.spell.correction(word)
                        if self.spell.candidates(word) is not None
                        else word
                    )
                    next_word = words[i + 1]

                    for j in range(len(next_word) + 1):
                        combined_word = word + next_word[:j]
                        if self.spell.correction(combined_word) == combined_word:
                            last_combined_word = combined_word
                    corrected_text.append(last_combined_word)
                    if last_combined_word == word + next_word:
                        del words[i + 1]
                else:
                    corrected_text.append(
                        self.spell.correction(word)
                        if self.spell.candidates(word) is not None
                        else word
                    )
        corrected_text = " ".join(corrected_text)

        return corrected_text
    
    def n_correction_plus(self, text: str, verbose=False) -> str:
        return self._batch(self._n_correction_plus, text, verbose)

    def _symspell_correction(self, text: str) -> str:
        segmentation_res = self.sym_spell.word_segmentation(
            text, max_edit_distance=self.sym_spell._max_dictionary_edit_distance
        ).corrected_string
        return self.sym_spell.lookup_compound(
            segmentation_res,
            max_edit_distance=self.sym_spell._max_dictionary_edit_distance,
            transfer_casing=True,
        )[0].term
    
    def symspell_correction(self, text: str) -> str:
        return self._batch(self._symspell_correction, text)

    def _correct_with_strat(self, text: str, strategy: CorrectionStrategy, verbose=False) -> str:
        if strategy == CorrectionStrategy.NORVIG:
            return self.n_correction(text, verbose)
        elif strategy == CorrectionStrategy.SYMSPELL:
            return self.symspell_correction(text)
        elif strategy == CorrectionStrategy.BART_NORVIG:
            return self.correct(text, verbose)
        elif strategy == CorrectionStrategy.BART_SYMSPELL:
            return self.correct_fast(text, verbose)
        raise ValueError("Correction strategy must be one of: `CorrectionStrategy` enum!")

    def correct_with_strat(self, text: str, strategy: CorrectionStrategy, verbose=False) -> str:
        return self._batch(self._correct_with_strat, text, strategy, verbose)
    
    def _correct(self, text: str, verbose=False) -> str:
        pred_text_n = self.n_correction_plus(text, verbose)
        pred_text_nb = normalize_spaces(
            self.bart_spelling_model(pred_text_n, **self.params)
        )
        if verbose:
            print(f"Original text                    : {text}")
            print(f"First spell checker correction   : {pred_text_n}")
            print(f"Advanced spell checker correction: {pred_text_nb}")
            print()
        return pred_text_nb
    
    def correct(self, text: str, verbose=False) -> str:
        return self._batch(self._correct, text, verbose)

    def _correct_fast(self, text: str, verbose=False) -> str:
        pred_text_sym = self.symspell_correction(text)
        pred_text_symb = normalize_spaces(
            self.bart_spelling_model(pred_text_sym, **self.params)
        )
        if verbose:
            print(f"Original text                    : {text}")
            print(f"First spell checker correction   : {pred_text_sym}")
            print(f"Advanced spell checker correction: {pred_text_symb}")
            print()
        return pred_text_symb

    def correct_fast(self, text: str, verbose=False) -> str:
        return self._batch(self._correct_fast, text, verbose)
    
    def _correct_llm(self, text: str, verbose=False) -> str:
        pred_text_b = normalize_spaces(
            self.bart_spelling_model(text, **self.params)
        )
        if verbose:
            print(f"Original text : {text}")
            print(f"LLM correction: {pred_text_b}")
            print()
        return pred_text_b
    
    def correct_llm(self, text: str, verbose=False) -> str:
        return self._batch(self._correct_llm, text, verbose)
    
    def correct_and_eval(
        self, text: str, ground_truth: str, number_of_errors=-1, 
        measure_time=False, verbose=False) -> tuple[str, dict]:
        stats = {}
        if measure_time:
            start = time.perf_counter_ns()
            pred_text_nb = self.correct(text, verbose)
            end = time.perf_counter_ns()
            elapsed = end - start
            stats["elapsed_time_ms"] = int(elapsed / (10 ** 6))
        else:
            pred_text_nb = self.correct(text, verbose)
        if verbose:
            print(f"Ground truth text                : {ground_truth}")
            if measure_time:
                print(f"Elapsed time (ms): {elapsed / (10 ** 6)}")
            print()
        
        stats.update({
            "ground_truth_text": ground_truth,
            "introduced_errors": (number_of_errors if number_of_errors >= 0 else None),
            "error_rate": (number_of_errors / len(ground_truth) if number_of_errors >= 0 and len(ground_truth) > 0 else None),
            "corrected_text": pred_text_nb,
            "accuracy_punct": compute_accuracy(ground_truth, pred_text_nb, True),
            "accuracy_no_punct": compute_accuracy(ground_truth, pred_text_nb, False),
            "WER": compute_WER(ground_truth, pred_text_nb),
            "avg_CER": compute_avg_CER(ground_truth, pred_text_nb),
        })

        return pred_text_nb, stats

    def correct_fast_and_eval(
        self, text: str, ground_truth: str, number_of_errors=-1, 
        measure_time=False, verbose=False) -> tuple[str, dict]:
        stats = {}
        if measure_time:
            start = time.perf_counter_ns()
            pred_text_symb = self.correct_fast(text, verbose)
            end = time.perf_counter_ns()
            elapsed = end - start
            stats["elapsed_time_ms"] = int(elapsed / (10 ** 6))
        else:
            pred_text_symb = self.correct_fast(text, verbose)
        if verbose:
            print(f"Ground truth text                : {ground_truth}")
            if measure_time:
                print(f"Elapsed time (ms): {elapsed / (10 ** 6)}")
            print()
        
        stats.update({
            "ground_truth_text": ground_truth,
            "introduced_errors": (number_of_errors if number_of_errors >= 0 else None),
            "error_rate": (number_of_errors / len(ground_truth) if number_of_errors >= 0 and len(ground_truth) > 0 else None),
            "corrected_text": pred_text_symb,
            "accuracy_punct": compute_accuracy(ground_truth, pred_text_symb, True),
            "accuracy_no_punct": compute_accuracy(ground_truth, pred_text_symb, False),
            "WER": compute_WER(ground_truth, pred_text_symb),
            "avg_CER": compute_avg_CER(ground_truth, pred_text_symb),
        })

        return pred_text_symb, stats
    
    def n_correction_and_eval(
        self, text: str, ground_truth: str, number_of_errors=-1, 
        measure_time=False, verbose=False) -> tuple[str, dict]:
        stats = {}
        if measure_time:
            start = time.perf_counter_ns()
            pred_text_n = self.n_correction(text, verbose)
            end = time.perf_counter_ns()
            elapsed = end - start
            stats["elapsed_time_ms"] = int(elapsed / (10 ** 6))
        else:
            pred_text_n = self.n_correction(text, verbose)
        if verbose:
            print(f"Ground truth text                : {ground_truth}")
            print(f"Text                             : {text}")
            print(f"Norvig SpellChecker correction   : {pred_text_n}")
            if measure_time:
                print(f"Elapsed time (ms): {elapsed / (10 ** 6)}")
            print()
        
        stats.update({
            "ground_truth_text": ground_truth,
            "introduced_errors": (number_of_errors if number_of_errors >= 0 else None),
            "error_rate": (number_of_errors / len(ground_truth) if number_of_errors >= 0 and len(ground_truth) > 0 else None),
            "corrected_text": pred_text_n,
            "accuracy_punct": compute_accuracy(ground_truth, pred_text_n, True),
            "accuracy_no_punct": compute_accuracy(ground_truth, pred_text_n, False),
            "WER": compute_WER(ground_truth, pred_text_n),
            "avg_CER": compute_avg_CER(ground_truth, pred_text_n),
        })

        return pred_text_n, stats
    
    def llm_correction_and_eval(
        self, text: str, ground_truth: str, number_of_errors=-1, 
        measure_time=False, verbose=False) -> tuple[str, dict]:
        stats = {}
        if measure_time:
            start = time.perf_counter_ns()
            pred_text_b = self.correct_llm(text, verbose)
            end = time.perf_counter_ns()
            elapsed = end - start
            stats["elapsed_time_ms"] = int(elapsed / (10 ** 6))
        else:
            pred_text_b = self.correct_llm(text, verbose)
        if verbose:
            print(f"Ground truth text: {ground_truth}")
            print(f"Text             : {text}")
            print(f"LLM correction   : {pred_text_b}")
            if measure_time:
                print(f"Elapsed time (ms): {elapsed / (10 ** 6)}")
            print()
        
        stats.update({
            "ground_truth_text": ground_truth,
            "introduced_errors": (number_of_errors if number_of_errors >= 0 else None),
            "error_rate": (number_of_errors / len(ground_truth) if number_of_errors >= 0 and len(ground_truth) > 0 else None),
            "corrected_text": pred_text_b,
            "accuracy_punct": compute_accuracy(ground_truth, pred_text_b, True),
            "accuracy_no_punct": compute_accuracy(ground_truth, pred_text_b, False),
            "WER": compute_WER(ground_truth, pred_text_b),
            "avg_CER": compute_avg_CER(ground_truth, pred_text_b),
        })

        return pred_text_b, stats
    
    def symspell_correction_and_eval(
        self, text: str, ground_truth: str, number_of_errors=-1, 
        measure_time=False, verbose=False) -> tuple[str, dict]:
        stats = {}
        if measure_time:
            start = time.perf_counter_ns()
            pred_text_sym = self.symspell_correction(text)
            end = time.perf_counter_ns()
            elapsed = end - start
            stats["elapsed_time_ms"] = int(elapsed / (10 ** 6))
        else:
            pred_text_sym = self.symspell_correction(text)
        if verbose:
            print(f"Ground truth text                : {ground_truth}")
            print(f"Text                             : {text}")
            print(f"SymSpell correction              : {pred_text_sym}")
            if measure_time:
                print(f"Elapsed time (ms): {elapsed / (10 ** 6)}")
            print()
        
        stats.update({
            "ground_truth_text": ground_truth,
            "introduced_errors": (number_of_errors if number_of_errors >= 0 else None),
            "error_rate": (number_of_errors / len(ground_truth) if number_of_errors >= 0 and len(ground_truth) > 0 else None),
            "corrected_text": pred_text_sym,
            "accuracy_punct": compute_accuracy(ground_truth, pred_text_sym, True),
            "accuracy_no_punct": compute_accuracy(ground_truth, pred_text_sym, False),
            "WER": compute_WER(ground_truth, pred_text_sym),
            "avg_CER": compute_avg_CER(ground_truth, pred_text_sym),
        })

        return pred_text_sym, stats
        
    def correct_with_strat_and_eval(
        self, text: str, ground_truth: str, strategy: CorrectionStrategy,
        number_of_errors=-1, measure_time = False, verbose=False) -> tuple[str, dict]:
        if strategy == CorrectionStrategy.NORVIG:
            return self.n_correction_and_eval(text, ground_truth, number_of_errors, measure_time, verbose)
        elif strategy == CorrectionStrategy.SYMSPELL:
            return self.symspell_correction_and_eval(text, ground_truth, number_of_errors, measure_time, verbose)
        elif strategy == CorrectionStrategy.BART:
            return self.llm_correction_and_eval(text, ground_truth, number_of_errors, measure_time, verbose)
        elif strategy == CorrectionStrategy.BART_NORVIG:
            return self.correct_and_eval(text, ground_truth, number_of_errors, measure_time, verbose)
        elif strategy == CorrectionStrategy.BART_SYMSPELL:
            return self.correct_fast_and_eval(text, ground_truth, number_of_errors, measure_time, verbose)
        raise ValueError("Correction strategy must be one of: `CorrectionStrategy` enum!")



class OCRSpellChecker(AdvancedSpellChecker):
    """OCR Spell Checker (based on the AdvancedSpellChecker class):
    it exploits the Advanced Spell Checker to correct OCR texts (from images) 
    or PDFs using:
    1.  the pytesseract module for OCR based on Google's Tesseract
        (https://github.com/tesseract-ocr/tesseract);
    2.  the PyPDF2 package to read PDF documents.
    """
    def __init__(
        self,
        max_new_tokens=DEFAULT_MAX_LEN,
        num_beams=DEFAULT_NUM_BEAMS,
        temperature=DEFAULT_TEMPERATURE,
        max_edit_distance=DEFAULT_MAX_EDIT_DISTANCE,
    ) -> None:
        super().__init__(max_new_tokens, num_beams, temperature, max_edit_distance)

    SUPPORTED_IMG_FORMATS = ["png", "jpg", "jpeg", "gif", "tiff", "tif"]

    @staticmethod
    def is_image(path: str) -> bool:
        return os.path.isfile(path) and any(path.lower().endswith('.' + ext) for ext in OCRSpellChecker.SUPPORTED_IMG_FORMATS)

    @staticmethod
    def is_pdf(path: str) -> bool:
        return os.path.isfile(path) and path.lower().endswith(".pdf")
    
    def read_image_from_path(self, path: str) -> str:
        if OCRSpellChecker.is_image(path):
            return pytesseract.image_to_string(Image.open(path), lang="eng")
        raise Exception(f"{path} is not in a supported image format! Supported image formats are: {OCRSpellChecker.SUPPORTED_IMG_FORMATS}")

    def read_image(self, img: Image) -> str:
        return pytesseract.image_to_string(img, lang="eng")
    
    def read_pdf_from_path(self, path: str, page=0, read_all=False) -> str:
        if OCRSpellChecker.is_pdf(path):
            reader = PdfReader(path)
            if read_all:
                num_pages = len(reader.pages)
                return "\n".join([reader.pages[p].extract_text(0) for p in range(num_pages)])
            else:
                return reader.pages[page].extract_text(0)
        raise Exception(f"{path} is not a PDF file!")
    
    def read_pdf(self, reader: PdfReader, page=0, read_all=False) -> str:
        if read_all:
            num_pages = len(reader.pages)
            return "\n".join([reader.pages[p].extract_text(0) for p in range(num_pages)])
        else:
            return reader.pages[page].extract_text(0)