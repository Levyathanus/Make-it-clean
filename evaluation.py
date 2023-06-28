import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os

from asc import (
    clear_text,
    AdvancedSpellChecker, 
    CorrectionStrategy, 
    OCRSpellChecker,
)

ASCII_CHARS = list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!(),./:;?")

EVAL_DATASET_PATH = "./data/EVAL_100K/data_gt_mod.json"
CORRECTED_DATASET_PATH = "./data/EVAL_100K/data_corr.json"
CORRECTED_OCR_DIR = "./data/old_books/"
STATISTICS_FILE_PATH = "./all_stats.json"

NORVIG_LABEL = "norvig"
SYMSPELL_LABEL = "symspell"
BART_LABEL = "bart"
BART_NORVIG_LABEL = "bart_norvig"
BART_SYMSPELL_LABEL = "bart_symspell"
LABELS = [NORVIG_LABEL, SYMSPELL_LABEL, BART_LABEL, BART_NORVIG_LABEL, BART_SYMSPELL_LABEL]

MEASURES = ["elapsed_time_ms", "accuracy_punct", "accuracy_no_punct", "WER", "avg_CER"]
AVG_MEASURES = ["avg_elapsed_time_ms", "avg_accuracy_punct", "avg_accuracy_no_punct", "avg_WER", "avg_avg_CER"]

PLOT_TITLES = {
    "avg_elapsed_time_ms": r"\textbf{Average execution time (ms)}", 
    "avg_accuracy_punct": r"\textbf{Average accuracy}", 
    "avg_accuracy_no_punct": r"\textbf{Average accuracy (no punctuation)}", 
    "avg_WER": r"\textbf{Average Word Error Rate (WER)}", 
    "avg_avg_CER": r"\textbf{Average Character Error Rate (CER)}"
}

PLOT_LABELS_Y = {
    "avg_elapsed_time_ms": "Execution time [ms]", 
    "avg_accuracy_punct": "", 
    "avg_accuracy_no_punct": "", 
    "avg_WER": "", 
    "avg_avg_CER": ""
}

PLOT_COLORS = {
    NORVIG_LABEL: "red",
    SYMSPELL_LABEL: "green",
    BART_LABEL: "violet",
    BART_NORVIG_LABEL: "blue",
    BART_SYMSPELL_LABEL: "orange"
}

params = {
    "max_new_tokens": 128,
    "num_beams": 4,
    "temperature": 0.5,
    "max_edit_distance": 3,
}
spell = AdvancedSpellChecker(**params)
spell_ocr = OCRSpellChecker(**params)

def is_json_file(path: str) -> bool:
    return os.path.isfile(path) and path.endswith(".json")

def is_txt_file(path: str) -> bool:
    return os.path.isfile(path) and path.endswith(".txt")

def split(l: list, n: int) -> list:
    k, m = divmod(len(l), n)
    return [l[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

def get_file_name(path: str) -> str:
    name = path
    if path.find('/') != -1:
        base_name = path.split('/')[-1]
        if base_name.find('.') != -1:
            name = base_name.split('.')[0]
    elif path.find('\\') != -1:
        base_name = path.split('\\')[-1]
        if base_name.find('.') != -1:
            name = base_name.split('.')[0]
    elif path.find('.') != -1:
        name = path.split('.')[0]
    return name

def evaluate_ocr(image_path: str, gt_file_path: str) -> dict:
    if OCRSpellChecker.is_image(image_path) and is_txt_file(gt_file_path):
        gt_text = ""
        with open(gt_file_path, 'r', encoding="UTF-8") as f:
            gt_text = clear_text(f.read())
        text = clear_text(spell_ocr.read_image_from_path(image_path))
        _, n_correction = spell_ocr.correct_with_strat_and_eval(
            text, 
            gt_text,
            CorrectionStrategy.NORVIG,
            measure_time=True,
            verbose=False
        )
        _, sym_correction = spell_ocr.correct_with_strat_and_eval(
            text, 
            gt_text,
            CorrectionStrategy.SYMSPELL,
            measure_time=True,
            verbose=False
        )
        _, bart_correction = spell_ocr.correct_with_strat_and_eval(
            text, 
            gt_text,
            CorrectionStrategy.BART,
            measure_time=True,
            verbose=False
        )
        _, bart_n_correction = spell_ocr.correct_and_eval(
            text, 
            gt_text,
            measure_time=True,
            verbose=False
        )
        _, bart_sym_correction = spell_ocr.correct_fast_and_eval(
            text, 
            gt_text,
            measure_time=True,
            verbose=False
        )
        stats = { "id": get_file_name(image_path) }
        stats[NORVIG_LABEL] = n_correction
        stats[SYMSPELL_LABEL] = sym_correction
        stats[BART_LABEL] = bart_correction
        stats[BART_NORVIG_LABEL] = bart_n_correction
        stats[BART_SYMSPELL_LABEL] = bart_sym_correction

        return stats
    else:
        raise FileNotFoundError(f"{image_path} must be a supported image ({OCRSpellChecker.SUPPORTED_IMG_FORMATS}) and {gt_file_path} must be a text file (.txt)")

def evaluate(path=EVAL_DATASET_PATH) -> list:
    assert is_json_file(path)
    data = []
    with open(path, 'r') as f:
        data = json.load(f)
    assert isinstance(data, list) and len(data) > 0

    corrected_samples = []
    for sample in data:
        text = sample["text"]
        gt_text = sample["ground_truth"]
        introduced_errors = sample["number_of_errors"]
        _, n_correction = spell.correct_with_strat_and_eval(
            text, 
            gt_text,
            CorrectionStrategy.NORVIG,
            number_of_errors=introduced_errors,
            measure_time=True,
            verbose=False
        )
        _, sym_correction = spell.correct_with_strat_and_eval(
            text, 
            gt_text,
            CorrectionStrategy.SYMSPELL,
            number_of_errors=introduced_errors,
            measure_time=True,
            verbose=False
        )
        _, bart_correction = spell.correct_with_strat_and_eval(
            text, 
            gt_text,
            CorrectionStrategy.BART,
            number_of_errors=introduced_errors,
            measure_time=True,
            verbose=False
        )
        _, bart_n_correction = spell.correct_and_eval(
            text, 
            gt_text, 
            number_of_errors=introduced_errors,
            measure_time=True,
            verbose=False
        )
        _, bart_sym_correction = spell.correct_fast_and_eval(
            text, 
            gt_text, 
            number_of_errors=introduced_errors,
            measure_time=True,
            verbose=False
        )
        stats = { "id": sample["id"] }
        stats[NORVIG_LABEL] = n_correction
        stats[SYMSPELL_LABEL] = sym_correction
        stats[BART_LABEL] = bart_correction
        stats[BART_NORVIG_LABEL] = bart_n_correction
        stats[BART_SYMSPELL_LABEL] = bart_sym_correction
        corrected_samples.append(stats)
    return corrected_samples

def evaluate_and_save(evaluation_path: str, saving_path: str) -> None:
    save_evaluations(evaluate(evaluation_path), saving_path)

def evaluate_ocr_and_save(image_path: str, gt_file_path: str, save_file_path=None) -> None:
    if save_file_path is None:
        save_evaluations(evaluate_ocr(image_path, gt_file_path), CORRECTED_OCR_DIR + get_file_name(image_path) + ".json")
    else:
        save_evaluations(evaluate_ocr(image_path, gt_file_path), save_file_path)

def save_evaluations(stats: any, path: str) -> None:
    if path.endswith(".json"):
        with open(path, 'w') as f:
            json.dump(stats, f, indent=2)
    else:
        raise FileNotFoundError(f"{path}: should be a JSON file!")
    
def compute_average_performance(corrected_samples: list, verbose=False) -> tuple[dict, dict]:
    avg_stats = { 
        label: {
            "avg_elapsed_time_ms": -1,
            "avg_introduced_errors": -1,
            "avg_error_rate": -1,
            "avg_accuracy_punct": -1,
            "avg_accuracy_no_punct": -1,
            "avg_WER": -1,
            "avg_avg_CER": -1
        } for label in LABELS}
    
    avg_delta = {
        "avg_delta_elapsed_time_ms": -1,
        "avg_delta_accuracy_punct": -1,
        "avg_delta_accuracy_no_punct": -1,
        "avg_delta_WER": -1,
        "avg_delta_avg_CER": -1
    }

    for sample in corrected_samples:
        for label in LABELS:
            for k, v in sample[label].items():
                if k in MEASURES + ["introduced_errors", "error_rate"]:
                    if avg_stats[label]["avg_" + k] == -1:
                        avg_stats[label]["avg_" + k] = v
                    else:
                        avg_stats[label]["avg_" + k] = avg_stats[label]["avg_" + k] + v
        for (k1, v1), (k2, v2) in zip(sample[BART_NORVIG_LABEL].items(), sample[BART_SYMSPELL_LABEL].items()):
            assert k1 == k2
            if k1 in MEASURES:
                if avg_delta["avg_delta_" + k1] == -1:
                    avg_delta["avg_delta_" + k1] = ((v2 - v1) / v1 if k1 == "elapsed_time_ms" else (v2 - v1))
                else:
                    avg_delta["avg_delta_" + k1] = avg_delta["avg_delta_" + k1] + ((v2 - v1) / v1 if k1 == "elapsed_time_ms" else (v2 - v1))
    for label in LABELS: 
        for k, v in avg_stats[label].items():
            avg_stats[label][k] = v / len(corrected_samples)
    assert len(set(avg_stats[label]["avg_introduced_errors"] for label in LABELS)) <= 1
    assert len(set(avg_stats[label]["avg_error_rate"] for label in LABELS)) <= 1
    for k, v in avg_delta.items():
        avg_delta[k] = v / len(corrected_samples)
    avg_delta.update({"avg_introduced_errors": avg_stats[LABELS[0]]["avg_introduced_errors"]})
    avg_delta.update({"avg_error_rate": avg_stats[LABELS[0]]["avg_error_rate"]})

    if verbose:
        print(avg_stats)
        print(avg_delta)
        print()
        print("Fast SymSpell wrt regular Norvig with {0:.1f} introduced errors on average ({1:2.2%} average error rate):"
            .format(avg_delta['avg_introduced_errors'], avg_delta['avg_error_rate']))
        for k, v in avg_delta.items():
            if k not in ["avg_introduced_errors", "avg_error_rate"]:
                if v < 0:
                    print("There's a {:2.2%} DECREASE for {}".format(-v, k.removeprefix('avg_delta_')))
                else:
                    print("There's a {:2.2%} INCREASE for {}".format(v, k.removeprefix('avg_delta_')))
    return avg_stats, avg_delta

def compute_average_performance_batch(corrected_samples: list, batch_size: int, path=None) -> list:
    n = len(corrected_samples)
    assert n > 0 and batch_size <= n
    batch_stats = []
    num_batches = math.ceil(n / batch_size)
    batches = split(corrected_samples, num_batches)
    for batch in batches:
        avg_stat, _ = compute_average_performance(batch)
        batch_stats.append(avg_stat)

    if path is not None and path.endswith(".json"):
        with open(path, 'a') as f:
            json.dump(batch_stats, f, indent=2)
    
    return batch_stats

def compute_average_performance_from_file(path: str) -> tuple[dict, dict]:
    if is_json_file(path):
        with open(path, 'r') as f:
            corrected_samples = json.load(f)
        return compute_average_performance(corrected_samples)
    else:
        raise FileNotFoundError(f"{path}: should be a JSON file!")

def plot_evaluation(corrected_samples: list, batch_size=1, statistics_file_path=None) -> None:
    if batch_size <= 1 and all("id" in sample.keys() for sample in corrected_samples):
        idxs = [sample["id"] for sample in corrected_samples]
    else:
        idxs = list(range(len(corrected_samples) // batch_size))
    
    if batch_size <= 1:
        avg_stats, avg_delta = compute_average_performance(corrected_samples)
    else:
        corrected_samples = compute_average_performance_batch(corrected_samples, batch_size)

    if batch_size <= 1 and statistics_file_path is not None and statistics_file_path.endswith(".json"):
        with open(statistics_file_path, "a") as f:
            json.dump(avg_stats, f, indent=2)
            json.dump(avg_delta, f, indent=2)

    plt.rcParams['text.usetex'] = True
    for k in AVG_MEASURES:
        yn = []
        ys = []
        yb = []
        ybn = []
        ybs = []
        for sample in corrected_samples:
            yn.append(sample[NORVIG_LABEL][k])
            ys.append(sample[SYMSPELL_LABEL][k])
            yb.append(sample[BART_LABEL][k])
            ybn.append(sample[BART_NORVIG_LABEL][k])
            ybs.append(sample[BART_SYMSPELL_LABEL][k])
        figure = plt.figure()
        ax = figure.add_subplot(1, 1, 1)
        ax.plot(idxs, yn, "o-", color=PLOT_COLORS[NORVIG_LABEL], label="Norvig$^+$")
        ax.plot(idxs, ys, "o-", color=PLOT_COLORS[SYMSPELL_LABEL], label="SymSpell")
        ax.plot(idxs, yb, "o-", color=PLOT_COLORS[BART_LABEL], label="BART")
        ax.plot(idxs, ybn, "o-", color=PLOT_COLORS[BART_NORVIG_LABEL], label="BART + Norvig$^+$")
        ax.plot(idxs, ybs, "o-", color=PLOT_COLORS[BART_SYMSPELL_LABEL], label="BART + SymSpell")
        ax.set_title(k)
        ax.grid(axis='x')
        ax.set_xticks(idxs)
        ax.set_xlabel("Run ID")
        if k != "avg_elapsed_time_ms":
            ax.set_yticks(list(np.arange(0, 1.05, 0.05)))
        else:
            ax.set_ylim(bottom=0)
        ax.set_ylabel(k)
        ax.legend()
        plt.show()

def plot_evaluation_hist(corrected_samples: list, statistics_file_path=None) -> None:
    plt.rcParams['text.usetex'] = True
    idxs = ["Norvig$^+$", "SymSpell", "BART", "BART +\nNorvig$^+$", "BART +\nSymSpell"]
    avg_stats, _ = compute_average_performance(corrected_samples)

    if statistics_file_path is not None and statistics_file_path.endswith(".json"):
        with open(statistics_file_path, 'w') as f:
            json.dump(avg_stats, f, indent=2)

    for k in AVG_MEASURES:
        y_stat = []
        for label in LABELS:
            y_stat.append(avg_stats[label][k])
        figure = plt.figure()
        ax = figure.add_subplot(1, 1, 1)
        ax.bar(idxs, y_stat, color=list(PLOT_COLORS.values()))
        ax.set_title(PLOT_TITLES[k], fontdict={"fontweight": "bold"}, pad=10)
        if k != "avg_elapsed_time_ms":
            if k in ["avg_accuracy_punct", "avg_accuracy_no_punct"]:
                ax.set_yticks(list(np.arange(0, 1.05, 0.05)))
                print(f"MAX {k}: {max(y_stat)}")
            else:
                ax.set_yticks(list(np.arange(0, 1.05, 0.05)))
                print(f"MIN {k}: {min(y_stat)}")
        else:
            ax.set_yticks(list(plt.yticks()[0]))
            print(f"MIN {k}: {min(y_stat)}")
        plt.show()

def plot_evaluation_from_file(path: str, batch_size=1, statistics_file_path=None, hist=False) -> None:
    if is_json_file(path):
        with open(path, 'r') as f:
            corrected_samples = json.load(f)
        if hist:
            plot_evaluation_hist(corrected_samples, statistics_file_path)
        else:
            plot_evaluation(corrected_samples, batch_size, statistics_file_path)
    else:
        raise FileNotFoundError(f"{path}: should be a JSON file!")

def plot_eval_hist_from_file(statistics_file_path: str) -> None:
    if is_json_file(statistics_file_path):
        plt.rcParams['text.usetex'] = True
        idxs = ["Norvig$^+$", "SymSpell", "BART", "BART +\nNorvig$^+$", "BART +\nSymSpell"]
        with open(statistics_file_path, 'r') as f:
            all_stats = json.load(f)
        for k in AVG_MEASURES:
            y_stat = []
            for label in LABELS:
                y_stat.append(all_stats[label][k])
            figure = plt.figure()
            ax = figure.add_subplot(1, 1, 1)
            ax.bar(idxs, y_stat, color=list(PLOT_COLORS.values()))
            ax.set_title(PLOT_TITLES[k], pad=10)
            if k != "avg_elapsed_time_ms":
                ax.set_yticks(list(np.arange(0, 1.05, 0.05)))
            else:
                ax.set_yticks(list(plt.yticks()[0]))
            print(f"MAX {k}: {max(y_stat)}")
            print(f"MIN {k}: {min(y_stat)}")
            plt.show()
    else:
        raise FileNotFoundError(f"{statistics_file_path}: should be a JSON file!") 

def average_from_files(number_of_files: int, prefix: str, statistics_file_path: str) -> dict:
    all_stats = {
        LABEL: {
            "avg_elapsed_time_ms": 0,
            "avg_introduced_errors": 0,
            "avg_error_rate": 0,
            "avg_accuracy_punct": 0,
            "avg_accuracy_no_punct": 0,
            "avg_WER": 0,
            "avg_avg_CER": 0
        }
        for LABEL in LABELS
    }
    count = 0
    for i in range(number_of_files):
        with open(prefix + str(i) + ".json", 'r') as f:
            data = json.load(f)
            count += len(data)
            for sample in data:
                for LABEL in LABELS:
                    for k in MEASURES + ["introduced_errors", "error_rate"]:
                        if sample[LABEL][k] >= 0:
                            all_stats[LABEL]["avg_" + k] += sample[LABEL][k]
                        else:
                            raise Exception(f"Not a valid measure in {prefix + str(i) + '.json'} for sample: {sample}")
    
    all_stats["count"] = count
    for LABEL in LABELS:
        for k in AVG_MEASURES + ["avg_introduced_errors", "avg_error_rate"]:
            all_stats[LABEL][k] /= count
    
    if statistics_file_path.endswith(".json"):
        with open(statistics_file_path, 'w') as f:
            json.dump(all_stats, f, indent=2)
    return all_stats
            

if __name__ == "__main__":
    average_from_files(10, "./data/EVAL_100K/data_corr_", STATISTICS_FILE_PATH)
    plot_eval_hist_from_file(STATISTICS_FILE_PATH)