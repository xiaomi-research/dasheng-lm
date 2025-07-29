import re
import json
import argparse

import zhconv
import editdistance as ed
from evaluate_tokenizer import EvaluationTokenizer
from whisper_normalizer.english import EnglishTextNormalizer
from whisper_normalizer.basic import BasicTextNormalizer
from cn_tn import TextNorm


english_normalizer = EnglishTextNormalizer()
chinese_normalizer = TextNorm(
    to_banjiao=False,
    to_upper=False,
    to_lower=False,
    remove_fillers=False,
    remove_erhua=False,
    check_chars=False,
    remove_space=False,
    cc_mode="",
)
basic_normalizer = BasicTextNormalizer()


def remove_sp(text, language):
    PUNCS = "!,.?;:"
    gt = re.sub(r"<\|.*?\|>", " ", text)
    gt = re.sub(r"\s+", r" ", gt)
    gt = re.sub(f" ?([{PUNCS}])", r"\1", gt)
    gt = gt.lstrip(" ")
    if language == "zh":
        gt = re.sub(r"\s+", r"", gt)
    return gt


def compute_wer(result_file):
    tokenizer = EvaluationTokenizer(tokenizer_type="none", lowercase=True, punctuation_removal=True, character_tokenization=False)

    distance = 0
    ref_length = 0
    print_count = 10
    print_index = 0
    sample_count = 0
    with open(result_file, "r", encoding="utf8") as reader:
        for line in reader:
            json_obj = json.loads(line)

            ref = json_obj["text"]
            pred = json_obj["model_output"]
            language = json_obj["lang"]

            ref = remove_sp(ref, language)
            pred = remove_sp(pred, language)

            # normalize text
            if language in ["yue"]:
                ref = zhconv.convert(ref, "zh-cn")
                pred = zhconv.convert(pred, "zh-cn")
            if language in ["en"]:
                ref = english_normalizer(ref)
                pred = english_normalizer(pred)
            if language in ["zh"]:
                ref = chinese_normalizer(ref)
                pred = chinese_normalizer(pred)
            else:
                ref = basic_normalizer(ref)
                pred = basic_normalizer(pred)

            # token
            ref_items = tokenizer.tokenize(ref).split()
            pred_items = tokenizer.tokenize(pred).split()
            if language in ["zh", "yue"]:
                ref_items = [x for x in "".join(ref_items)]
                pred_items = [x for x in "".join(pred_items)]
            
            if len(ref_items) <= 0 or len(pred_items) <= 0:
                continue
            if print_index <= print_count:
                print(f"ref: {ref}")
                print(f"pred: {pred}")
                print(f"ref_items:\n{ref_items}\n{len(ref_items)}\n{ref_items[0]}")
                print(f"pred_items:\n{pred_items}\n{len(ref_items)}\n{ref_items[0]}")
                print_index += 1

            distance += ed.eval(ref_items, pred_items)
            ref_length += len(ref_items)
            sample_count += 1

    wer = distance / ref_length
    print(f"----- Dataset: {json_obj['dataset_name']}, WER: {wer} -----")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute WER.")
    parser.add_argument('-i', '--input', help="Experimental Result", required=True)
    args = parser.parse_args()
    compute_wer(args.input)
