import json
import argparse
from fense.evaluator import Evaluator


def do_compute(input_file, fense_evaluator):
    fense = []
    with open(input_file, "r", encoding="utf8") as reader:
        for line in reader:
            json_obj = json.loads(line)
            if "text" in json_obj:
                ref = json_obj["text"]
            else:
                ref = json_obj["answer"]
            hyp = json_obj["model_output"]
            score, error_prob, penalized_score = fense_evaluator.sentence_score(hyp, ref, return_error_prob=True)
            fense.append(score)
    print(f"----- Dataset: {json_obj['dataset_name']}, FENSE: {sum(fense) / len(fense)} -----")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute FENSE.")
    parser.add_argument('-i', '--input', help="Experimental Result", required=True)
    args = parser.parse_args()
    input_file = args.input
    fense_evaluator = Evaluator(device='cpu', sbert_model='paraphrase-TinyBERT-L6-v2', echecker_model='echecker_clotho_audiocaps_base')
    do_compute(input_file, fense_evaluator)
