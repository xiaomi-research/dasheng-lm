import re
import json
import argparse
import itertools
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute ACC.')
    parser.add_argument('-i', '--input', help="Experimental Result", required=True)
    args = parser.parse_args()
    with open(args.input, "r", encoding="utf8") as reader:
        count = 0
        correct = 0
        for line in reader:
            temp = json.loads(line)
            ref = temp["label"][0].lower().lstrip().strip()
            hyp = temp["model_output"].lower().lstrip().strip()
            if ref in hyp:
                correct += 1
            count += 1
    print(f"----- Dataset: {temp['dataset_name']}, ACC: {correct / count} -----")
