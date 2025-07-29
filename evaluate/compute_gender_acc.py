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
    refs, hyps = [], []
    with open(args.input, "r", encoding="utf8") as reader:
        for line in reader:
            temp = json.loads(line)
            hyp = temp["model_output"].lower().lstrip().strip()
            if ("male" in hyp) and ("female" not in hyp):
                hyp = "male"
            elif ("female" in hyp) and ("male" not in hyp.replace("female", "")):
                hyp = "female" 
            refs.append(temp["label"][0].lower().lstrip().strip())
            hyps.append(hyp)
    score = accuracy_score(refs, hyps)
    print(f"----- Dataset: {temp['dataset_name']}, ACC: {score} -----")
