# Copyright 2025 Horizon Team, MiLM Plus, Xiaomi Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import json
import argparse
import itertools
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, average_precision_score


def process_text(text, model_name, dataset_name):
    if 'qwen' in model_name.lower():
        if ";" in text:
            text = text.split(";")
        else:
            text = text.split(",")
        if dataset_name == 'FSD50K':
            text = [t.lstrip().strip().replace("_", " ").replace(" - ", " and ") for t in text]
        else:
            text = [t.lstrip().strip().replace("_", " ").replace(" - ", ", ") for t in text]  
    elif 'kimi' in model_name.lower():
        text = text.split(",")
        text = [t.lstrip().strip().replace("_", " ") for t in text] 
    else:
        text = text.split(";")
        if dataset_name == 'FSD50K':
            text = [t.lstrip().strip().replace(", and ", " and ").replace(", ", " and ") for t in text]
        else:
            text = [t.lstrip().strip() for t in text] 
    return text


def get_mAP(ref, pred):
    unique_labels = set(itertools.chain(*[s for s in refs]))
    pred_res = []
    for i in range(len(ref)):
        pred_res.append([j for j in pred[i] if j in unique_labels])
    multi = MultiLabelBinarizer().fit_transform(ref + pred_res)
    ref_multi = multi[:len(ref)]
    hyp_multi = multi[len(ref):]
    return average_precision_score(ref_multi, hyp_multi, average="macro")


def get_mAP_ours(ref, pred):
    unique_labels = set(itertools.chain(*[s for s in refs]))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    target_tensor = np.zeros((len(ref), len(unique_labels)), dtype=np.int64)
    pred_tensor = np.zeros((len(pred), len(unique_labels)), dtype=np.int64)

    for i, labels in enumerate(ref):
        indices = [label_to_index[j] for j in labels if j in label_to_index]
        target_tensor[i, indices] = 1

    for i, labels in enumerate(pred):
        indices = [label_to_index[j] for j in labels if j in label_to_index]
        pred_tensor[i, indices] = 1
    return average_precision_score(target_tensor, pred_tensor, average="macro")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute mAP.')
    parser.add_argument('-i', '--input', help="Experimental Result", required=True)
    args = parser.parse_args()

    data = []
    refs, hyps = [], []
    with open(args.input, "r", encoding="utf8") as reader:
        for line in reader:
            temp = json.loads(line)
            refs.append([s.lower() for s in temp["label"]])
            hypothesis = temp["model_output"].lower()
            hyps.append(process_text(hypothesis, temp['model_name'], temp['dataset_name']))
    score = get_mAP(refs, hyps)
    print(f"----- Dataset: {temp['dataset_name']}, mAP: {score} -----")
