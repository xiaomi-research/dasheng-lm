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
