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

import json
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute ACC.")
    parser.add_argument("-i", "--input", help="Experimental Result", required=True)
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
