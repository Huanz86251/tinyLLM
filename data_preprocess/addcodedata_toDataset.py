# Data source: HumanEval-XL (Chinese, Python split)
# Repo: https://github.com/floatai/HumanEval-XL  | HF dataset: floatai/HumanEval-XL
# License: Data under Apache-2.0; repo/code under MIT. Used for research; original authors credited.

import json
from tqdm import tqdm
import os
def getquestion_id(datapath):
    question_id=-1
    with open(datapath, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            tempquestion_id = data["question"]
            if tempquestion_id > question_id:
                question_id = tempquestion_id
    return question_id
def insert_code(codepath,datasetpath):
    question_id=getquestion_id(datasetpath)
    with open(datasetpath, "a", encoding="utf-8") as write_file:
        with open(codepath, "r", encoding="utf-8") as f:
            index=0
            for line in tqdm(f):
                question_id +=1
                data = json.loads(line)
                uniqueKey=f"code{index}"
                index+=1
                desc=data["description"]
                code=data["canonical_solution"]
                text=f"{desc}\n代码实现如下：\n```python\n{code}\n```"
                newsample={"text":text,"question": question_id, "uniqueKey":uniqueKey}
                write_file.write(json.dumps(newsample,ensure_ascii=False)+"\n")

def main():
    codepath = r"D:\WuDaoCorporaText-2.0-open\Chinese.jsonl"
    datasetpath = "../data/wudao_filtered_5gb.jsonl"
    insert_code(codepath,datasetpath)

if __name__=="__main__":
    main()