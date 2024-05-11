# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import os
import re

# import openai
pattern = re.compile(r"^\d.*\.txt$")

done = []
qwithnum = []
questions = []
# ques_dir = 'gate_xec_2019'
ques_dir = os.environ['ques_dir']
temp = float(os.environ['temperature'])

question_path = f'/home/civil/phd/cez198233/llama/raw_qa/{ques_dir}/'

allq = os.listdir(question_path)
for q in allq:
    if pattern.match(q):
        qwithnum.append(q)
        
for q in qwithnum:
    qnew = f'llama3_70b_temp_{temp}'+q
    if qnew in allq:
        done.append(q)
    else:
        questions.append(q)

print(len(qwithnum), len(done), len(questions))
print('Starting in ',question_path)
print('--------------------------------------------------------')
qidlist = []
qlist = []
dialogs = []
print(len(qwithnum), len(done), len(questions))
print('Starting in ',question_path)
print('--------------------------------------------------------')
for qid, question in enumerate(questions):

        f = open(os.path.join(question_path,f'{questions[qid]}'),'r')
        question = ' '.join(f.readlines()).replace('\n', ' ')
        f.close()
        
        messages=[{"role": "system", "content": "Solve the following question with highly detailed step by step explanation. Write the correct answer inside a dictionary at the end in the following format. The key 'answer' has a list which can be filled by all correct options or by a number as required while answering  the question. For example for question with correct answer as option (a), return {'answer':[a]} at the end of solution. For question with multiple options'a,c' as answers, return {'answer':[a,c]}. And for question with numerical values as answer (say 1.33), return {'answer':[1.33]}"},
                 {"role": "user", "content": question}]
        dialogs.append(messages)
        qidlist.append(qid)

# print(questions), print(dialogs) 
# questions =  questions[:4]
# dialogs = dialogs[:4]

from typing import Optional

import fire

from llama import Llama


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 1.,
    top_p: float = temp,
    max_seq_len: int = 8192,
    max_batch_size: int = 2,
    max_gen_len: Optional[int] = None,
):
    
    max_batch_size = len(dialogs)

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,)

    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
        
    outputs = []

    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print("\n==================================\n")
        
        outputs.append(result['generation']['content'])
        
    for qid, output_string in enumerate(outputs): 
        f = open(os.path.join(question_path,f'llama3_70b_temp_{temp}_{questions[qid]}'),'w')# f'llama3_8b_temp_{temp}'
        f.write(output_string)
        f.close()
        
        
if __name__ == "__main__":
    fire.Fire(main)
