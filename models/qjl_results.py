"""You can use this code 2 merge and report the QJL results"""
import json
import os

# cd /XXX/output/longbench

def process_jsonl_files():
    for filename in os.listdir(''):
        unique_objects = {}  
        total_score = 0.0
        count = 0

        if filename.endswith('.jsonl'):
            with open(filename, 'r', encoding='utf-8') as file:
                for line in file:
                    obj = json.loads(line.strip())  
                    _id = obj['_id']               
                    score = obj['score']          

                    if _id not in unique_objects:  
                        unique_objects[_id] = score
                        total_score += score
                        count += 1

        dataset = filename.split('.jsonl')[0]
        if count != 0:
            print(f'{dataset}: {(total_score / count):.4f}')


if __name__ == "__main__":
    process_jsonl_files()



