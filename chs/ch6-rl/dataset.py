import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
import pandas as pd
from datasets import load_dataset

def prepare_gsm8k(save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # 加载 gsm8k 数据集（官方 datasets 库）
    dataset = load_dataset("gsm8k", "main")

    # train 集
    train_data = dataset['train']
    train_records = [{"question": item['question'], "answer": item['answer']} for item in train_data]

    # test 集
    test_data = dataset['test']
    test_records = [{"question": item['question'], "answer": item['answer']} for item in test_data]

    # 保存成 parquet 文件
    train_df = pd.DataFrame(train_records)
    test_df = pd.DataFrame(test_records)

    train_path = os.path.join(save_dir, "train.parquet")
    test_path = os.path.join(save_dir, "test.parquet")

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print(f"Saved train set to {train_path}")
    print(f"Saved test set to {test_path}")

if __name__ == "__main__":
    home = os.path.dirname(os.path.abspath(__file__))
    save_directory = os.path.join(home, "data", "gsm8k")
    prepare_gsm8k(save_directory)
