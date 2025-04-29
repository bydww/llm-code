# verl强化学习实践

## 简介
本项目实现了一个基于 **verl** 强化学习框架的训练示例。verl 是字节跳动与香港大学开源的强化学习框架，有效解决了大模型强化学习系统灵活性和效率不足的问题。下面展示了如何基于Qwen2-7B-Instruct模型在GSM8K数据集上进行强化学习训练，读者需要首先安装verl框架。

## 使用说明
- 首先需要下载 gsm8k 数据集，下载代码在 `dataset.py` 中。
```
python dataset.py
```
- 运行 `verl_train_script.sh` 脚本，将自动加载模型和数据集，进行强化学习训练。
```
bash verl_train_script.sh
```