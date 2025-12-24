import json
import matplotlib.pyplot as plt
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Plot Training and Evaluation Loss from JSONL logs")
    # 默认路径为您代码中设置的路径
    parser.add_argument("--log_file", type=str, default="./models/LLaDA-Table-SFT-WithEval/trainer_metrics.jsonl", help="Path to the trainer_metrics.jsonl file")
    parser.add_argument("--output_file", type=str, default="./models/LLaDA-Table-SFT-WithEval/loss_curve.png", help="Path to save the output image")
    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.log_file):
        print(f"Error: Log file not found at {args.log_file}")
        print("Please check the path or run this script in the same directory as the logs.")
        return

    train_data = {} # key: step, value: loss
    eval_data = {}  # key: step, value: eval_loss

    print(f"Reading logs from {args.log_file}...")
    
    with open(args.log_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                step = record.get('step')
                
                # 区分训练日志和评估日志
                # 训练日志通常含 'loss' 但不含 'eval_loss'
                # 评估日志含 'eval_loss'
                if 'eval_loss' in record:
                    eval_data[step] = record['eval_loss']
                elif 'loss' in record:
                    train_data[step] = record['loss']
            except json.JSONDecodeError:
                continue

    # 排序数据
    train_steps = sorted(train_data.keys())
    train_losses = [train_data[k] for k in train_steps]

    eval_steps = sorted(eval_data.keys())
    eval_losses = [eval_data[k] for k in eval_steps]

    if not train_steps and not eval_steps:
        print("No valid loss data found in the log file.")
        return

    # 绘图
    plt.figure(figsize=(12, 6))

    if train_steps:
        plt.plot(train_steps, train_losses, label='Training Loss', alpha=0.8, linewidth=1.5)
    
    if eval_steps:
        plt.plot(eval_steps, eval_losses, label='Evaluation Loss', color='red', marker='o', linestyle='--', linewidth=2, markersize=5)

    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training & Evaluation Loss Curve', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)

    # 保存图片
    plt.savefig(args.output_file, dpi=300, bbox_inches='tight')
    print(f"Success! Loss curve saved to: {args.output_file}")
    # plt.show() # 如果在本地带界面的环境运行，可以取消注释这行直接显示

if __name__ == "__main__":
    main()