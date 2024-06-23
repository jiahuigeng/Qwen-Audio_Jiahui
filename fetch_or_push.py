import os
import argparse
from shutil import copyfile
from pathlib import Path

# 获取用户的主目录

def operate(args):
    home_directory = Path.home()
    tgt_path = os.path.join(home_directory, ".cache","huggingface","modules","transformers_modules","Qwen","Qwen-Audio-Chat","a52fffad1964a463791eb6a10d354b6de31a069d")
    loc_path = os.path.join("local_lib")
    if not os.path.exists(loc_path):
        os.makedirs(loc_path)

    exists = os.path.exists(tgt_path)
    print(f"tgt path exsits: {exists}")

    if exists:
        files = os.listdir(tgt_path)
        files = [file for file in files if file.endswith(".py")]
        if args.op == "fetch":
            for file in files:
                copyfile(os.path.join(tgt_path,file), os.path.join(loc_path,file))

        elif args.op == "push":
            for file in files:
                copyfile(os.path.join(loc_path, file), os.path.join(tgt_path,file))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--op", default=str, choices=["fetch", "push"], help="fetch or push")
    args = parser.parse_args()

    operate(args)

