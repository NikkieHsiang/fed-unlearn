import os
import pickle
import time
from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm

import config
from utils import clients, server
from utils.dataloader import get_loaders
from utils.model import get_model
from utils.utils import get_results, save_param, update_results
from pathlib import Path
import re

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True


def find_resume_round(args):
    """扫描 case0 模型目录，返回已保存的最大 round 编号，找不到则返回 -1。"""
    model_dir = Path(f"./results/models/case0")
    if not model_dir.exists():
        return -1
    stem = (
        f"case0_{args.dataset}_C{args.num_clients}_BS{args.batch_size}"
        f"_R{args.num_rounds}_UR{args.num_unlearn_rounds}"
        f"_PR{args.num_post_training_rounds}_E{args.local_epochs}_LR{args.lr}"
    )
    pattern = re.compile(rf"^{re.escape(stem)}_round(\d+)\.pt$")
    rounds = [int(m.group(1)) for f in model_dir.iterdir() if (m := pattern.match(f.name))]
    return max(rounds) if rounds else -1


if __name__ == "__main__":
    args = config.get_args()
    train_loaders, test_loader, test_loader_poison = get_loaders(args)

    res = get_results(args)
    num_rounds = args.num_rounds

    # ── 断点续训：检测已保存的最大 round ──────────────────────────
    resume_round = find_resume_round(args)
    if resume_round >= 0:
        stem = (
            f"case0_{args.dataset}_C{args.num_clients}_BS{args.batch_size}"
            f"_R{args.num_rounds}_UR{args.num_unlearn_rounds}"
            f"_PR{args.num_post_training_rounds}_E{args.local_epochs}_LR{args.lr}"
        )
        # 补全 round 0 ~ resume_round 的 val acc（重跑 evaluate，不重新训练）
        print(f"检测到断点，从 round {resume_round + 1} 继续训练")
        print(f"正在补全 round 0–{resume_round} 的评估数据...")
        for r in range(resume_round + 1):
            ckpt = Path(f"./results/models/case0/{stem}_round{r}.pt")
            param = torch.load(ckpt, map_location=args.device)
            # val acc 可以重新 evaluate，train acc 已无法恢复，用 0 占位
            res["train"]["loss"]["avg"].append(0.0)
            res["train"]["acc"]["avg"].append(0.0)
            res = update_results(args, res, param, test_loader, test_loader_poison)
        # 加载最新 checkpoint 作为起始 global_param
        ckpt = Path(f"./results/models/case0/{stem}_round{resume_round}.pt")
        global_param = torch.load(ckpt, map_location=args.device)
        start_round = resume_round + 1
    else:
        model = get_model(args)
        global_param = model.state_dict()
        start_round = 0
    # ──────────────────────────────────────────────────────────────

    start_time = time.time()
    for round in range(start_round, num_rounds):
        print(
            "Round {}/{}: lr {} {}".format(
                round + 1, args.num_rounds, args.lr, args.out_file
            )
        )

        train_loss, test_loss = 0, 0
        train_corr, test_acc = 0, 0
        train_total = 0
        list_params = []

        chosen_clients = [i for i in range(args.num_clients)]

        for client in tqdm(chosen_clients):
            print(f"-----------client {client} starts training----------")
            tem_param, train_summ = clients.client_train(
                args,
                deepcopy(global_param),
                train_loaders[client],
                epochs=args.local_epochs,
            )

            save_param(
                args,
                param=tem_param,
                case=0,
                client=client,
                round=round,
                is_global=False,
            )

            train_loss += train_summ["loss"]
            train_corr += train_summ["correct"]
            train_total += train_summ["total"]

            list_params.append(tem_param)

        res["train"]["loss"]["avg"].append(train_loss / len(chosen_clients))
        res["train"]["acc"]["avg"].append(train_corr / train_total)

        print(
            "Train loss: {:5f} acc: {:5f}".format(
                res["train"]["loss"]["avg"][-1],
                res["train"]["acc"]["avg"][-1],
            )
        )

        # server aggregation
        global_param = server.FedAvg(list_params)

        save_param(args, param=global_param, case=0, round=round)

        res = update_results(args, res, global_param, test_loader, test_loader_poison)

    total_time = time.time() - start_time
    res["time"] = total_time
    print(f"Time {total_time}")

    with open(args.out_file, "wb") as fp:
        pickle.dump(res, fp)
