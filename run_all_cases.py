import argparse
import subprocess
import sys
from pathlib import Path


def parse_cases(cases_str: str):
    """
    将类似 "0-5,7,9-10" 的字符串解析成有序且去重的 case 列表。
    """
    result = []
    for part in cases_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            start = int(start)
            end = int(end)
            if start > end:
                start, end = end, start
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))

    # 去重并保持顺序
    seen = set()
    ordered = []
    for c in result:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def main():
    """
    顺序自动执行 case0.py ~ case5.py，同时把你在命令行里写的参数透传给各个 case。

    使用示例（在项目根目录）：

        # 按默认 config.py 跑 case0-5
        # （等价于依次 python case0.py ... python case5.py）
        python run_all_cases.py

        # 只跑 case0,1,2
        python run_all_cases.py --cases 0-2

        # 跑 case1,3,5，并修改实验参数
        python run_all_cases.py --cases 1,3,5 --num_rounds 20 --num_unlearn_rounds 2 --lr 0.01

    说明：
        - 本脚本自己的参数只有一个：--cases
        - 其它所有参数（例如 --num_rounds、--lr、--poisoned_percent、--is_onboarding 等）
          都会原样传给各个 caseX.py，由 config.py 里的 argparse 解析。
    """

    parser = argparse.ArgumentParser(description="顺序运行 case0.py ~ case5.py 的小工具")
    parser.add_argument(
        "--cases",
        type=str,
        default="0-5",
        help='要运行的 case 列表，格式如 "0-5" 或 "0-2,4,5"（默认 0-5 全部）',
    )

    # 只解析本脚本关心的参数，其余透传给 case 脚本
    args, extra_args = parser.parse_known_args()

    cases = parse_cases(args.cases)
    root = Path(__file__).resolve().parent
    python_exe = sys.executable  # 使用当前环境的 python（例如 venv）

    print(f"将依次运行 case: {cases}")
    print(f"附加参数（传递给各个 caseX.py）: {extra_args}")
    print("-" * 60)

    for c in cases:
        case_file = root / f"case{c}.py"
        if not case_file.exists():
            print(f"[跳过] 未找到文件: {case_file}")
            continue

        cmd = [python_exe, str(case_file), *extra_args]
        print(f"[运行] {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[错误] 运行 case{c}.py 失败，退出码 {e.returncode}")
            # 出错就直接停止，避免后续 case 在依赖缺失时继续跑
            sys.exit(e.returncode)

    print("-" * 60)
    print("所有指定的 case 已运行完成。")


if __name__ == "__main__":
    main()


