from argparse import ArgumentParser
import re

import matplotlib.pyplot as plt

# example
# --input=../code/batch_normalization/output/training_time.txt
# --output=./output/visualize_training_time.png


def visualize_training_time(input_path, output_path, max_rank=10):
    with open(input_path, "r") as f:
        lines = f.readlines()

        values = []
        for line in lines:
            label = re.search(r"\(\w+\):", line)
            if label:
                label = label[0][1:-2]
                idx = 0
                # 페인트공 problem
                while True:
                    label_with_idx = label + str(idx)
                    if label_with_idx not in [x["label"] for x in values]:
                        break
                    idx += 1
                values.append(
                    {
                        "label": label + str(idx),
                    }
                )

                forward_time = re.search(r'"forward": (\d+\.?\d*)', line)
                backward_time = re.search(r'"backward": (\d+\.?\d*)', line)

                if forward_time:
                    ft = forward_time[0].split(":")[1][1:-1]

                    if len(ft) == 0:
                        ft = 0.0
                    ft = float(ft)

                    values[-1].update({"forward": ft})
                if backward_time:
                    bt = backward_time[0].split(":")[1][1:-1]

                    if len(bt) == 0:
                        bt = 0.0
                    bt = float(bt)
                    # backward_times.append(bt)
                    values[-1].update({"backward": bt})

    plt.subplot(2, 1, 1)

    max_rank = min(max_rank, len(values))

    values_by_forward = sorted(values, key=lambda x: x["forward"], reverse=True)
    remaining_forward = sum([x["forward"] for x in values_by_forward[max_rank:]])
    sum_forward = sum([x["forward"] for x in values_by_forward])

    values_by_backward = sorted(values, key=lambda x: x["backward"], reverse=True)
    remaining_backward = sum([x["backward"] for x in values_by_backward[max_rank:]])
    sum_backward = sum([x["backward"] for x in values_by_backward])
    total_time = sum_forward + sum_backward

    plt.pie(
        [x["forward"] for x in values_by_forward[:max_rank]] + [remaining_forward],
        labels=[x["label"] for x in values_by_forward[:max_rank]] + ["Other"],
        autopct=(lambda pct: f"{pct/100*total_time:.2f}s({pct:.1f}%)"),
        textprops={"fontsize": 8},
        radius=2,
    )
    plt.title("Forward Time")

    plt.subplot(2, 1, 2)
    plt.subplots_adjust(hspace=1)
    plt.pie(
        [x["backward"] for x in values_by_backward[:max_rank]] + [remaining_backward],
        labels=[x["label"] for x in values_by_backward[:max_rank]] + ["Other"],
        autopct=(lambda pct: f"{pct/100*total_time:.2f}s({pct:.1f}%)"),
        textprops={"fontsize": 8},
        radius=2,
    )
    plt.title("Backward Time")

    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()


parser = ArgumentParser()

parser.add_argument("--input", type=str, required=True)
parser.add_argument("--output", type=str, required=False)
parser.add_argument("--max_rank", type=int, default=10)

args = parser.parse_args()

visualize_training_time(args.input, args.output, args.max_rank)
