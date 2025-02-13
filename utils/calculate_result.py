from argparse import ArgumentParser
import sys, os
import re


# std
def mean(values):
    return sum(values) / len(values)


def std(values):
    m = mean(values)
    return (sum([(v - m) ** 2 for v in values]) / len(values)) ** 0.5


def get_result(output_folder, filename):
    # output_folder/test_acc.txt

    with open(os.path.join(output_folder, filename), "r") as f:
        lines = f.readlines()

        values = []
        for line in lines:
            values.append(float(line))
    return values[-1]


def get_multi_result(output_folders, filename):
    results = []

    for folder in output_folders:
        results.append(get_result(folder, filename))

    return results


def get_result_time(output_folder, filename):
    # output_folder/test_acc.txt

    with open(os.path.join(output_folder, filename), "r") as f:
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

    return values


def get_multi_result_time(output_folders, filename):
    results = []

    for folder in output_folders:
        results.append(get_result_time(folder, filename))

    return results


parser = ArgumentParser()

parser.add_argument(
    "-o",
    "--output",
    type=str,
    nargs="+",
)

args = parser.parse_args()

results = get_multi_result(args.output, "test_acc.txt")

times = get_multi_result_time(args.output, "training_time.txt")

print(
    "training_time mean:",
    mean([sum([x["forward"] + x["backward"] for x in time]) for time in times]),
)


print(
    "training_time std:",
    std([sum([x["forward"] + x["backward"] for x in time]) for time in times]),
)

print("mean:", mean(results))

print("std:", std(results))
