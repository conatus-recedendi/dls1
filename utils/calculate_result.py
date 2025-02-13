from argparse import ArgumentParser
import sys, os


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


parser = ArgumentParser()

parser.add_argument(
    "-o",
    "--output",
    type=str,
    nargs="+",
)

args = parser.parse_args()

results = get_multi_result(args.output, "test_acc.txt")


print("mean:", mean(results))

print("std:", std(results))
