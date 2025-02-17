import os, sys


sys.path.append("../../")
import numpy as np
import matplotlib.pyplot as plt

from dataset.mnist import load_mnist
from common.deep_conv_net import DeepConvNet


_, (x, t) = load_mnist(flatten=False)

batch_size = 100

misclassified = []
misclassified_label = []
misclassified_real_label = []

model = DeepConvNet()

model.load_params(
    "../code/deep_cnn/output/output_seed=1000_id=rhwlgwgd/output_seed=1000_epoch=1.pkl"
)

for i in range(int(x.shape[0] / batch_size)):
    print(i)
    tx = x[i * batch_size : (i + 1) * batch_size]
    tt = t[i * batch_size : (i + 1) * batch_size]
    y = model.predict(tx, train_flag=False)
    y = np.argmax(y, axis=1)

    misclassified_case = np.where(y != tt)
    misclassified.append(tx[misclassified_case])
    misclassified_label.append(tt[misclassified_case])
    misclassified_real_label.append(y[misclassified_case])


misclassified = np.vstack(misclassified).reshape(-1, 28, 28)
misclassified_label = np.concatenate(misclassified_label)
misclassified_real_label = np.concatenate(misclassified_real_label)

print(misclassified.shape)
print(misclassified_label.shape)
print(misclassified_real_label.shape)

fig = plt.figure()

fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
current_view = 1
with open("misclassified.txt", "wb") as f:
    # plt 28x
    for val in misclassified:

        ax = fig.add_subplot(4, 5, current_view, xticks=[], yticks=[])
        ax.imshow(val.reshape(28, 28), cmap=plt.cm.binary, interpolation="nearest")
        ax.text(1, 26, f"{misclassified_label[current_view - 1]}", color="red")
        ax.text(22, 26, f"{misclassified_real_label[current_view - 1]}", color="blue")

        current_view += 1

        if current_view > 4 * 5:
            break

plt.show()
plt.savefig("misclassified.png")


with open("misclassified_answer.txt", "w") as f:
    for idx, val in enumerate(misclassified_label):
        f.write(f"{misclassified_label[idx]} {misclassified_real_label[idx]}\n")
