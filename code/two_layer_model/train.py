import os, sys
import json
import wandb

sys.path.append("../../")
import numpy as np

from dataset.mnist import load_mnist

from common.multi_layer_net import MultiLayerNet
from common.trainer import Trainer
from common.optimizer import SGD


### training ###


def run():
    (x_train, t_train), (x_test, t_test) = load_mnist(
        normalize=True, one_hot_label=True
    )

    wandb.init(
        name="two_layer_model re-implementation",
    )

    wandb.config.update({"baseline": True}, allow_val_change=True)

    np.random.seed(wandb.config.seed)
    # 각 실험의 고유한 키 생성
    output_name = "output/output_seed=" + str(wandb.config.seed) + "_id=" + wandb.run.id

    # 폴더가 없으면 생성
    if not os.path.exists(output_name):
        os.makedirs(output_name)

    x_train = x_train[: wandb.config.training_size]
    t_train = t_train[: wandb.config.training_size]

    model = MultiLayerNet(
        input_size=784, hidden_size_list=[100], output_size=10, weight_init_std=0.01
    )
    optimizer = SGD(lr=wandb.config.learning_rate)

    trainer = Trainer(
        model,
        optimizer=optimizer,
        x_train=x_train,
        t_train=t_train,
        x_test=x_test,
        t_test=t_test,
        batch_size=wandb.config.batch_size,
        epochs=wandb.config.epochs,
        seed=wandb.config.seed,
        output_name=output_name,
    )

    trainer.train()

    mem_usage = model.memory_usage()

    with open(f"{output_name}/train_loss.txt", "w") as f:
        for loss in trainer.loss_history:
            f.write(str(loss) + "\n")

    with open(f"{output_name}/train_acc.txt", "w") as f:
        for acc in trainer.train_acc_history:
            f.write(str(acc) + "\n")

    with open(f"{output_name}/test_acc.txt", "w") as f:
        for acc in trainer.test_acc_history:
            f.write(str(acc) + "\n")

    with open(f"{output_name}/memory_usage.txt", "w") as f:
        for layer, usage in mem_usage.items():
            print(layer, usage)
            print(json.dumps(usage))
            f.write(f"{layer}: {json.dumps(usage)}\n")
    # 각 layer의 실행 시간 확인
    times = model.training_time()

    with open(f"{output_name}/training_time.txt", "w") as f:
        for layer, timing in times.items():
            print(layer, timing)
            print(json.dumps(timing))
            f.write(f"{layer}: {json.dumps(timing)}\n")

    # save pkl file

    # matrix: check diff between numerical difference and back propagation


train_loss_history = []


# seeds = [1000, 2000, 3000, 4000, 5000]

wandb_sweep_config = {
    "method": "grid",
    "name": "two_layer_model",
    "metric": {"name": "test_acc", "goal": "maximize"},
    "parameters": {
        "seed": {"value": 1000},
        "learning_rate": {"value": 0.1},
        "epochs": {"value": 100},
        "batch_size": {"value": 100},
        "training_size": {"value": 60000},
    },
}

sweep_id = wandb.sweep(sweep=wandb_sweep_config, project="MLP, CNN")

wandb.agent(sweep_id, function=run)
