import os, sys
import json
import wandb

sys.path.append("../../")
import numpy as np

from dataset.mnist import load_mnist

from common.multi_layer_net import MultiLayerNet
from common.trainer import Trainer
from common.optimizer import SGD, Momentum, AdaGrad, Adam


### training ###

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)


def run():

    wandb.init(
        name="gradient_descent",
    )

    np.random.seed(wandb.config.seed)
    # 각 실험의 고유한 키 생성
    output_name = "output/output_seed=" + str(wandb.config.seed) + "_id=" + wandb.run.id

    # 폴더가 없으면 생성
    if not os.path.exists(output_name):
        os.makedirs(output_name)

    model = MultiLayerNet(
        input_size=784,
        hidden_size_list=[100],
        output_size=10,
        weight_init_std=0.01,
    )
    optimizer = {
        "SGD": SGD(),
        "Momentum": Momentum(),
        "AdaGrad": AdaGrad(),
        "Adam": Adam(),
    }[wandb.config.gradient_descent]

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


train_loss_history = []


wandb_sweep_config = {
    "method": "grid",
    "name": "gradient_descent",
    "metric": {"name": "test_acc", "goal": "maximize"},
    "parameters": {
        "seed": {"values": [1000, 2000, 3000, 4000, 5000]},
        "gradient_descent": {"values": ["SGD", "Momentum", "AdaGrad", "Adam"]},
        "learning_rate": {"value": 0.01},
        "epochs": {"value": 100},
        "batch_size": {"value": 100},
        "model": {"value": "MultiLayerNet"},
        "batch_norm": {"value": False},
        "weight_decay_lambda": {"value": 0},
        "dataset": {"value": "mnist"},
        "activation": {"value": "relu"},
        "dropout": {"value": 0},
    },
}

sweep_id = wandb.sweep(sweep=wandb_sweep_config, project="DILab - scratch 1")

wandb.agent(sweep_id, function=run)
