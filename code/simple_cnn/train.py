import os, sys
import json
import wandb

sys.path.append("../../")
import numpy as np

from dataset.mnist import load_mnist

from common.multi_layer_net import MultiLayerNet
from common.trainer import Trainer
from common.optimizer import SGD, Momentum, AdaGrad, Adam
from common.simple_conv_net import SimpleConvNet


### training ###

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

x_train = x_train.reshape(-1, 1, 28, 28)
x_test = x_test.reshape(-1, 1, 28, 28)


def run():

    wandb.init(
        name="simple_cnn",
    )

    np.random.seed(wandb.config.seed)
    # 각 실험의 고유한 키 생성
    output_name = "output/output_seed=" + str(wandb.config.seed) + "_id=" + wandb.run.id

    # 폴더가 없으면 생성
    if not os.path.exists(output_name):
        os.makedirs(output_name)

    model = SimpleConvNet(
        input_dim=(1, 28, 28),
        conv_param={
            "filter_num": 30,
            "filter_size": 5,
            "pad": 0,
            "stride": 1,
        },
        hidden_size=100,
        output_size=10,
        weight_init_std=wandb.config.weight_init_std,
    )

    optimizer = {
        "SGD": SGD(lr=wandb.config.learning_rate),
        "Momentum": Momentum(lr=wandb.config.learning_rate),
        "AdaGrad": AdaGrad(lr=wandb.config.learning_rate),
        "Adam": Adam(lr=wandb.config.learning_rate),
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

    # print(x_test.shape)
    ### Custom Log
    w1 = model.params["W1"]
    image_w1_list = []
    for i in range(30):
        image_w1_list.append(wandb.Image(w1[i][0].reshape(5, 5)))
    wandb.log({f"conv1_filter_before": image_w1_list})

    trainer.train()

    w1 = model.params["W1"]
    image_w1_list = []
    for i in range(30):
        image_w1_list.append(wandb.Image(w1[i][0].reshape(5, 5)))
    wandb.log({f"conv1_filter_after": image_w1_list})

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


wandb_sweep_config = {
    "name": "simple_cnn",
    "method": "grid",
    "metric": {"name": "test_acc", "goal": "maximize"},
    "parameters": {
        "seed": {"value": 1000},
        "gradient_descent": {"value": "SGD"},
        "learning_rate": {"value": 0.01},
        "epochs": {"value": 64},
        "batch_size": {"value": 100},
        "model": {"value": "SimpleConvNet"},
        "batch_norm": {"value": False},
        "weight_decay_lambda": {"value": 0},
        "dataset": {"value": "mnist"},
        "activation": {"value": "relu"},
        "weight_init_std": {"value": 0.01},
        "dropout": {"value": 0},
    },
}

sweep_id = wandb.sweep(sweep=wandb_sweep_config, project="DILab - scratch 1")

wandb.agent(sweep_id, function=run)
