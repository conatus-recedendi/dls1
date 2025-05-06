import os, sys
import json
import wandb

sys.path.append("../../")
import numpy as np

from dataset.mnist import load_mnist

from common.trainer import Trainer
from common.optimizer import SGD, Momentum, AdaGrad, Adam
from common.deep_conv_net import DeepConvNet
from common.utils import to_gpu
from common import config

from common.nlp import preprocess, create_context_target, convert_one_hot
from common.cbow import CBOW

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

os.environ["WANDB_DISABLED"] = "true"


### parameter ###

window_size = 2
### load dataset ###

text = "You say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text)


vocab_size = len(word_to_id)
contexts, target = create_context_target(corpus, window_size)


print("before", contexts)
target = convert_one_hot(target, len(word_to_id))

contexts = convert_one_hot(contexts, len(word_to_id))
print("after", contexts)

x_train = contexts
t_train = target

x_test = contexts
t_test = target

# (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# config.GPU = True

if config.GPU:
    x_train = to_gpu(x_train)
    t_train = to_gpu(t_train)
    x_test = to_gpu(x_test)
    t_test = to_gpu(t_test)


def run():

    wandb.init(
        name="cbow_small",
    )

    np.random.seed(wandb.config.seed)
    # 각 실험의 고유한 키 생성
    output_name = "output/run-" + wandb.run.id

    # 폴더가 없으면 생성
    if not os.path.exists(output_name):
        os.makedirs(output_name)
    print(wandb.config.model_params)
    model = CBOW(
        vocab_size,
        hidden_size=wandb.config.model_params["hidden_size"],
        window_size=wandb.config.model_params["window_size"],
    )

    optimizer = {
        "SGD": SGD(lr=wandb.config.learning_rate),
        "Momentum": Momentum(lr=wandb.config.learning_rate),
        "AdaGrad": AdaGrad(lr=wandb.config.learning_rate),
        "Adam": Adam(lr=wandb.config.learning_rate),
    }[wandb.config.gradient_descent]

    print("x_train, t_train", x_train, t_train)
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

    with open(f"{output_name}/train_loss.txt", "w") as f:
        for loss in trainer.loss_history:
            f.write(str(loss) + "\n")

    with open(f"{output_name}/train_acc.txt", "w") as f:
        for acc in trainer.train_acc_history:
            f.write(str(acc) + "\n")

    with open(f"{output_name}/test_acc.txt", "w") as f:
        for acc in trainer.test_acc_history:
            f.write(str(acc) + "\n")

    mem_usage = model.memory_usage()
    with open(f"{output_name}/memory_usage.txt", "w") as f:
        for layer, usage in mem_usage.items():
            f.write(f"{layer}: {json.dumps(usage)}\n")
    # 각 layer의 실행 시간 확인
    times = model.training_time()
    with open(f"{output_name}/training_time.txt", "w") as f:
        for layer, timing in times.items():
            f.write(f"{layer}: {json.dumps(timing)}\n")


wandb_sweep_config = {
    "name": "cbow_small",
    "method": "grid",
    "metric": {"name": "test_acc", "goal": "maximize"},
    "parameters": {
        "seed": {"value": [1000]},
        "gradient_descent": {"value": "Adam"},
        "learning_rate": {"value": 0.001},
        "epochs": {"value": 1000},
        "batch_size": {"value": 3},
        "model": {"value": "CBOW_small"},
        "model_params": {"value": {"hidden_size": 5, "window_size": window_size}},
        # "batch_norm": {"value": False},
        # "weight_decay_lambda": {"value": 0},
        # "dataset": {"value": ""},
        # "activation": {"value": "relu"},
        # "weight_init_std": {"value": "he"},
        # "dropout": {"value": 0.15},
    },
}

sweep_id = wandb.sweep(sweep=wandb_sweep_config, project="DILab - scratch 1")

wandb.agent(sweep_id, function=run)
