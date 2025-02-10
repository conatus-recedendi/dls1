import pickle
import os
import wandb


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        x_train,
        t_train,
        x_test,
        t_test,
        batch_size,
        epochs,
        seed,
        output_name,
    ):
        self.model = model
        self.optimizer = optimizer
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed
        self.output_name = output_name

        self.loss_history = []
        self.train_acc_history = []
        self.test_acc_history = []

    def train_one_batch(self, inputs, labels):
        grads = self.model.gradient(inputs, labels)
        self.optimizer.update(self.model.params, grads)

        loss = self.model.loss(inputs, labels)
        return loss

    def train(self):

        for epoch in range(self.epochs):
            len_inputs = len(self.x_train)
            for i in range(0, len_inputs, self.batch_size):
                inputs_batch = self.x_train[i : i + self.batch_size]
                labels_batch = self.t_train[i : i + self.batch_size]

                loss = self.train_one_batch(inputs_batch, labels_batch)

                # TODO: record loss per iter
                self.loss_history.append(loss)

            acc_train = self.model.accuracy(self.x_train, self.t_train)
            acc_test = self.model.accuracy(self.x_test, self.t_test)
            self.train_acc_history.append(acc_train)
            self.test_acc_history.append(acc_test)
            wandb.log(
                {
                    "train_loss": loss,
                    "train_acc": acc_train,
                    "test_acc": acc_test,
                }
            )

            # if last saved model exist, remove it,
            # then save current model
            # if epoch > 0:
            last_saved_model = (
                f"{self.output_name}/output_seed={self.seed}_epoch={epoch-1}.pkl"
            )
            if os.path.exists(last_saved_model) and epoch > 0:
                os.remove(last_saved_model)
            pkl_file = f"{self.output_name}/output_seed={self.seed}_epoch={epoch}.pkl"
            with open(pkl_file, "wb") as f:
                pickle.dump(self.model.params, f)
