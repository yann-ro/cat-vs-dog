import gc
import os
import time

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt


class Trainer:
    """_summary_"""

    def __init__(
        self,
        model,
        dataloader,
        lr=1e-4,
        device="cpu",
        criterion=nn.BCELoss(),
        optimizer=optim.Adam,
        scheduler=lambda x: optim.lr_scheduler.StepLR(x, step_size=5, gamma=.5),
        root="",
    ):
        """_summary_

        Args:
            model (_type_): _description_
            device (_type_): _description_
            root (str, optional): _description_. Defaults to ''.
        """
        # env
        self.device = device
        self.root = os.path.abspath(root)
        self.dataloader = dataloader
        self.current_epoch = 0

        # hyperparameters
        self.lr = lr
        self.criterion = criterion
        self.criterion.to(self.device)

        # model
        self.net = model
        self.net.to(self.device)

        self.optimizer = optimizer(self.net.parameters(), lr=self.lr)
        self.lr_scheduler = scheduler(self.optimizer)

        # history
        self.history = {
            "epoch": [],
            "train_accuracy": [],
            "train_loss": [],
            "valid_accuracy": [],
            "valid_loss": [],
        }
        self.test_accuracy = 0

    @staticmethod
    def accuracy(predictions, trues):
        """_summary_

        Args:
            predictions (_type_): _description_
            trues (_type_): _description_

        Returns:
            _type_: _description_
        """
        predictions = [
            1 if predictions[i] >= 0.5 else 0 for i in range(len(predictions))
        ]
        acc = [1 if predictions[i] == trues[i] else 0 for i in range(len(predictions))]
        acc = np.sum(acc) / len(predictions)

        return acc * 100

    @staticmethod
    def format_label(label):
        """_summary_

        Args:
            label (_type_): _description_

        Returns:
            _type_: _description_
        """
        label = label.split(".")[-1]
        label = label.replace("_", " ")
        label = label.title()
        return label.replace(" ", "")

    def train(self, iterator: DataLoader):
        """_summary_

        Args:
            iterator (DataLoader): _description_

        Returns:
            _type_: _description_
        """

        # Local Parameters
        epoch_loss = []
        epoch_acc = []
        start_time = time.time()

        # Iterating over data loader
        for images, labels in tqdm(iterator):
            # Loading images and labels to device
            images = images.to(self.device)
            labels = labels.to(self.device)
            labels = labels.reshape(
                (labels.shape[0], 1)
            )  # [N, 1] - to match with preds shape

            # Reseting Gradients
            self.optimizer.zero_grad()

            # Forward
            preds = self.net(images)

            # Calculating Loss
            _loss = self.criterion(preds, labels)
            loss = _loss.item()
            epoch_loss.append(loss)

            # Calculating Accuracy
            acc = self.accuracy(preds, labels)
            epoch_acc.append(acc)

            # Backward
            _loss.backward()
            self.optimizer.step()

            del images
            del labels

        # Overall Epoch Results
        end_time = time.time()
        total_time = end_time - start_time

        # Acc and Loss
        epoch_loss = np.mean(epoch_loss)
        epoch_acc = np.mean(epoch_acc)

        return epoch_loss, epoch_acc, total_time

    def evaluate(self, iterator, best_val_acc, mode="test"):
        """_summary_

        Args:
            iterator (_type_): _description_
            best_val_acc (_type_): _description_
            mode (str, optional): _description_. Defaults to "test".

        Returns:
            _type_: _description_
        """

        # Local Parameters
        epoch_loss = []
        epoch_acc = []
        start_time = time.time()

        # Iterating over data loader
        for images, labels in iterator:
            # Loading images and labels to device
            images = images.to(self.device)
            labels = labels.to(self.device)
            labels = labels.reshape((labels.shape[0], 1))

            # forward
            preds = self.net(images)

            # loss
            _loss = self.criterion(preds, labels)
            loss = _loss.item()
            epoch_loss.append(loss)

            # accuracy
            acc = self.accuracy(preds, labels)
            epoch_acc.append(acc)

            del images
            del labels

        end_time = time.time()
        total_time = end_time - start_time

        # Acc and Loss
        epoch_loss = np.mean(epoch_loss)
        epoch_acc = np.mean(epoch_acc)

        # Saving best model
        if epoch_acc > best_val_acc and mode == "val":
            best_val_acc = epoch_acc
            torch.save(
                self.net.state_dict(),
                os.path.join(self.root, "best_models/dog_cat_resnet50_best.pth"),
            )

        return epoch_loss, epoch_acc, total_time, best_val_acc

    def train_data(self, num_epochs):
        """_summary_"""
        best_val_acc = 0
        torch.cuda.empty_cache()
        try:
            for _ in range(num_epochs):
                
                print(f"Epoch: {self.current_epoch}")
                
                # train
                loss, acc, _time = self.train(self.dataloader.train_iterator)
                
                print(f"Train - Loss : {loss:.4f} Acc : {acc:.4f} Time: {_time:.4f}")
                self.history["train_accuracy"].append(acc / 100)
                self.history["train_loss"].append(loss)

                # eval
                loss, acc, _time, best_val_acc = self.evaluate(
                    self.dataloader.valid_iterator, best_val_acc=best_val_acc, mode="val"
                )
                
                print(f"Valid - Loss : {loss:.4f} Acc : {acc:.4f} Time: {_time:.4f}\n")
                self.history["valid_accuracy"].append(acc / 100)
                self.history["valid_loss"].append(loss)
                
                self.current_epoch += 1
                self.history["epoch"].append(self.current_epoch)

        except KeyboardInterrupt:
            pass

        # test
        loss, acc, _time, best_val_acc = self.evaluate(
            self.dataloader.test_iterator, best_val_acc=best_val_acc
        )
        print(f"Test  - Loss : {loss:.4f} Acc : {acc:.4f} Time: {_time:.4f}")
        self.test_accuracy = acc

        # save model
        model_path = os.path.join(self.root, "best_models/dog_cat_net.pth")
        torch.save(self.net.state_dict(), model_path)

    @staticmethod
    def clean_gpu_memory():
        """_summary_"""
        gc.collect()
        torch.cuda.empty_cache()

    def plot_history(self, title="", saving_path=""):
        """_summary_"""
        n_epochs = len(self.history["epoch"])

        plt.figure(figsize=(24, 6))
        plt.subplot(1, 2, 1)
        plt.plot(self.history["epoch"], self.history["train_accuracy"][:n_epochs], label="train")
        plt.plot(
            self.history["epoch"], self.history["valid_accuracy"][:n_epochs], label="validation"
        )
        plt.ylim(0.5, 1.1)
        plt.grid()
        plt.legend()
        plt.title("Accuracy")

        plt.subplot(1, 2, 2)
        plt.plot(self.history["epoch"], self.history["train_loss"][:n_epochs], label="train")
        plt.plot(self.history["epoch"], self.history["valid_loss"][:n_epochs], label="validation")
        plt.grid()
        plt.legend()
        plt.title("Loss")

        plt.suptitle(f"{title}\nTest accuracy {self.test_accuracy:.4f}")

        if saving_path:
            plt.savefig(os.path.join(saving_path, f"figures/{title}.png"))
        plt.show()
        
