from trainer import ModelTrainer
from model_1 import DropoutModel8x8

import sys

import torch
import torch.nn as nn
import torch.nn.functional as func

""" test code for model 1 """
def trainNew():
    game_name = "Super Mario Bros"
    device = "cuda"
    learning_rate = 0.0001
    batch_size = 10
    n_epochs = 1

    model = DropoutModel8x8(32).to(device)
    trainer = ModelTrainer(learning_rate, batch_size, n_epochs)
    trainer.loadAndPrepare(game_name)
    print(trainer.processor.data.shape)

    loss_func = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    trained_model = trainer.train(game_name, model, loss_func, optimizer, device, is_save = True)


def continueTrain():
    path = "saved_model/model_05-21_16-01"
    game_name = "Super Mario Bros"
    batch_size = 10
    n_epochs = 1
    device = "cuda"
    learning_rate = 0.0001

    trainer = ModelTrainer(learning_rate, batch_size, n_epochs)
    trainer.loadAndPrepare(game_name)

    model = DropoutModel8x8(32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.MSELoss().to(device)

    model, optimizer, info = trainer.loadModel(model, optimizer, path)

    trainer.batch_size = info["batch_size"]
    trained_model = trainer.train(info["game_name"], model, loss_func, optimizer, info["device"], 
        is_save = True)


def usage():
    print("Invalid arguments, usage: python .\\model_1_main.py [-n|-c]")
    print("-n: train a new model")
    print("-c: continue to train an existing model")
    return


if __name__ == "__main__":

    if len(sys.argv) != 2:
        usage()
    elif sys.argv[1] == "-c":
        continueTrain()
    elif sys.argv[1] == "-n":
        trainNew()
    else:
        usage()
