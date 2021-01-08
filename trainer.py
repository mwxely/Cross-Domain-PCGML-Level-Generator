from time import strftime
import numpy as np
import torch

from data_loader import DataLoader
from data_processor import DataProcessor

# The whole path of json file is VGLC_directory + game name + dataset_info[game name][0]
# the path of level folder is VGLC_directory + game name + dataset_info[game name][1]
VGLC_PATH = "C:/Users/jh yang/Desktop/TheVGLC/"
DATASET_INFO = {
    "Super Mario Bros": ("/Multi-layer/smb-multi-layer.json", "/Multi-layer/Structural Layer"),
    "Super Mario Bros 2": ("/../Super Mario Bros/smb.json", "/Processed/WithEnemies"),
    "Super Mario Kart": ("/smk.json", "/Processed"),
    "Doom": ("/doom_tile.json", "/Processed"),
    "Doom2": ("/doom_tile.json", "/Processed"),
    "Kid Icarus": ("/KidIcarus.json", "/Processed"),
    "Lode Runner": ("/Loderunner.json", "/Processed"),
    "MegaMan": ("/MM.json", "/level_txt")
}

class ModelTrainer():
    def __init__ (self, learning_rate, batch_size, n_epochs):
        self.processor = DataProcessor(None)
        self.loader = DataLoader()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs


    def train(self, game_name, model, loss_func, optimizer, device, is_save = False):
        """
        Do training
        NOTE: this method only provides a very basic training process, a formal way to use it is to make a subclass of this class and re-implement this method to better fit your model
        """
        model.train()
        for epoch in range(self.n_epochs):
            print("epoch {}".format(epoch+1))
            temp_processor = self.processor.copy()
            temp_processor.shuffle()
            temp_processor.makeBatch(self.batch_size) # (num_batches, batch_size, row, col, channel)

            temp_processor.swapAxes(2, 4) # (num_batches, batch_size, channel, col, row)
            temp_processor.swapAxes(3, 4) # (num_batches, batch_size, channel, row, col)

            # NOBUG
            temp_processor.data = torch.tensor(temp_processor.data, dtype=torch.float32)

            counter = 0
            for batch in temp_processor.data:
                batch = batch.to(device)
                
                optimizer.zero_grad()
                output = model(batch)
                loss = loss_func(output, batch)

                loss.backward()
                optimizer.step()
                counter += 1
                if (counter % 500 == 0):
                    print("batch #{:05d}, loss:{:.8f}".format(counter, loss))
        if is_save:
            self.saveModel(model, optimizer, game_name, device, self.learning_rate, self.batch_size)
        return model


    def loadAndPrepare(self, game_name):
        """ 
        Load all level of game, and process all levels to data format ready for training
        Argument:
        game_name (string): the name of the game, see DATASET_INFO
        """
        json_path = VGLC_PATH + game_name + DATASET_INFO[game_name][0]
        data_folder_path = VGLC_PATH + game_name + DATASET_INFO[game_name][1]

        self.loader.loadJson(json_path)
        self.loader.loadFolder(data_folder_path)
        # now data is like list[lvl_1, lvl_2, lvl_3 ...]

        # make each level into segments, so the data size are all the same, only number of data are different, we can use concat_datasets to put all of them together
        list_of_data = list()
        for each_level in self.loader.loaded_data:
            self.processor.data = each_level
            self.processor.makeSegments(8, 1)
            list_of_data.append(self.processor.data)
        # now data is like list[lvl_1(num_segments, row, col, features), lvl_2, ...]

        self.processor.concatDatasets(list_of_data)
        # new data is like list[l1p1, l1p2, l1p3, ... , l2p1, l2p2, ...]

        # now the ready data is in self.processor.data

    
    def saveModel(self, model, optimizer, game_name, device, learning_rate, batch_size):
        """ Save the model """
        model_file_name = "model" + strftime("_%m-%d_%H-%M")
        torch.save({
            "game_name": game_name, 
            "device": device,
            "learning_rate": learning_rate,
            "batch_size": batch_size, 
            "model": model.state_dict(), 
            "optimizer": optimizer.state_dict(), 
            }, "./saved_model/" + model_file_name)

        print("Model saved as " + model_file_name)

    def loadModel(self, model, optimizer, path):
        """
        load the model
        Arguments:
            model: a not trained model, must have the same structure as the loading model
            optimizer: an optimizer
            path: the path to the model file
        Returns:
            model: loaded model
            optimizer: loaded optimizer
            info: a dictionary containing game name, device, learning rate and batch size
        """
        checkpoint = torch.load(path)

        info = dict()

        print("loading model:")
        print("{}: {}".format("game_name", checkpoint["game_name"]))
        info["game_name"] = checkpoint["game_name"]

        print("{}: {}".format("device", checkpoint["device"]))
        info["device"] = checkpoint["device"]

        print("{}: {}".format("learning_rate", checkpoint["learning_rate"]))
        info["learning_rate"] = checkpoint["learning_rate"]

        print("{}: {}".format("batch_size", checkpoint["batch_size"]))
        info["batch_size"] = checkpoint["batch_size"]

        print("loading model and optimizer", end = ", ")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("loaded.")

        return model, optimizer, info


def test():
    pass


if __name__ == "__main__":
    test()
