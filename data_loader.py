import os
import json
import numpy as np

"Code by Jinming Wang, Zuhao Yang"
class DataLoader():
    """
    This only works when reading .txt file
    """
    def __init__(self):
        self.tile_reprs = list() # character for each tile
        self.loaded_files = list() # all files that are loaded
        self.loaded_data = list() # a list of tensors representing levels


    def __getitem__ (self, idx):
        """ Can directly use loader[idx] to access level tensor """
        return self.loaded_data[idx]


    def __str__ (self):
        """ Return a string description of this data loader """
        string = "data loader:\n"
        for i in range(len(self.loaded_files)):
            string += "level: {}, dimension: {}.\n" \
                .format(self.loaded_files[i], str(np.shape(self.loaded_data[i])))
        return string


    def __repr__ (self):
        print(self.__str__())


    def loadJson (self, path):
        """
        Loads a json file so we know the level representation
        path (string): the path of json file
        """

        # get all lines in json, concatenate then into a big string then parse it
        with open(path, "r") as file_content:
            all_lines = file_content.readlines()
            all_content_str = "".join(all_lines)
            json_dict = json.loads(all_content_str)
            self.tile_reprs = list(json_dict['tiles']['structural-tiles'].keys())

            # remove this empty char
            self.tile_reprs.remove("-")


    def loadFolder(self, path):
        """
        Loads and parse all files in a folder
        path (string): the path of folder
        """
        for file_name in os.listdir(path):
            if (file_name.split(".")[-1] == "txt"):
                file_path = path + "/" + file_name
                self.loadFile(file_path)


    def loadFile(self, path):
        """
        Loads a file and parse it
        path (string): the path of the file
        """
        print("loading \'{}\',".format(path.split('/')[-1]), end = " ")
        with open(path, "r") as file_content:
            list_of_lines = file_content.readlines() # get all lines of level representation

            # remove '\n' if a line has '\n' at the end
            for i in range(len(list_of_lines)):
                if (list_of_lines[i][-1] == "\n"):
                    list_of_lines[i] = list_of_lines[i][:-1] # remove '\n'

            #  calculate dimensions of the level tensor
            width = len(list_of_lines[0])
            height = len(list_of_lines)
            depth = len(self.tile_reprs)
            level_tensor = np.zeros((height, width, depth)) # this tensor represent level

            # traverse the entire level space to populate the tensor
            for row in range(height):
                for col in range(width):
                    char = list_of_lines[row][col]
                    # if this character does not exist in json file, it is all zero
                    try:
                        level_tensor[row, col, self.tile_reprs.index(char)] = 1
                    except:
                        pass


            self.loaded_files.append(path.split('/')[-1].split('.')[0])
            self.loaded_data.append(level_tensor) # store this loaded level
        print("success.")


def test ():
    loader = DataLoader()
    json_path = "../TheVGLC/Super Mario Bros/Multi-layer/smb-multi-layer.json"
    file_path1 = "../TheVGLC/Super Mario Bros/Multi-layer/Structural Layer/mario-1-1.txt"
    file_path2 = "../TheVGLC/Super Mario Bros/Multi-layer/Structural Layer/mario-2-1.txt"
    loader.loadJson(json_path)
    loader.loadFile(file_path1)
    loader.loadFile(file_path2)
    print(loader)
    print(np.shape(loader[1]))


if __name__ == "__main__":
    test()
