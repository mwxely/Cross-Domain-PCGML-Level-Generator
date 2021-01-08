from data_loader import DataLoader
from data_processor import DataProcessor
from trainer import ModelTrainer
from model_1 import DropoutModel8x8

import torch
import numpy as np

#   #########   #########     ########   #########
#       #       #            ###     #       #
#       #       #######        #####         #
#       #       #            #     ###       #
#       #       #########    #######         #

#    #######    #       #   #          ##       ##
#   #       #   ##      #   #            ##   ##
#   #       #   # ##    #   #               #
#   #       #   #   ##  #   #               #
#    #######    #     ###   #########       #

# DO NOT CHANGE, THIS IS ONLY AN ATTEMPT

def loadLevel():
    json_path = "../TheVGLC/Super Mario Bros/Multi-layer/smb-multi-layer.json"
    level_path = "../TheVGLC/Super Mario Bros/Multi-layer/Structural Layer/mario-1-1.txt"

    loader = DataLoader()
    loader.loadJson(json_path)
    loader.loadFile(level_path)

    original_size = loader.loaded_data[0].shape

    processor = DataProcessor(loader.loaded_data[0])
    processor.makeSegments(8, 1)

    return loader, processor, original_size


def test(model, processor, device, loss_func, original_size):
    model.eval()
    processor.makeBatch(1)

    processor.swapAxes(2, 4) # (num_batches, 1, channel, col, row)
    processor.swapAxes(3, 4) # (num_batches, 1, channel, row, col)

    processor.data = torch.tensor(processor.data, dtype=torch.float32)

    outputs = list()

    # Do evalution, the model outputs many level segments
    counter = 0
    for batch in processor.data:
        batch = batch.to(device)
        
        output = model(batch)
        loss = loss_func(output, batch)

        outputs.append(output[0].cpu().detach().numpy()) # (channel, row, col)

        counter += 1
        if (counter % 100 == 0):
            print("batch #{:05d}, loss:{:.8f}".format(counter, loss))


    # put all level segments together to form a complete level
    # because segments overlap each other, so just add all predictions of one position
    # later, we have to choose the best sprite for every sprite position, we have to have a 
    # threshold to decide whether to accept a sprite
    gen_level_tensor = np.zeros(original_size)
    addition_count = np.zeros(original_size)
    height, width, depth = original_size
    segment_size = [8,8]
    stride = [1,1]

    segment_idx = 0
    for row in range(0, height - segment_size[0] + 1, stride[0]):
        for col in range(0, width - segment_size[1] + 1, stride[1]):
            segment = outputs[segment_idx]
            segment = np.swapaxes(segment, 0, 2) # (col, row, channel)
            segment = np.swapaxes(segment, 0, 1) # (row, col, channel)
            gen_level_tensor[row:row+segment_size[0], col:col+segment_size[1]] += segment
            addition_count[row:row+segment_size[0], col:col+segment_size[1]] += 1

            segment_idx += 1
    gen_level_tensor /= addition_count

    print(original_size, gen_level_tensor.shape)

    gen_level = np.zeros(original_size[0:2])
    for row in range(original_size[0]):
        for col in range(original_size[1]):
            max_sprite = np.argmax(gen_level_tensor[row, col])
            mean = 3*np.average(gen_level_tensor[row, col])
            print(mean)
            gen_level[row, col] = max_sprite if gen_level_tensor[row, col, max_sprite] > 0.01 else 0
    print(gen_level.shape)

    return gen_level


if __name__ == "__main__":
    loader, processor, original_size = loadLevel()
    device = "cuda"

    model = DropoutModel8x8(32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_func = torch.nn.MSELoss().to(device)

    trainer = ModelTrainer(0.0001, 1, 1)
    model, _, _ = trainer.loadModel(model, optimizer, "saved_model/model_05-20_17-17")

    gen_level = test(model, processor, device, loss_func, original_size)
    gen_level = gen_level.astype(int)

    output_file = open("generated.txt", "w")
    for row in range(original_size[0]):
        for col in range(original_size[1]):
            if (gen_level[row, col] == 0):
                output_file.write("-")
            else:
                output_file.write(loader.tile_reprs[gen_level[row, col]])
        output_file.write("\n")
    output_file.close()





