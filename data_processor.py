import numpy as np

"Code by Jinming Wang, Zuhao Yang"
class DataProcessor():
    def __init__ (self, data):
        """ Argument: data: the data to be processed """
        self.data = data


    def copy(self):
        return DataProcessor(self.data.copy())


    def makeSegments(self, size, stride):
        """
        Split a self.data into many small segments
        Arguments:
        size (int or 2-tuple): size of each segment, if given int then segment is a square
        stride (int or 2-tuple): the step or distance between segments, if given int then vertical and horizontal steps are the same

        self.data = splitted segments as a list, each element is a self.data (np array)
        Example: 
        [[A B C D]                                          
         [E F G H] -> makeSegments, size=(2, 3), stride=1 ->
         [I J K L]]

         ->  [ [[A B C]  [[B C D]  [[E F G]  [[F G H]
                [E F G]]  [F G H]]  [I J K]]  [J K L]] ]
        """

        # Convert size and stride into 2-tuples if they are not
        if type(size) == int:
            size = (size, size)
        if type(stride) == int:
            stride = (stride, stride)

        # Initialize segments to correct size
        if (self.data.ndim == 3):
            height, width, depth = np.shape(self.data)
            num_segments = ((height - size[0])/stride[0] + 1) * \
                            ((width - size[1])/stride[1] + 1)
            segments = np.zeros((int(num_segments), size[0], size[1], depth))
        elif (self.data.ndim == 2):
            height, width = np.shape(self.data)
            num_segments = ((height - size[0])/stride[0] + 1) * \
                            ((width - size[1])/stride[1] + 1)
            segments = np.zeros((int(num_segments), size[0], size[1]))
        else:
            print("Dimension of given self.data must be 2 or 3")
            return None

        # Do splitting, iterate every row and column of self.data, get a small partition of correct size and assign this partition to segments
        segment_idx = 0
        for row in range(0, height - size[0] + 1, stride[0]):
            for col in range(0, width - size[1] + 1, stride[1]):
                if (self.data.ndim == 2):
                    segments[segment_idx] = self.data[row:row+size[0], col:col+size[1]]
                else:
                    segments[segment_idx] = self.data[row:row+size[0], col:col+size[1], :]
                segment_idx += 1

        self.data = segments


    def shuffle(self):
        """
        Random shuffle all elements in a dataset

        Argument:
        self.data (array like): a set of all data

        self.data = shuffled_self.data
        """
        np.random.shuffle(self.data)

    
    def makeBatch(self, batch_size, ignore_remainder = False):
        """
        Split self.data into batches of size batch_size

        Arguments:
        batch_size (int): the size of each batch
        ignore_remainder (boolean): if set to true, if the last batch cannot reach batch_size, then ignore the last batch and all data in it, if set to false, then some random data from self.data are added to the last batch

        self.data = A list of batches, where each batch contains batch_size data
        """

        data_num = np.shape(self.data)[0]
        batch_num = data_num//batch_size
        remainder_size = data_num % batch_num
        if remainder_size != 0:
            if (ignore_remainder):
                self.data = np.split(self.data[0:data_num - remainder_size], batch_num)
            else:
                num_data_needed = batch_size - remainder_size

                # take some random data from original data set and append them to the end
                # NOBUG
                random_indices = np.random.randint(0, data_num, num_data_needed)
                appenda = np.zeros((num_data_needed, *self.data.shape[1:]))
                for i, j in enumerate(random_indices):
                    appenda[i] = self.data[j]

                self.data = np.concatenate((self.data, appenda))
                self.data = np.split(self.data, batch_num + 1)
        else:
            self.data = np.split(self.data, batch_num)

        
    def concatDatasets(self, list_of_dataset):
        """
        Used to concatenate two data sets, after you loaded a list of different levels, you may apply makeSegments on each level, then you got something like
        [[lvl1_seg1, lvl1_seg2, ...], [lvl2_seg1, lvl2_seg2, ...], ...]
        This function can turn this into
        [lvl1_seg1, lvl1_seg2, ..., lvl2_seg1, lvl2_seg2, ... lvln_segn]

        Arguments: 
        list_of_dataset: a list of lists of data

        self.data = A list of data
        """

        # TODO: complete this implementation as described above
        self.data = np.concatenate(list_of_dataset)

    def swapAxes(self, axis_1, axis_2):
        """
        Swap two axes of self.data, (batch_number, batch_size, row, column, depth)
        """
        self.data = np.swapaxes(self.data, axis_1, axis_2)


def test():
    data = np.random.randint(0, 5, (3, 3, 2))
    processor = DataProcessor(data)
    processor.makeSegments(2, 1)

    print(np.shape(processor.data))

    print("The first element in segments:\n", processor.data[0])

    processor.shuffle()
    print("The first element in shuffled segments:\n", processor.data[0])

    processor.makeBatch(2)
    print("Size of segments after make batches:", np.shape(processor.data))
    print("Expected: (2,2,2,2,2), 2 batches in total, 2 data in each batch, \n" + \
        "\teach data has 2 rows, 2 columns and depth of 2")


if __name__ == "__main__":
    test()
