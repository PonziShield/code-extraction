import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import csv
import pandas as pd
from opcode_extraction.bytecode_disassembler.disassembler import Disassembler
from embedding_vector.embedding_vector import EmbeddingVector
import torch
import numpy as np

class CreateDataset:
    def __init__(self,no_of_files):
        self.no_of_files = no_of_files
        self.dataset = "./data/PonziDataset.csv"
        self.api_key="74EXH3ZYYXPYAA9M1AAUCHSXXQ62MVXANW"

        df = pd.read_csv(self.dataset)
        self.filtered_df = df[['address', 'contractCode', 'label']]


    def createFeatureDataset(self):
        filtered_df = self.filtered_df
        for i in range(len(filtered_df)):

            if i >= self.no_of_files:
                break

            # print(filtered_df.loc[i, "address"], filtered_df.loc[i, "contractCode"])
            fileNameToSave = './data/data_set/' + filtered_df.loc[i, "address"] + '.csv'
            disassembler_instance = Disassembler(filtered_df.loc[i, "contractCode"])
            disassembler_instance.load_opcodes()
            encoding_opcodes = disassembler_instance.get_encoded_opcodes()

            array_length = 1000
            no_of_opcodes = 256

            # Get the first 2000 opcodes or add zero padding
            selected_opcodes = encoding_opcodes[:array_length] + [0] * max(0, array_length - len(encoding_opcodes))
            # data = selected_opcodes

            # padded_opcodes = list(set(selected_opcodes))
            #
            # # Add zero padding using extend to 256
            # padded_opcodes.extend([0] * max(0, no_of_opcodes - len(padded_opcodes)))
            # padded_opcodes.sort()


            # for the embedding vector
            embed_vec = EmbeddingVector()
            encoding_opcodes_tensor = torch.tensor(selected_opcodes, dtype=torch.float32)
            reshaped_tensor = encoding_opcodes_tensor.view(1, array_length, 1)
            torch.set_printoptions(precision=1)
            embedding_vector = embed_vec(reshaped_tensor)
            rounded_embedding_vector = torch.round(embedding_vector * 100) / 100


            data = rounded_embedding_vector

            header_names = []
            for i in range(8):
                header_name = "col_" + str(i+1)
                header_names.append(header_name)

            self.write_to_csv(fileNameToSave, data, header_names)
            print("-" * 70)


    def write_to_csv(self, filename, data, header_names):
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)

            # Write header
            csvwriter.writerow(header_names)

            # Reshape the tensor to a 2D list (256 rows x 8 columns)
            tensor_data = data.view(-1, 8).tolist()

            # Write rows
            for row in tensor_data:
                csvwriter.writerow(row)

        print(f'Data has been written to {filename}.')

    def create_numpy_array(self):
        all_data = []
        all_labels = []
        filtered_df = self.filtered_df
        for i in range(len(filtered_df)):

            if i >= self.no_of_files:
                break

            # print(filtered_df.loc[i, "address"], filtered_df.loc[i, "label"])
            fileNameToRead = './data/data_set/' + filtered_df.loc[i, "address"] + '.csv'

            data = pd.read_csv(fileNameToRead)

            # Extract the relevant data
            features = data.iloc[:, :8].to_numpy()
            label = filtered_df.loc[i, "label"]

            all_data.append((features))
            all_labels.append(label)

            print("-" * 70)

        data_array = np.array(all_data)
        labels_array = np.array(all_labels)

        print(data_array.shape)
        print(labels_array.shape)

        # Save the numpy array to a file (e.g., npy or npz format)
        np.save('./data/other/data.npy', data_array)
        np.save('./data/other/labels.npy', labels_array)

        print('Numpy arrays have been created using the data.')




cd = CreateDataset(300)
# cd.createFeatureDataset()
cd.create_numpy_array()
