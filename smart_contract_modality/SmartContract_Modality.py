import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import warnings
# Ignore the specific warning
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.instancenorm")

from bytecode_extraction.evm_bytecode_extraction import get_contract_bytecode, get_contract_name
from opcode_extraction.bytecode_disassembler.disassembler import Disassembler
from embedding_vector.embedding_vector import EmbeddingVector
from ast import literal_eval
import torch
from torch import nn, Tensor
from classifier.classifier import Classifier
from binary_accuracy.binary_accuracy import binary_acc
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from training.custom_data import CustomData
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
import torch.optim as optim
import matplotlib.pyplot as plt
# import seaborn as sn
import csv
import pandas as pd
import itertools



array_length = 1000

device = torch.device("cpu")
# model = Classifier(d_model=8, seq_len=array_length, nhead=8, dim_feedforward=8, nlayers=8, device=device)
# model.to(device)

X_test = np.load("../dataset_creation/more/X_test.npy", allow_pickle=True)
y_test = np.load("../dataset_creation/more/y_test.npy", allow_pickle=True)

test_data = CustomData(X_test, y_test)
test_loader = DataLoader(dataset=test_data, batch_size=2)

PATH = '../sc_model/PonziShield_sc_v1.pth'
loaded_model = torch.load(PATH)
loaded_model.eval()

y_pred = []
y_true = []
sigmoid = nn.Sigmoid()

for X_batch, y_batch in test_loader:
        #print("w.requires_grad:",X_batch.requires_grad)
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        # print(X_batch.shape)
        # print(y_batch.shape)
        results,result_before_bin_classifier = loaded_model(X_batch.float())
        # print(results.shape)
        y_batch_pred = torch.round(sigmoid(results))
        y_pred.extend(y_batch_pred.cpu().detach().numpy())
        y_true.extend(y_batch.cpu().detach().numpy())


count_true = 0
for i in range(len(y_pred)):
    if y_true[i] == y_pred[i]:
        count_true += 1
acc = count_true / len(y_pred)
# print(len(y_test))
# print(acc)


def create_tensor_inputs(embedding_dir, contract_address):
    dataset = "../dataset_creation/data/PonziDataset.csv"
    api_key = "74EXH3ZYYXPYAA9M1AAUCHSXXQ62MVXANW"

    df = pd.read_csv(self.dataset)
    filtered_df = df[['address', 'label']]

    all_data = []
    all_labels = []
    for i in range(len(contract_address)):
        # print(filtered_df.loc[i, "address"], filtered_df.loc[i, "label"])
        fileNameToRead = embedding_dir + str(contract_address[i]) + '.csv'
        data = pd.read_csv(fileNameToRead)
        # Extract the relevant data (assuming the label column is named 'label')
        features = data.iloc[:, :8].to_numpy()
        label = filtered_df.loc[i, "label"]
        # label = data['label'][1]
        # print(features[1], labels[1])
        all_data.append((features))
        all_labels.append(label)
        # print("-----------------------------------------------------------------------")
    data_array = np.array(all_data)
    labels_array = np.array(all_labels)

    # Reshape the array to (301*108, 11) for normalization
    reshaped_data = data_array.reshape((-1, 8))
    # Initialize the StandardScaler
    scaler = StandardScaler()
    # Fit the scaler on the reshaped data and transform it
    normalized_data = scaler.fit_transform(reshaped_data)
    # Reshape the normalized data back to the original shape
    normalized_data_array = normalized_data.reshape(data_array.shape)


    data_tensor = torch.tensor(normalized_data_array, dtype=torch.float32)
    labels_tensor = torch.tensor(labels_array, dtype=torch.float32)

    print(data_tensor.shape)
    print(labels_tensor.shape)
    return data_tensor, labels_tensor


class SmartContractModality(nn.Module):
    def __init__(
            self,
            device,
            inductor=True,
            embedding_dir='../dataset_creation/data/data_set/',
            model_path='../sc_model/PonziShield_sc_v1.pth',
    ):
        super(SmartContractModality, self).__init__()

        self.device = device
        self.embedding_dir = embedding_dir
        self.model = torch.load(model_path)

    def forward(self, dapp_addresses, train):
        if train == False:
            # do realtime prediction
            return

        # create 3d tensor [dapp_count,sequence_length,features]
        data_tensor, labels_tensor = create_tensor_inputs(self.embedding_dir, dapp_addresses)
        data_tensor = data_tensor.to(self.device)
        results, result_before_bin_classifier = self.model(data_tensor.float())

        # results shape = [dapp_count,1], result_before_bin_classifier = [dapp_count,sequence_length,features]
        return results, result_before_bin_classifier