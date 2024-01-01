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
import pandas as pd
import itertools


if __name__ == '__main__':
    # contract_addr = "0x6e38a457c722c6011b2dfa06d49240e797844d66"
    # contract_name = get_contract_name(contract_addr)
    # bytecode = get_contract_bytecode(contract_addr)
    # # print(contract_name, bytecode)
    #
    # disassembler_instance = Disassembler(bytecode)
    # # opcode_sequences = disassembler_instance.get_disassembled_code()
    # disassembler_instance.load_opcodes()
    # opcodes = disassembler_instance.get_disassembled_opcode_values()
    # encoding_opcodes = disassembler_instance.get_encoded_opcodes()
    #
    #
    array_length = 1000
    #
    # # Get the first 2000 opcodes or add zero padding
    # selected_opcodes = encoding_opcodes[:array_length] + [0] * max(0, array_length - len(encoding_opcodes))
    # padded_opcodes = list(set(selected_opcodes))
    #
    # no_of_opcodes = 256
    # # Add zero padding using extend to 256
    # padded_opcodes.extend([0] * max(0, no_of_opcodes - len(padded_opcodes)))
    # padded_opcodes.sort()
    #
    #
    #
    # # for the embedding vector
    # embed_vec = EmbeddingVector()
    # encoding_opcodes_tensor = torch.tensor(padded_opcodes, dtype=torch.float32)
    # reshaped_tensor = encoding_opcodes_tensor.view(1, no_of_opcodes, 1)
    # torch.set_printoptions(precision=1)
    # embedding_vector = embed_vec(reshaped_tensor)
    # rounded_embedding_vector = torch.round(embedding_vector * 100) / 100
    # # print(reshaped_tensor)
    # # print(rounded_embedding_vector)
    #
    #
    # # transformer encoder
    device = torch.device("cpu")
    model = Classifier(d_model=8, seq_len=array_length, nhead=8, dim_feedforward=8, nlayers=8, device=device)
    model.to(device)
    # clf = model(embedding_vector)
    # print(clf)

    x = np.load("dataset_creation\data\other\data.npy")
    y = np.load("dataset_creation\data\other\labels.npy")

    # Normalization
    reshaped_data = x.reshape((-1, 8))

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit the scaler on the reshaped data and transform it
    normalized_data = scaler.fit_transform(reshaped_data)

    # Reshape the normalized data back to the original shape
    normalized_data_array = normalized_data.reshape(x.shape)
    # print(normalized_data_array[:5])


    X, X_test, Y, y_test = train_test_split(normalized_data_array, y, test_size=0.1, random_state=42, stratify=y)

    # Assuming you have split the data using the code provided in the previous response

    # Check class distribution in original 'y'
    unique_classes_y, counts_y = np.unique(y, return_counts=True)
    # print("Original class distribution:")
    # print(dict(zip(unique_classes_y, counts_y)))

    # Check class distribution in 'y_train'
    unique_classes_train, counts_train = np.unique(Y, return_counts=True)
    # print("\nClass distribution in y_train:")
    # print(dict(zip(unique_classes_train, counts_train)))

    # Check class distribution in 'y_test'
    unique_classes_test, counts_test = np.unique(y_test, return_counts=True)
    # print("\nClass distribution in y_test:")
    # print(dict(zip(unique_classes_test, counts_test)))


    # np.save("dataset_creation/more/X_test.npy", X_test)
    # np.save("dataset_creation/more/y_test.npy", y_test)

    # load test
    X_test = np.load("dataset_creation/more/X_test.npy", allow_pickle=True)
    y_test = np.load("dataset_creation/more/y_test.npy", allow_pickle=True)

    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.1, random_state=42,
                                                      stratify=Y)  # validation split

    # np.save("dataset_creation/more/X_train.npy", X_train)
    # np.save("dataset_creation/more/y_train.npy", y_train)
    # np.save("dataset_creation/more/X_val.npy", X_val)
    # np.save("dataset_creation/more/y_val.npy", y_val)

    # load train val
    X_train = np.load("dataset_creation/more/X_train.npy", allow_pickle=True)
    y_train = np.load("dataset_creation/more/y_train.npy", allow_pickle=True)
    X_val = np.load("dataset_creation/more/X_val.npy", allow_pickle=True)
    y_val = np.load("dataset_creation/more/y_val.npy", allow_pickle=True)

    train_data = CustomData(X_train, y_train)
    val_data = CustomData(X_val, y_val)
    test_data = CustomData(X_test, y_test)

    EPOCHS = 20
    BATCH_SIZE = 8
    LEARNING_RATE = 0.001

    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=1)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Hyperparameter Tuning
    # hyperparameters = {
    #     'epochs': [10, 20],
    #     'batch_size': [8, 16],
    #     'learning_rate': [0.001, 0.01]
    # }
    #
    # # Generate all combinations of hyperparameters
    # param_combinations = list(itertools.product(*hyperparameters.values()))
    #
    # # Iterate over each combination
    # for params in param_combinations:
    #     # Unpack the parameters
    #     epochs, batch_size, learning_rate = params
    #
    #     # Create a new model and optimizer with the current hyperparameters
    #     model = Classifier(d_model=8, seq_len=no_of_opcodes, nhead=8, dim_feedforward=8, nlayers=8, device=device)
    #     model.to(device)
    #     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #     # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    #
    #     # Training loop
    #     for e in range(1, epochs + 1):
    #         # ... (your existing training code)
    #         epoch_loss = 0
    #         epoch_acc = 0
    #         model.train()
    #         for X_batch, y_batch in train_loader:
    #             # print("w.requires_grad:",X_batch.requires_grad)
    #             X_batch, y_batch = X_batch.to(device), y_batch.to(device)
    #             optimizer.zero_grad()
    #
    #             y_pred, result_before_bin_classifier = model(X_batch.float())
    #             # print(f"y_pred = {y_pred}")
    #             # y_pred = (y_pred>0.5).float()
    #             # print(f"y_pred = {y_pred}")
    #             # print(f"y_batch.unsqueeze(1) = {y_batch.unsqueeze(1)}")
    #
    #             loss = criterion(y_pred.float(), y_batch.unsqueeze(1).float())
    #             acc = binary_acc(y_pred.float(), y_batch.unsqueeze(1).float())
    #
    #             loss.backward()
    #             optimizer.step()
    #
    #             epoch_loss += loss.item()
    #             epoch_acc += acc.item()
    #
    #         val_loss = 0
    #         val_accuracy = 0
    #         model.eval()
    #         for X_batch, y_batch in test_loader:
    #             X_batch, y_batch = X_batch.to(device), y_batch.to(device)
    #             y_pred, result_before_bin_classifier = model(X_batch.float())
    #             loss = criterion(y_pred.float(), y_batch.unsqueeze(1).float())
    #             acc = binary_acc(y_pred.float(), y_batch.unsqueeze(1).float())
    #             val_loss += loss.item()
    #             val_accuracy += acc.item()
    #
    #         print(
    #             f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} | Acc: {epoch_acc / len(train_loader):.3f} | Val_Loss: {val_loss / len(test_loader):.5f} | Val_Acc: {val_accuracy / len(test_loader):.3f}')
    #
    #     # Print the results for each hyperparameter combination
    #     print(f'Hyperparameters: Epochs={epochs}, Batch Size={batch_size}, Learning Rate={learning_rate}')
    #     print(
    #         f'Final Training Loss: {epoch_loss / len(train_loader):.5f}, Final Training Acc: {epoch_acc / len(train_loader):.3f}')
    #     print(
    #         f'Final Validation Loss: {val_loss / len(test_loader):.5f}, Final Validation Acc: {val_accuracy / len(test_loader):.3f}')
    #     print('\n')
    #
    #



    for e in range(1, EPOCHS + 1):
        epoch_loss = 0
        epoch_acc = 0
        model.train()
        for X_batch, y_batch in train_loader:
            # print("w.requires_grad:",X_batch.requires_grad)
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            # print(model(X_batch.float()))
            # print("X_batch size:", X_batch.size())
            y_pred, result_before_bin_classifier = model(X_batch.float())
            # print(y_pred.shape)
            # print(f"y_pred = {y_pred}")
            # y_pred = (y_pred>0.5).float()
            # print(f"y_pred = {y_pred}")
            # print(f"y_batch.unsqueeze(1) = {y_batch.unsqueeze(1)}")
            # print(y_pred.float(), y_batch.unsqueeze(1).float())
            loss = criterion(y_pred.float(), y_batch.unsqueeze(1).float())
            # print(loss)
            acc = binary_acc(y_pred.float(), y_batch.unsqueeze(1).float())

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        val_loss = 0
        val_accuracy = 0
        model.eval()
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred, result_before_bin_classifier = model(X_batch.float())
            loss = criterion(y_pred.float(), y_batch.unsqueeze(1).float())
            acc = binary_acc(y_pred.float(), y_batch.unsqueeze(1).float())
            val_loss += loss.item()
            val_accuracy += acc.item()

        # Check for early stopping
        # if val_accuracy > best_validation_accuracy:
        #     best_validation_accuracy = val_accuracy
        #     no_improvement_counter = 0
        #     # Save the trained best fakebert if needed
        #     torch.save(fakebert.state_dict(), '/content/drive/Shareddrives/test/FYP/fake-news/fakebert-twitterus.pth')
        # else:
        #     no_improvement_counter += 1

        print(
            f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} | Acc: {epoch_acc / len(train_loader):.3f} | Val_Loss: {val_loss / len(test_loader):.5f} | Val_Acc: {val_accuracy / len(test_loader):.3f}')
        # print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')

    PATH = 'sc_modal/PonziShield_sc_v1.pth'
    # torch.save(model, PATH)

    loaded_model = torch.load(PATH)
    loaded_model.eval()

    y_pred = []
    y_true = []
    sigmoid = nn.Sigmoid()



    for X_batch, y_batch in test_loader:
        # print("w.requires_grad:",X_batch.requires_grad)
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        results, result_before_bin_classifier = loaded_model(X_batch.float())
        y_batch_pred = torch.round(sigmoid(results))

        y_pred.extend(y_batch_pred.cpu().detach().numpy())
        y_true.extend(y_batch.cpu().detach().numpy())

    count_true = 0
    for i in range(len(y_pred)):
        if y_true[i] == y_pred[i]:
            count_true += 1
    acc = count_true / len(y_pred)
    # print(len(y_test))
    print(acc)
    #
    # print(y_true)
    # print(y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # print(tn, fp, fn, tp)

    # Calculate Recall
    recall = tp / (tp + fn)

    # Calculate Precision
    precision = tp / (tp + fp)

    # Calculate F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall)

    print("Recall: {:.4f}".format(recall))
    print("Precision: {:.4f}".format(precision))
    print("F1 Score: {:.4f}".format(f1_score))

    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1), index=['0', '1'], columns=['0', '1'])
    plt.figure(figsize=(12, 7))
    # sn.heatmap(df_cm, annot=True)
    plt.savefig('output.png')



    # y_test = torch.randint(2, (100,))
    # print(binary_acc(clf, y_test))
