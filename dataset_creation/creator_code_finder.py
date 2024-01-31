import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import csv
import pandas as pd

class CreatorCodeFinder:
    def __init__(self,no_of_files):
        self.no_of_files = no_of_files
        self.dataset = "./dataset_creation/data/PonziDataset.csv"
        self.api_key="74EXH3ZYYXPYAA9M1AAUCHSXXQ62MVXANW"

        df = pd.read_csv(self.dataset)
        self.filtered_df = df[['address', 'contractCode', 'label']]
        self.smart_contract_details = {}


    def creator_codes_find(self):
        filtered_df = self.filtered_df
        for i in range(len(filtered_df)):
            if i >= self.no_of_files:
                break

            creator = filtered_df.loc[i, "creator"]
            address = filtered_df.loc[i, "address"]

            # Check if the creator is already in the dictionary
            if creator in self.smart_contract_details:
                # If yes, append the new address to the existing list
                self.smart_contract_details[creator].append(address)
            else:
                # If no, create a new entry with the creator and a list containing the address
                self.smart_contract_details[creator] = [address]

        print("Smart Contract Details:", self.smart_contract_details)


creator_finder = CreatorCodeFinder(6498)
creator_finder.creator_codes_find()
