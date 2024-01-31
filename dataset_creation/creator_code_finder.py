import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import csv
import pandas as pd

class CreatorCodeFinder:
    def __init__(self,no_of_files):
        self.no_of_files = no_of_files
        self.dataset = "./data/PonziDataset.csv"
        self.api_key="74EXH3ZYYXPYAA9M1AAUCHSXXQ62MVXANW"

        df = pd.read_csv(self.dataset)
        self.filtered_df = df[['address', 'creator', 'label']]
        self.smart_contract_details = {}



    def creator_codes_find(self):
        filtered_df = self.filtered_df
        for i in range(len(filtered_df)):
            if i >= self.no_of_files:
                break

            creator = filtered_df.loc[i, "creator"]
            address = filtered_df.loc[i, "address"]
            label = filtered_df.loc[i, "label"]

            # Check if the creator is already in the dictionary
            if creator not in self.smart_contract_details:
                # If no, create a new entry for the creator
                self.smart_contract_details[creator] = {"ponzi": [], "non-ponzi": []}

            # Append the address to the corresponding label within the nested dictionary
            if label == 1:
                self.smart_contract_details[creator]["ponzi"].append(address)
            else:
                self.smart_contract_details[creator]["non-ponzi"].append(address)

        # Save data to CSV
        self.save_to_csv()

    def save_to_csv(self):
        rows = []
        for creator, addresses in self.smart_contract_details.items():
            ponzi_count = len(addresses['ponzi'])
            non_ponzi_count = len(addresses['non-ponzi'])

            # Skip if the sum of ponzi count and non-ponzi count is equal to 1
            if ponzi_count + non_ponzi_count <= 1:
                continue

            ponzi_addresses = ', '.join(addresses['ponzi'])
            non_ponzi_addresses = ', '.join(addresses['non-ponzi'])
            rows.append([creator, ponzi_count, ponzi_addresses, non_ponzi_count, non_ponzi_addresses])

            # Create a DataFrame and save to CSV
        result_df = pd.DataFrame(rows, columns=['Creator', 'Ponzi_Addresses_Count', 'Ponzi_Addresses',
                                                'Non_Ponzi_Addresses_Count', 'Non_Ponzi_Addresses'])
        fileNameToSave = './creator_find/creator_details_filtered.csv'
        result_df.to_csv(fileNameToSave, index=False)
        print("Data saved to creator_details.csv")


creator_finder = CreatorCodeFinder(6498)
creator_finder.creator_codes_find()
