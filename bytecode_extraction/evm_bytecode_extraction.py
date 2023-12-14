import requests

def get_contract_bytecode(address):
    etherscan_api_key = 'WZ4HKEKRUIP7HSMCBHGDI6G6CSG4S29QJ2'
    etherscan_url = f'https://api.etherscan.io/api?module=proxy&action=eth_getCode&address={address}&apikey={etherscan_api_key}'

    try:
        response = requests.get(etherscan_url)
        data = response.json()

        if data.get('result'):
            bytecode = data['result']
            return bytecode
        else:
            return "Contract bytecode not found"
    except Exception as e:
        return f"Error fetching contract bytecode: {str(e)}"


def get_contract_name(address):
    # Use Etherscan API to fetch the contract name
    etherscan_api_key = 'WZ4HKEKRUIP7HSMCBHGDI6G6CSG4S29QJ2'
    etherscan_url = f'https://api.etherscan.io/api?module=contract&action=getsourcecode&address={address}&apikey={etherscan_api_key}'

    try:
        response = requests.get(etherscan_url)
        data = response.json()

        if data.get('status') == '1' and data.get('result'):
            contract_name = data['result'][0]['ContractName']
            return contract_name
        else:
            return "Contract name not found"
    except Exception as e:
        return f"Error fetching contract name: {str(e)}"



# if __name__ == '__main__':
#     print(get_contract_name("0x815fF5e32F974862839C56f69CFD190F10E262F6"))
#     print(get_contract_bytecode("0x815fF5e32F974862839C56f69CFD190F10E262F6"))

