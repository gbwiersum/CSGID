from OracleDBConnector import OracleDBConnector
OracleDBConnector()
import pandas as pd

def Image_getter():
    df = pd.DataFrame(columns = ["ImagePath", "HumanScore", "AIScore"])
    # TODO: get columns
    # TODO: Write images to local drive
    return df

Image_getter()
