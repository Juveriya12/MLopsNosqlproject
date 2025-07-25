#Fetch from mongodb
import pandas as pd
from config import collection

def fetch_data():
    data= list(collection.find())
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df.drop(columns=["_id"],inplace= True)

    # Convert 'services' list to number of services
    df["num_services"]= df["services"].apply(lambda x: len(x))
    df.drop(columns=["services"], inplace= True)
    return df 

