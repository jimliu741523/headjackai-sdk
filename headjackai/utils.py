import pandas as pd

def pandas_check(df):
    assert isinstance(df, pd.DataFrame), "data frame must be pandas data frame"
