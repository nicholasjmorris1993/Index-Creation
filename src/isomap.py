import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import Isomap


def index(df):
    model = Index()
    model.create_index(df)

    return model


class Index:
    def create_index(self, df):
        # standardize the columns to take on values between 0 and 1
        columns = df.columns
        scaler = MinMaxScaler()
        df = scaler.fit_transform(df)
        df = pd.DataFrame(df, columns=columns)

        # train a PCA model
        n_comp = 1  # number of principal components
        component = Isomap(n_components=n_comp)
        component.fit(df)

        # compute index for all the data
        self.index = pd.DataFrame(
            component.transform(df), 
            columns=["Index"],
        )
