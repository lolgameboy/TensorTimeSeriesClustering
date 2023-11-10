import os
import pandas as pd


class DAL:

    folder = "data/"

    def __init__(self, name):
        self.datafile = self.folder + name

        if not os.path.isfile(self.datafile):
            ov = pd.DataFrame(columns=["df_key", "description"])
            ov.to_hdf(self.datafile, "overview")

    def overview(self):
        return pd.read_hdf(self.datafile, key="overview")

    def get(self, df_key):
        return pd.read_hdf(self.datafile, key=df_key)

    def add(self, df, df_key, description, overview=True):
        if overview:
            self._add_to_overview(df_key, description)
        df.to_hdf(self.datafile, df_key)

    def _add_to_overview(self, df_key, description):
        ov = self.overview()
        d = {}
        d["df_key"] = df_key
        d["description"] = description
        ov = ov[ov.df_key != df_key].append(d, ignore_index=True)
        ov.to_hdf(self.datafile, "overview")

dal = DAL("amie-kinect-data.hdf")
ov = dal.overview()
rows = ov
execs = dal.get(rows.df_key.values[0])

Ankle = execs["AnkleLeftX"]
print(Ankle)