import pandas as pd


class EnformerDataLoader:
    def __init__(self, data: pd.DataFrame):
        self.data = data


# DEBUGGING PURPOSES ONLY #
if __name__ == "__main__":
    data = EnformerDataLoader(pd.read_csv("meuleman_data/train_all_classifier_light.csv", sep="\t"))
    dataframe = data.data
    print(dataframe)
