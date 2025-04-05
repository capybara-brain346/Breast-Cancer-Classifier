import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickletools


class Model:
    def __init__(self, data) -> None:
        self.data = data
        self.random_forest_model = None
        self.y_test = None
        self.y_train = None
        self.x_test = None
        self.x_train = None

    def data_splitting(self):
        x = self.data.drop(["target"], axis=1)
        y = self.data["target"]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=0.4, random_state=60
        )

    def train(self):
        self.data_splitting()
        self.random_forest_model = RandomForestClassifier(max_depth=3).fit(
            self.x_train, self.y_train
        )
        return self.random_forest_model

    def final_build(self):
        with open("artifacts/model.pkl", "wb") as file:
            pickle.dump(self.random_forest_model, file)


def main():
    model_data = pd.read_csv("data/model_training_data.csv")
    Breast_Cancer_Classifier = Model(data=model_data)
    Breast_Cancer_Classifier.train()
    Breast_Cancer_Classifier.final_build()
    with open("artifacts/model.pkl", "rb") as f:
        pickletools.dis(f)


if __name__ == "__main__":
    main()
