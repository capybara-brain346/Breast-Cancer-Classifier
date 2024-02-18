import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class model:
    data: pd.core.frame.DataFrame

    def data_spliting(self):
        x = self.data.drop(['target'],axis=1)
        y = self.data.target.to_numpy()
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.4, random_state=60)

    def train(self):
        self.data_spliting()
        self.random_forest_model = RandomForestClassifier().fit(self.x_train, self.y_train)
        return self.random_forest_model
    
    def final_build(self):
        with open(r"model\random_forest_model.pkl", "wb") as file:
            pickle.dump(self.train, file)    
        
def main():
    
    Breast_Cancer_Classifier = model(data=r"../data/model_training_data.csv")
    Breast_Cancer_Classifier.final_build()
       
if __name__=="__main__":
    main()
