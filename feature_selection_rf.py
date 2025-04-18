import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


class FeatureSelector:
    def __init__(self, csv_path='breast_cancer_data.csv'):
        # Loading dataset
        self.df = pd.read_csv(csv_path)

        # Encoding diagnosis: M -> 1, B -> 0
        self.df['diagnosis'] = LabelEncoder().fit_transform(self.df['diagnosis'])

        # Drop ID column
        if 'id' in self.df.columns:
            self.df.drop(columns=['id'], inplace=True)

        self.y = self.df['diagnosis']
        self.X = self.df.drop(columns=['diagnosis'])

        # Splitting the dataset
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.30, random_state=42, shuffle=True
        )

        # Initialize and train the model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=2,
            criterion="gini",
            random_state=42
        )
        self.model.fit(self.x_train, self.y_train)

    def __len__(self):
        return self.X.shape[1]

    def accuracy(self):
        """Returns accuracy score of the trained model."""
        return self.model.score(self.x_test, self.y_test)


def main():
    fs = FeatureSelector()
    acc = fs.accuracy()
    print(f'Accuracy using all {len(fs)} features: {acc:.5f}')
    return acc


if __name__ == "__main__":
    main()
