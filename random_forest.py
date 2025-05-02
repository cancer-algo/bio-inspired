import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


class FeatureSelection:
    def __init__(self, data_path='1000_encoded_gastric_cancer_data.csv', test_size=0.3, random_state=42):
        self.df = pd.read_csv(data_path)

        # Split into features and label
        self.x = self.df.iloc[:, :-1]
        self.y = self.df.iloc[:, -1]

        # Train-test split
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=test_size, random_state=random_state, shuffle=True
        )

        # Random Forest model initialization and training
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=2,
            criterion="gini",
            random_state=random_state
        )
        self.model.fit(self.x_train, self.y_train)

    def __len__(self):
        return self.x.shape[1]

    def accuracy(self, feature_mask):
        """
        Evaluates model accuracy using a given binary feature mask.
        """
        selected_indices = [i for i, selected in enumerate(feature_mask) if selected == 1]
        if not selected_indices:
            return 0.0  # Prevent evaluation on empty feature set

        x_test_selected = self.x_test.iloc[:, selected_indices]
        return self.model.score(x_test_selected, self.y_test)


def main():
    fs = FeatureSelection()
    all_features_mask = [1] * len(fs)
    accuracy_score = round(fs.accuracy(all_features_mask), 5)

    print(f'Accuracy with all {len(fs.x.columns)} features: {accuracy_score}')


if __name__ == "__main__":
    main()
