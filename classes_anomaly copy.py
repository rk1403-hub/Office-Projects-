import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from pymongo import MongoClient
import certifi



class DataProcessor:
    def __init__(self, connection_string):
        self.client = MongoClient(connection_string, tlsCAFile=certifi.where())
        self.db = self.client.dashboard

    def fetch_data_splunk(self, site, key):
        return list(self.db.splunk_logs.find({"site": site, "key": key}))

    def fetch_data_sccv(self, site):
        return list(self.db.get_collection("sccvLog").find({'site': site}))

class RegressionModel:
    def __init__(self):
        self.model = LinearRegression()

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X).round().astype(int)

class VulnerabilityPredictor:
    def __init__(self, data_processor, regression_model):
        self.data_processor = data_processor
        self.regression_model = regression_model

    def process_sccv_data(self, data_sccv):
        df_sccv = pd.DataFrame(data_sccv)
        df_sccv['current_vulns_count'] = df_sccv['current_vulns'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        df_sccv['dateMonth'] = pd.to_datetime(df_sccv['dateMonth'])
        df_sccv['month'] = df_sccv['dateMonth'].dt.month
        df_sccv['year'] = df_sccv['dateMonth'].dt.year
        return df_sccv

    def train_and_evaluate_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.regression_model.train_model(X_train, y_train)
        y_pred = self.regression_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error: {mse}")

    def predict_vulnerabilities(self, df_sccv):
        df_sccv_filtered = df_sccv[df_sccv['current_vulns_count'] != 0]
        X = df_sccv_filtered[['month', 'year']]
        y = df_sccv_filtered['current_vulns_count']
        self.train_and_evaluate_model(X, y)

        non_zero_indices = df_sccv.index[df_sccv['current_vulns_count'] != 0].tolist()
        predictions = self.regression_model.predict(df_sccv.loc[non_zero_indices, ['month', 'year']])
        df_sccv['predicted_vulns'] = 0
        df_sccv.loc[non_zero_indices, 'predicted_vulns'] = predictions
        print(df_sccv[['month', 'year', 'current_vulns_count', 'predicted_vulns']])

    def predict_new_entry(self):
        new_month = int(input("Enter the month (1-12): "))
        new_year = int(input("Enter the year: "))
        new_entry = pd.DataFrame({'month': [new_month], 'year': [new_year]})
        predicted_vulns = self.regression_model.predict(new_entry[['month', 'year']])
        print(f"Predicted current vulnerabilities count for {new_month}-{new_year}: {predicted_vulns[0]}")

def main():
    connection_string = "mongodb+srv://prajwal:wSnhpJZel3jMaesA@staging.awytu.mongodb.net/?retryWrites=true&w=majority"
    data_processor = DataProcessor(connection_string)
    regression_model = RegressionModel()
    predictor = VulnerabilityPredictor(data_processor, regression_model)

    site = "ayrshire"
    data_sccv = data_processor.fetch_data_sccv(site)

    df_sccv = predictor.process_sccv_data(data_sccv)
    predictor.predict_vulnerabilities(df_sccv)
    predictor.predict_new_entry()

#main function 
if __name__ == "__main__":
    main()
