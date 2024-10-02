from flask import Flask, jsonify, render_template
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load the dataset
def load_data():
    file_path = 'D:\\Customer Segmentation\\Mall_Customers.csv'
    logging.debug(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    return df

# Process and segment the data
def segment_customers():
    try:
        df = load_data()
        X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # K-Means clustering
        kmeans = KMeans(n_clusters=5, random_state=0)
        df['Cluster'] = kmeans.fit_predict(X_scaled)

        # Cluster descriptions
        cluster_descriptions = {
            0: "High Income, High Spending",
            1: "High Income, Low Spending",
            2: "Low Income, High Spending",
            3: "Low Income, Low Spending",
            4: "Middle Income, Moderate Spending"
        }

        # Add descriptive cluster names
        df['ClusterName'] = df['Cluster'].map(cluster_descriptions)

        # Convert to a list of dictionaries
        result = df[['CustomerID', 'Cluster', 'ClusterName']].to_dict(orient='records')
        logging.debug("Segmentation successful")
        return result
    except Exception as e:
        logging.error(f"Error during segmentation: {e}")
        return []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/segment')
def segment():
    data = segment_customers()
    logging.debug(f"Segmented data: {data}")
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
