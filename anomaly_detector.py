#!/usr/bin/env python

"""
Anomaly Detector Module
This module integrates the transformer-based anomaly detection model with the SDN controller.
It loads the saved model and provides methods to detect anomalies in network flows.
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import time
import warnings
warnings.filterwarnings('ignore')

# Import functions from the ads module
from ads.ads import preprocess_features, prepare_data_for_transformer

class AnomalyDetector:
    def __init__(self, model_path='ads/saved_transformer_model.h5'):
        """
        Initialize the anomaly detector with the saved transformer model.
        
        Args:
            model_path: Path to the saved model file
        """
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.encoder = None
        self.target_encoder = None
        self.class_names = None
        
        # Load the model and encoders
        self._load_model()
        
        print(f"Anomaly Detector initialized with model from {model_path}")
    
    def _load_model(self):
        """
        Load the saved transformer model and initialize encoders.
        """
        try:
            # Load the model
            self.model = load_model(self.model_path)
            print("Model loaded successfully")
            
            # Initialize encoders (these will be fit during the first detection)
            self.scaler = StandardScaler()
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            self.target_encoder = OneHotEncoder(sparse_output=False)
            
            # Define class names based on the TON_IoT dataset
            self.class_names = ['normal', 'backdoor', 'ddos', 'injection', 'mitm', 'password', 'ransomware', 'scanning', 'xss']
            
            # Fit target encoder with class names
            self.target_encoder.fit(np.array(self.class_names).reshape(-1, 1))
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def _preprocess_flow_data(self, df):
        """
        Preprocess network flow data for the transformer model.
        
        Args:
            df: DataFrame containing network flow data
            
        Returns:
            Preprocessed data ready for the transformer model
        """
        # Ensure all required columns exist
        required_columns = [
            'ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port', 'proto', 'service', 
            'duration', 'src_bytes', 'dst_bytes', 'conn_state', 'src_pkts', 'dst_pkts'
        ]
        
        # Add missing columns with default values
        for col in required_columns:
            if col not in df.columns:
                if col in ['ts', 'duration', 'src_bytes', 'dst_bytes', 'src_port', 'dst_port', 'src_pkts', 'dst_pkts']:
                    df[col] = 0
                else:
                    df[col] = '-'
        
        # Add label and type columns (for compatibility with preprocessing function)
        df['label'] = 0  # Default to normal (will be predicted)
        df['type'] = 'normal'  # Default to normal (will be predicted)
        
        # Extract features similar to the training process
        try:
            # Convert timestamp to datetime and extract features
            df['ts'] = pd.to_numeric(df['ts'], errors='coerce')
            df['ts'] = pd.to_datetime(df['ts'], unit='s', errors='coerce')
            
            # Extract time features
            df['hour'] = df['ts'].dt.hour
            df['day'] = df['ts'].dt.day
            df['day_of_week'] = df['ts'].dt.dayofweek
        except Exception as e:
            print(f"Error processing timestamps: {str(e)}")
            # Create dummy time features
            df['hour'] = 0
            df['day'] = 0
            df['day_of_week'] = 0
        
        # Extract IP features
        try:
            df['src_ip_parsed'] = df['src_ip'].apply(lambda x: int(''.join([i.zfill(3) for i in str(x).split('.')]) if isinstance(x, str) and '.' in str(x) else 0))
            df['dst_ip_parsed'] = df['dst_ip'].apply(lambda x: int(''.join([i.zfill(3) for i in str(x).split('.')]) if isinstance(x, str) and '.' in str(x) else 0))
        except Exception as e:
            print(f"Error processing IP addresses: {str(e)}")
            df['src_ip_parsed'] = 0
            df['dst_ip_parsed'] = 0
        
        # Handle categorical columns
        categorical_cols = ['proto', 'service', 'conn_state']
        for col in categorical_cols:
            df[col] = df[col].fillna('-')
        
        # Handle numeric columns
        numeric_cols = ['duration', 'src_bytes', 'dst_bytes', 'src_pkts', 'dst_pkts', 
                       'src_port', 'dst_port', 'hour', 'day', 'day_of_week',
                       'src_ip_parsed', 'dst_ip_parsed']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Extract features (drop non-feature columns)
        X = df.drop(['ts', 'src_ip', 'dst_ip', 'label', 'type'], axis=1, errors='ignore')
        
        # Separate numerical and categorical features
        categorical_cols = X.select_dtypes(include=['object']).columns
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
        
        # Fit and transform or just transform based on whether encoders are already fitted
        if not hasattr(self.scaler, 'mean_') or self.scaler.mean_ is None:
            # First time preprocessing
            X_num = self.scaler.fit_transform(X[numerical_cols])
            X_cat = self.encoder.fit_transform(X[categorical_cols])
        else:
            # Using pre-fitted encoders
            X_num = self.scaler.transform(X[numerical_cols])
            try:
                X_cat = self.encoder.transform(X[categorical_cols])
            except Exception as e:
                print(f"Error transforming categorical features: {str(e)}")
                # If transformation fails, fit and transform again
                X_cat = self.encoder.fit_transform(X[categorical_cols])
        
        # Prepare data for transformer
        X_transformer = prepare_data_for_transformer(X_num, X_cat)
        
        return X_transformer
    
    def detect_anomalies(self, flow_data, threshold=0.5):
        """
        Detect anomalies in network flow data.
        
        Args:
            flow_data: DataFrame containing network flow data
            threshold: Probability threshold for anomaly detection
            
        Returns:
            Dictionary containing detection results
        """
        start_time = time.time()
        
        # Preprocess the flow data
        X_transformer = self._preprocess_flow_data(flow_data)
        
        # Make predictions
        y_pred_proba = self.model.predict(X_transformer)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Get class names for predictions
        predicted_classes = [self.class_names[i] if i < len(self.class_names) else f"unknown_{i}" for i in y_pred]
        
        # Identify anomalies (anything not classified as 'normal')
        anomaly_indices = [i for i, cls in enumerate(predicted_classes) if cls != 'normal']
        
        # Extract anomalies with their details
        anomalies = []
        for idx in anomaly_indices:
            anomaly = {
                'src_ip': flow_data.iloc[idx]['src_ip'] if 'src_ip' in flow_data.columns else 'unknown',
                'src_port': int(flow_data.iloc[idx]['src_port']) if 'src_port' in flow_data.columns else 0,
                'dst_ip': flow_data.iloc[idx]['dst_ip'] if 'dst_ip' in flow_data.columns else 'unknown',
                'dst_port': int(flow_data.iloc[idx]['dst_port']) if 'dst_port' in flow_data.columns else 0,
                'proto': flow_data.iloc[idx]['proto'] if 'proto' in flow_data.columns else 'unknown',
                'service': flow_data.iloc[idx]['service'] if 'service' in flow_data.columns else 'unknown',
                'type': predicted_classes[idx],
                'confidence': float(y_pred_proba[idx][y_pred[idx]])
            }
            anomalies.append(anomaly)
        
        # Calculate detection time
        detection_time = time.time() - start_time
        avg_detection_time = detection_time / len(flow_data) if len(flow_data) > 0 else 0
        
        return {
            'total_flows': len(flow_data),
            'anomalies': anomalies,
            'detection_time': detection_time,
            'avg_detection_time': avg_detection_time
        }

# Example usage
if __name__ == '__main__':
    # Create an instance of the anomaly detector
    detector = AnomalyDetector()
    
    # Load some sample flow data
    sample_data = pd.read_csv('ads/dataset_ami/test.csv')
    
    # Detect anomalies
    results = detector.detect_anomalies(sample_data)
    
    # Print results
    print(f"Analyzed {results['total_flows']} flows")
    print(f"Found {len(results['anomalies'])} anomalies")
    print(f"Detection time: {results['detection_time']:.4f} seconds")
    print(f"Average detection time per flow: {results['avg_detection_time']*1000:.2f} ms")
    
    # Print anomaly details
    for i, anomaly in enumerate(results['anomalies'][:10]):  # Show first 10 anomalies
        print(f"\nAnomaly {i+1}:")
        print(f"  Type: {anomaly['type']}")
        print(f"  Source: {anomaly['src_ip']}:{anomaly['src_port']}")
        print(f"  Destination: {anomaly['dst_ip']}:{anomaly['dst_port']}")
        print(f"  Protocol: {anomaly['proto']}")
        print(f"  Service: {anomaly['service']}")
        print(f"  Confidence: {anomaly['confidence']:.4f}")