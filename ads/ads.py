import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, LayerNormalization, Dropout # type: ignore
from tensorflow.keras.models import Model # type: ignore
import time
import warnings
import os # Added for directory creation
import joblib # Added for saving objects

warnings.filterwarnings('ignore')

# For reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_and_preprocess_data(file_path):
    """Load and preprocess the TON_IoT dataset."""
    # Define the column names based on the provided headings
    column_names = [
        'ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port', 'proto', 'service',
        'duration', 'src_bytes', 'dst_bytes', 'conn_state', 'missed_bytes',
        'src_pkts', 'src_ip_bytes', 'dst_pkts', 'dst_ip_bytes', 'dns_query',
        'dns_qclass', 'dns_qtype', 'dns_rcode', 'dns_AA', 'dns_RD', 'dns_RA',
        'dns_rejected', 'ssl_version', 'ssl_cipher', 'ssl_resumed',
        'ssl_established', 'ssl_subject', 'ssl_issuer', 'http_trans_depth',
        'http_method', 'http_uri', 'http_referrer', 'http_version',
        'http_request_body_len', 'http_response_body_len', 'http_status_code',
        'http_user_agent', 'http_orig_mime_types', 'http_resp_mime_types',
        'weird_name', 'weird_addl', 'weird_notice', 'label', 'type'
    ]

    # Try to detect the file format
    with open(file_path, 'r') as f:
        first_line = f.readline().strip()

    # Count separators to determine the likely separator
    tab_count = first_line.count('\t')
    comma_count = first_line.count(',')

    if comma_count > tab_count:
        separator = ','
        print(f"Detected comma-separated values (found {comma_count} commas)")
    else:
        separator = '\t'
        print(f"Detected tab-separated values (found {tab_count} tabs)")

    # Check if the first line looks like a header
    has_header = 'ts' in first_line and 'src_ip' in first_line

    try:
        # First attempt: Try reading with the detected separator
        if has_header:
            df = pd.read_csv(file_path, sep=separator, header=0, low_memory=False) # Added low_memory=False
            print("Reading file with header row")
        else:
            df = pd.read_csv(file_path, sep=separator, names=column_names, header=None, low_memory=False)
            print("Reading file without header row")

        # Assign column names if they weren't read from header
        if not has_header and len(df.columns) == len(column_names):
             df.columns = column_names
        elif len(df.columns) != len(column_names):
             print(f"Warning: Column count mismatch. Expected {len(column_names)}, got {len(df.columns)}. Attempting to align...")
             # Try to align based on common names or just use first N names
             df.columns = column_names[:len(df.columns)]


        # Check if we got the expected number of columns
        if len(df.columns) < len(column_names) - 5:  # Allow for some flexibility
            print(f"Warning: Expected around {len(column_names)} columns but got {len(df.columns)}. Trying alternative parsing...")

            # Second attempt: Try reading the file as a single text column and then split
            df_raw = pd.read_csv(file_path, header=None, names=['raw_data'])

            # Split the raw data into columns using the detected separator
            split_data = df_raw['raw_data'].str.split(separator, expand=True)

            # Assign column names (up to the number of columns we have)
            num_cols = min(len(column_names), split_data.shape[1])
            split_data.columns = column_names[:num_cols]

            df = split_data
            print(f"Parsed data into {df.shape[1]} columns using alternative method.")

    except Exception as e:
        print(f"Error reading CSV: {str(e)}")
        print("Attempting alternative parsing method...")

        # Fallback method: Read line by line and parse manually
        rows = []
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if i == 0 and has_header:
                    continue  # Skip header

                # Split the line and create a row
                values = line.strip().split(',') # Assuming comma as fallback
                rows.append(values)

        # Create DataFrame
        df = pd.DataFrame(rows)
        num_parsed_cols = df.shape[1]
        if num_parsed_cols >= len(column_names):
            df.columns = column_names[:num_parsed_cols] # Use actual number of parsed columns
        else:
            # Pad with NaN columns if needed
            df.columns = column_names[:num_parsed_cols] # Name existing columns
            for i in range(num_parsed_cols, len(column_names)):
                df[column_names[i]] = np.nan

        print(f"Manually parsed {len(df)} rows into {len(df.columns)} columns")

    # Print first few rows for debugging
    print("First 2 rows of the dataset:")
    print(df.head(2))

    # --- Data Cleaning and Feature Engineering ---
    # Convert timestamp to datetime and extract features
    try:
        df['ts'] = pd.to_numeric(df['ts'], errors='coerce')
        df['ts'] = pd.to_datetime(df['ts'], unit='s', errors='coerce')

        # Check for NaT values in timestamp
        nat_count = df['ts'].isna().sum()
        if nat_count > 0:
            print(f"Warning: {nat_count} invalid timestamp values found and converted to NaT")

        # Extract time features where timestamp is valid
        df['hour'] = df['ts'].dt.hour.fillna(0).astype(int)
        df['day'] = df['ts'].dt.day.fillna(0).astype(int)
        df['day_of_week'] = df['ts'].dt.dayofweek.fillna(0).astype(int)
    except Exception as e:
        print(f"Error processing timestamps: {str(e)}")
        # Create dummy time features if timestamp processing fails
        df['hour'] = 0
        df['day'] = 0
        df['day_of_week'] = 0

    # Extract IP features (convert IPs to numerical representations)
    def ip_to_int(ip_str):
        try:
            if isinstance(ip_str, str) and '.' in ip_str:
                parts = ip_str.strip().strip("'\"").split('.')
                if len(parts) == 4:
                    return int(''.join([part.zfill(3) for part in parts]))
            return 0
        except:
            return 0

    try:
        df['src_ip_parsed'] = df['src_ip'].apply(ip_to_int)
        df['dst_ip_parsed'] = df['dst_ip'].apply(ip_to_int)
    except Exception as e:
        print(f"Error processing IP addresses: {str(e)}")
        df['src_ip_parsed'] = 0
        df['dst_ip_parsed'] = 0

    # Handle missing values and types
    # Identify numeric and categorical columns based on original definition
    categorical_cols_def = ['proto', 'service', 'conn_state', 'dns_query', 'ssl_version',
                        'ssl_cipher', 'ssl_subject', 'ssl_issuer', 'http_method',
                        'http_uri', 'http_referrer', 'http_version', 'http_user_agent',
                        'http_orig_mime_types', 'http_resp_mime_types', 'weird_name',
                        'weird_addl']

    boolean_cols = ['dns_AA', 'dns_RD', 'dns_RA', 'dns_rejected', 'ssl_resumed',
                    'ssl_established', 'weird_notice']

    # Convert boolean-like columns to integers first
    for col in boolean_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: 1 if str(x).strip().lower() in ['t', 'true', '1'] else 0)
        else:
            print(f"Warning: Boolean column '{col}' not found. Skipping.")


    # Fill missing values and ensure correct types
    for col in df.columns:
        if col in categorical_cols_def:
            df[col] = df[col].fillna('-').astype(str) # Ensure string type
        elif col in boolean_cols:
             df[col] = df[col].fillna(0).astype(int) # Already converted, ensure int
        elif col not in ['ts', 'src_ip', 'dst_ip', 'label', 'type', 'hour', 'day', 'day_of_week', 'src_ip_parsed', 'dst_ip_parsed']:
            # Convert other columns to numeric, fillna with 0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Convert label to numeric if it's not already
    if 'label' in df.columns:
        df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
    else:
         print("Warning: 'label' column not found. Creating dummy label column.")
         df['label'] = 0


    # Ensure 'type' column has string values
    if 'type' in df.columns:
        df['type'] = df['type'].fillna('normal').astype(str)
        # If all values are NaN or empty, assign a default value
        if df['type'].str.strip().eq('').all() or df['type'].str.strip().eq('nan').all():
            print("Warning: 'type' column contains only empty or NaN values. Assigning default value 'normal'.")
            df['type'] = 'normal'
    else:
        print("Warning: 'type' column not found. Creating a default 'normal' type column.")
        df['type'] = 'normal'

    # Extract features and target
    # Drop original timestamp and IP strings, plus label/type
    cols_to_drop = [col for col in ['ts', 'src_ip', 'dst_ip', 'label', 'type'] if col in df.columns]
    X = df.drop(columns=cols_to_drop, errors='ignore')

    y_num = df['label']  # Numerical label
    y_cat = df['type']   # Categorical label

    print(f"Shape of feature set X before preprocessing: {X.shape}")
    print(f"Columns in X: {X.columns.tolist()}")

    return X, y_num, y_cat, df

def preprocess_features(X, y_cat, scaler=None, encoder=None, target_encoder=None):
    """Preprocess features for the model. Fits or transforms based on provided objects."""
    print(f"Preprocessing features for {'fitting' if scaler is None else 'transforming'}...")
    # Separate numerical and categorical features based on dtypes in the current DataFrame X
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64', 'int32']).columns.tolist() # Include int32

    print(f"Identified Numerical Columns ({len(numerical_cols)}): {numerical_cols}")
    print(f"Identified Categorical Columns ({len(categorical_cols)}): {categorical_cols}")


    if scaler is None: # Fitting mode (training data)
        # --- Initialize and Fit Scaler ---
        scaler = StandardScaler()
        print("Fitting StandardScaler...")
        X_num = scaler.fit_transform(X[numerical_cols])
        print("StandardScaler fitted.")

        # --- Initialize and Fit OneHotEncoder ---
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        print("Fitting OneHotEncoder for features...")
        X_cat = encoder.fit_transform(X[categorical_cols])
        print("OneHotEncoder for features fitted.")

        # --- Initialize and Fit Target Encoder ---
        target_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore') # Use ignore for target too? Or raise error?
        print("Fitting OneHotEncoder for target...")
        y_encoded = target_encoder.fit_transform(y_cat.values.reshape(-1, 1))
        print("OneHotEncoder for target fitted.")
        print(f"Target classes found: {target_encoder.categories_[0]}")

        return X_num, X_cat, scaler, encoder, target_encoder, y_encoded

    else: # Transforming mode (validation/test data)
        # --- Transform using existing scaler ---
        print("Transforming numerical features using loaded StandardScaler...")
        try:
             X_num = scaler.transform(X[numerical_cols])
        except ValueError as e:
             print(f"ERROR transforming numerical features: {e}")
             print("Ensure numerical columns match those used for fitting scaler.")
             print("Columns provided:", numerical_cols)
             if hasattr(scaler, 'feature_names_in_'): print("Scaler expected:", scaler.feature_names_in_)
             raise e # Re-raise error

        # --- Transform using existing encoder ---
        print("Transforming categorical features using loaded OneHotEncoder...")
        try:
             X_cat = encoder.transform(X[categorical_cols])
        except ValueError as e:
             print(f"ERROR transforming categorical features: {e}")
             print("Ensure categorical columns match those used for fitting encoder.")
             print("Columns provided:", categorical_cols)
             if hasattr(encoder, 'feature_names_in_'): print("Encoder expected:", encoder.feature_names_in_)
             raise e # Re-raise error

        # --- Transform target using existing target encoder ---
        print("Transforming target labels using loaded OneHotEncoder...")
        try:
            y_encoded = target_encoder.transform(y_cat.values.reshape(-1, 1))
        except ValueError as e:
             print(f"ERROR transforming target labels: {e}")
             print("Ensure target labels contain values known by the fitted target encoder.")
             raise e

        return X_num, X_cat, y_encoded


def transformer_block(inputs, head_size, num_heads, ff_dim, dropout=0):
    """Create a transformer block."""
    # Multi-head attention
    attention_output = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)

    # Add & normalize (first residual connection)
    x = LayerNormalization(epsilon=1e-6)(inputs + attention_output)

    # Feed-forward network
    ff_output = Dense(ff_dim, activation="relu")(x)
    ff_output = Dropout(dropout)(ff_output) # Added dropout here too
    ff_output = Dense(inputs.shape[-1])(ff_output) # Project back to input dim

    # Add & normalize (second residual connection)
    return LayerNormalization(epsilon=1e-6)(x + ff_output)

def build_transformer_model(input_shape, num_classes, head_size=256, num_heads=4, ff_dim=4,
                           num_transformer_blocks=4, mlp_units=[128], dropout=0.2, mlp_dropout=0.4):
    """Build the transformer model for anomaly detection."""
    inputs = Input(shape=input_shape)
    x = inputs

    # Create multiple transformer blocks
    for _ in range(num_transformer_blocks):
        x = transformer_block(x, head_size, num_heads, ff_dim, dropout)

    # Global average pooling
    x = tf.keras.layers.GlobalAveragePooling1D(data_format='channels_first')(x) # Use channels_first if shape is (batch, seq, features)

    # MLP for classification
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)

    outputs = Dense(num_classes, activation="softmax")(x)

    return Model(inputs, outputs)

def prepare_data_for_transformer(X_num, X_cat):
    """Prepare data for the transformer model."""
    # Combine numerical and categorical features
    X_combined = np.hstack([X_num, X_cat])

    # Reshape for transformer: (batch_size, sequence_length=1, features)
    X_reshaped = X_combined.reshape(X_combined.shape[0], 1, X_combined.shape[1])

    return X_reshaped

def train_model(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=64, model_save_path='ads/saved_transformer_model.h5'):
    """Train the transformer model."""
    # Early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
    )

    # Model checkpoint to save the best model
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        model_save_path, monitor='val_loss', save_best_only=True, verbose=1
    )

    # Train the model
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, model_checkpoint]
    )
    training_time = time.time() - start_time

    return model, history, training_time

def evaluate_model(model, X_test, y_test, y_test_cat, target_encoder):
    """Evaluate the model performance."""
    # Predict on test data
    start_time = time.time()
    y_pred_proba = model.predict(X_test)
    detection_time = time.time() - start_time
    avg_detection_time = detection_time / len(X_test)

    y_pred = np.argmax(y_pred_proba, axis=1)
    y_test_indices = np.argmax(y_test, axis=1)

    # Get class names from the target encoder
    class_names = target_encoder.categories_[0]
    print(f"\nClass mapping (Index -> Name): { {i: name for i, name in enumerate(class_names)} }")

    # Print class distribution for debugging
    print("\nClass distribution in test set (True Labels):")
    unique_classes_true, class_counts_true = np.unique(y_test_indices, return_counts=True)
    for cls_idx, count in zip(unique_classes_true, class_counts_true):
        print(f"Class {cls_idx} ({class_names[cls_idx]}): {count} samples")

    print("\nClass distribution in test set (Predicted Labels):")
    unique_classes_pred, class_counts_pred = np.unique(y_pred, return_counts=True)
    for cls_idx, count in zip(unique_classes_pred, class_counts_pred):
         # Handle cases where a class might not be predicted at all
        class_name = class_names[cls_idx] if cls_idx < len(class_names) else f"Unknown Index {cls_idx}"
        print(f"Class {cls_idx} ({class_name}): {count} samples predicted")

    # --- Overall Metrics ---
    # Use classification_report for detailed metrics per class
    report = classification_report(y_test_indices, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    print("\nClassification Report:")
    print(classification_report(y_test_indices, y_pred, target_names=class_names, zero_division=0))

    # Extract overall weighted metrics from the report
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1 = report['weighted avg']['f1-score']

    # --- AUC-ROC ---
    try:
        if len(np.unique(y_test_indices)) > 1 and y_test.shape[1] > 1:
             # Ensure y_test_proba matches shape requirements if needed
             if y_pred_proba.shape[1] != y_test.shape[1]:
                 print(f"Warning: Mismatch between prediction probability shape {y_pred_proba.shape} and test label shape {y_test.shape}. Adjusting probabilities.")
                 # Create a zero array with the target shape
                 temp_proba = np.zeros_like(y_test, dtype=float)
                 # Find common columns based on number of classes predicted vs actual
                 common_cols = min(y_pred_proba.shape[1], y_test.shape[1])
                 # Assign probabilities for common columns
                 temp_proba[:, :common_cols] = y_pred_proba[:, :common_cols]
                 y_pred_proba_adjusted = temp_proba
                 auc_roc = roc_auc_score(y_test, y_pred_proba_adjusted, multi_class='ovr', average='weighted')

             else:
                auc_roc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')

        elif len(np.unique(y_test_indices)) > 1 and y_test.shape[1] == 1: # Binary case encoded as single column? Unlikely with OHE
             print("Warning: Binary classification detected with OHE target? AUC calculation might be incorrect.")
             auc_roc = roc_auc_score(y_test_indices, y_pred_proba[:, 1]) # Assume index 1 is positive class
        else:
            print("Warning: Only one class found in y_test_indices or y_test shape issue. AUC-ROC calculation skipped.")
            auc_roc = float('nan')
    except ValueError as e:
         print(f"Warning: ValueError calculating AUC-ROC: {str(e)}. Setting AUC-ROC to NaN.")
         # Example: "Only one class present in y_true. ROC AUC score is not defined in that case."
         auc_roc = float('nan')
    except Exception as e:
        print(f"Warning: Generic error calculating AUC-ROC: {str(e)}. Setting AUC-ROC to NaN.")
        auc_roc = float('nan')


    # --- Confusion Matrix ---
    cm = confusion_matrix(y_test_indices, y_pred)
    print("\nConfusion Matrix:")
    # Plotting the confusion matrix for better visualization
    plt.figure(figsize=(10, 8)) # Adjusted size
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right') # Rotate labels if long
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('ads/confusion_matrix.png') # Save in ads directory
    print("Confusion Matrix saved as 'ads/confusion_matrix.png'")
    # plt.show() # Optional: show plot immediately

    # --- False Alarm Rate (FAR) ---
    # Calculated as the rate at which 'normal' instances are misclassified as 'attack'
    # FAR = False Positives for Attack / Total Actual Normal = 1 - Recall of Normal Class
    try:
        normal_class_name = 'normal' # Make sure this matches the name in your dataset/encoder
        if normal_class_name in report:
            recall_normal = report[normal_class_name]['recall']
            false_alarm_rate = 1.0 - recall_normal
        else:
            print(f"Warning: '{normal_class_name}' class not found in classification report. Cannot calculate FAR.")
            false_alarm_rate = float('nan')
    except Exception as e:
         print(f"Warning: Error calculating FAR: {str(e)}")
         false_alarm_rate = float('nan')

    # Prepare per-attack metrics from the report
    attack_metrics = {}
    for class_name, metrics_dict in report.items():
        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
             # Get count from support
            count = int(metrics_dict.get('support', 0)) # Ensure count is integer
            attack_metrics[class_name] = {
                'precision': metrics_dict['precision'],
                'recall': metrics_dict['recall'],
                'f1': metrics_dict['f1-score'],
                'count': count
            }


    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc,
        'false_alarm_rate': false_alarm_rate,
        'avg_detection_time': avg_detection_time,
        'attack_metrics': attack_metrics,
        'classification_report': report # Return the full report dict too
    }

def main():
    """Main function to execute the anomaly detection pipeline."""
    # Define paths for saving objects
    ads_dir = 'ads'
    scaler_path = os.path.join(ads_dir, 'scaler.joblib')
    encoder_path = os.path.join(ads_dir, 'encoder.joblib')
    target_encoder_path = os.path.join(ads_dir, 'target_encoder.joblib')
    model_save_path = os.path.join(ads_dir, 'saved_transformer_model.h5')
    history_plot_path = os.path.join(ads_dir, 'transformer_training_history.png')

    # Ensure the 'ads' directory exists
    os.makedirs(ads_dir, exist_ok=True)
    print(f"Ensured directory '{ads_dir}' exists.")


    # Load and preprocess data
    file_path = 'ads/dataset_ami/test.csv' # Make sure this path is correct
    print(f"\nLoading data from {file_path}...")

    try:
        X, y_num, y_cat, df = load_and_preprocess_data(file_path)
        print(f"Data loaded successfully. Original dataset shape: {df.shape}")

        # Analyze class distribution
        print("\nClass distribution in original dataset:")
        class_counts = y_cat.value_counts()
        for cls, count in class_counts.items():
            print(f"{cls}: {count} samples ({count/len(y_cat)*100:.2f}%)")

        # Get unique attack types
        attack_types = y_cat.unique()
        print(f"\nDetected attack types: {list(attack_types)}") # Print as list for clarity

        # Check if we have enough classes for meaningful classification
        if len(attack_types) < 2:
            print("\nERROR: Only one class detected in the dataset. Multi-class classification requires at least two classes.")
            print("Please check your dataset or data loading process.")
            return # Stop execution if only one class

        # --- Undersampling ---
        # NOTE: Undersampling significantly reduces dataset size to balance classes.
        print("\nPerforming undersampling to balance classes...")
        if len(attack_types) > 1:
            min_class_count = class_counts.min()
            # Cap larger classes at a multiple of the smallest, e.g., 3x or 5x
            undersample_multiplier = 3
            target_sample_size = min_class_count * undersample_multiplier
            print(f"Smallest class count: {min_class_count}. Aiming for max {target_sample_size} samples for larger classes.")

            balanced_df_list = []
            for attack_type in attack_types:
                class_df = df[df['type'] == attack_type].copy()
                if len(class_df) > target_sample_size :
                    sampled_df = class_df.sample(n=target_sample_size, random_state=42)
                    balanced_df_list.append(sampled_df)
                    print(f"Undersampled '{attack_type}' from {len(class_df)} to {len(sampled_df)}")
                else:
                    balanced_df_list.append(class_df)
                    print(f"Kept all {len(class_df)} samples for '{attack_type}'")

            balanced_df = pd.concat(balanced_df_list)

            print(f"\nBalanced dataset shape: {balanced_df.shape} (original: {df.shape})")
            df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True) # Shuffle the balanced dataset

            # Update X, y_num, y_cat after balancing
            cols_to_drop = [col for col in ['ts', 'src_ip', 'dst_ip', 'label', 'type'] if col in df.columns]
            X = df.drop(columns=cols_to_drop, errors='ignore')
            y_num = df['label'] if 'label' in df.columns else None
            y_cat = df['type'] if 'type' in df.columns else None

            # Show updated class distribution
            print("\nUpdated class distribution after balancing:")
            updated_counts = y_cat.value_counts()
            for cls, count in updated_counts.items():
                print(f"{cls}: {count} samples ({count/len(y_cat)*100:.2f}%)")

        # Use stratified split to maintain class distribution in train/val/test
        print("\nPerforming stratified train/validation/test split...")
        try:
            X_train, X_temp, y_cat_train, y_cat_temp = train_test_split(
                X, y_cat, test_size=0.3, random_state=42, stratify=y_cat
            )

            X_val, X_test, y_cat_val, y_cat_test = train_test_split(
                X_temp, y_cat_temp, test_size=0.5, random_state=42, stratify=y_cat_temp # 0.5 of 0.3 = 0.15 test
            )
        except ValueError as e:
             print(f"\nERROR during stratified split: {e}")
             print("This usually happens if a class has only 1 sample after balancing.")
             print("Consider adjusting undersampling or using the full dataset.")
             return


        print(f"Train set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")

        # Verify class distribution in splits
        print("\nClass distribution in train set:")
        train_counts = y_cat_train.value_counts(normalize=True) * 100
        print(train_counts.round(2))

        print("\nClass distribution in test set:")
        test_counts = y_cat_test.value_counts(normalize=True) * 100
        print(test_counts.round(2))

        # Preprocess features - Fit on Training data
        print("\nPreprocessing features (fitting on training data)...")
        X_num_train, X_cat_train, scaler, encoder, target_encoder, y_train_encoded = preprocess_features(X_train, y_cat_train)

        # --- Save preprocessing objects ---
        try:
            print(f"\nSaving fitted scaler to {scaler_path}")
            joblib.dump(scaler, scaler_path)

            print(f"Saving fitted encoder to {encoder_path}")
            joblib.dump(encoder, encoder_path)

            print(f"Saving fitted target encoder to {target_encoder_path}")
            joblib.dump(target_encoder, target_encoder_path)
            print("Preprocessing objects saved successfully.")
        except Exception as e:
            print(f"ERROR saving preprocessing objects: {e}")
            # Optionally, decide whether to continue training without saving
            # return # Or raise an error

        # Preprocess Validation and Test data - Transform only
        print("\nPreprocessing features (transforming validation data)...")
        X_num_val, X_cat_val, y_val_encoded = preprocess_features(X_val, y_cat_val, scaler, encoder, target_encoder)
        print("\nPreprocessing features (transforming test data)...")
        X_num_test, X_cat_test, y_test_encoded = preprocess_features(X_test, y_cat_test, scaler, encoder, target_encoder)


        # Prepare data for transformer
        print("\nPreparing data for transformer model...")
        X_train_transformer = prepare_data_for_transformer(X_num_train, X_cat_train)
        X_val_transformer = prepare_data_for_transformer(X_num_val, X_cat_val)
        X_test_transformer = prepare_data_for_transformer(X_num_test, X_cat_test)

        # Build and compile the model
        print("\nBuilding and compiling the transformer model...")
        input_shape = X_train_transformer.shape[1:] # Should be (1, num_features)
        num_classes = y_train_encoded.shape[1]

        # Hyperparameters - Adjusted based on the potentially smaller dataset
        model = build_transformer_model(
            input_shape=input_shape,
            num_classes=num_classes,
            head_size=64,   # Reduced head size
            num_heads=4,    # Reduced num heads
            ff_dim=128,     # Reduced feed-forward dimension
            num_transformer_blocks=2, # Reduced blocks
            mlp_units=[64], # Simplified MLP
            dropout=0.1,    # Reduced dropout
            mlp_dropout=0.2 # Reduced MLP dropout
        )

        # Print model summary
        model.summary()

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), # Reduced learning rate
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Train the model
        print("\nTraining the model...")
        # Note: Training runs for 'epochs' unless 'EarlyStopping' triggers sooner based on 'val_loss' improvement.
        model, history, training_time = train_model(
            model, X_train_transformer, y_train_encoded,
            X_val_transformer, y_val_encoded,
            epochs=25,      # Increased epochs slightly, let early stopping decide
            batch_size=32,   # Adjusted batch size
            model_save_path=model_save_path # Pass save path
        )

        # Load the best model saved by ModelCheckpoint for evaluation
        print("\nLoading best model weights from checkpoint...")
        try:
            # Ensure model is built before loading weights if loading into a fresh instance
            # model = build_transformer_model(...) # Rebuild if necessary
            # model.compile(...) # Recompile if necessary
            model.load_weights(model_save_path)
        except Exception as e:
            print(f"Warning: Could not load weights from '{model_save_path}'. Using the model state from the end of training. Error: {e}")


        # Evaluate the model
        print("\nEvaluating the model on the test set...")
        metrics = evaluate_model(model, X_test_transformer, y_test_encoded, y_cat_test, target_encoder) # Pass target_encoder

        # Print results
        print("\n--- Evaluation Results ---")
        print(f"Overall Precision (Weighted): {metrics['precision']:.4f}")
        print(f"Overall Recall (Weighted): {metrics['recall']:.4f}")
        print(f"Overall F1 Score (Weighted): {metrics['f1']:.4f}")
        print(f"AUC-ROC (One-vs-Rest, Weighted): {metrics['auc_roc']:.4f}")
        print(f"False Alarm Rate (Normal Misclassified as Attack): {metrics['false_alarm_rate']:.4f}")
        print(f"Average Detection Time per Sample: {metrics['avg_detection_time']*1000:.2f} ms")
        print(f"Total Training Time: {training_time:.2f} seconds")

        print("\n--- Metrics Per Class ---")
        # Find max length for alignment
        max_len = max(len(k) for k in metrics['attack_metrics'].keys()) if metrics['attack_metrics'] else 10
        print(f"{'Attack Type'.ljust(max_len)} | Count | Precision | Recall | F1-Score")
        print("-" * (max_len + 36))
        for attack_type, attack_metric in metrics['attack_metrics'].items():
             print(f"{attack_type.ljust(max_len)} | {str(attack_metric['count']).rjust(5)} | {attack_metric['precision']:9.4f} | {attack_metric['recall']:6.4f} | {attack_metric['f1']:8.4f}")


        # Plot training history
        print("\nGenerating training history plots...")
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(history_plot_path)
        print(f"Plot saved as '{history_plot_path}'")
        # plt.show() # Optional

    except FileNotFoundError:
        print(f"ERROR: Dataset file not found at {file_path}")
    except KeyError as e:
        print(f"ERROR: Missing expected column in the dataset: {e}")
        print("Please ensure the CSV file has the correct columns and format.")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging

if __name__ == "__main__":
    main()
