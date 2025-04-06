import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, LayerNormalization, Dropout
from tensorflow.keras.models import Model
import time
import warnings
import joblib # Import joblib for saving/loading sklearn objects
import os     # Import os for creating directories

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
    try:
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
    except FileNotFoundError:
        raise # Re-raise the error if the file doesn't exist

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
            df = pd.read_csv(file_path, sep=separator, header=0)
            print("Reading file with header row")
        else:
            df = pd.read_csv(file_path, sep=separator, names=column_names, header=None)
            print("Reading file without header row")

        # Check if we got the expected number of columns
        if len(df.columns) < len(column_names) - 5:  # Allow for some flexibility
            print(f"Warning: Expected around {len(column_names)} columns but got {len(df.columns)}. Trying alternative parsing...")

            # Second attempt: Try reading the file as a single text column and then split
            df = pd.read_csv(file_path, header=None, names=['raw_data'])

            # Split the raw data into columns
            split_data = df['raw_data'].str.split(',', expand=True)

            # Assign column names (up to the number of columns we have)
            num_cols = min(len(column_names), split_data.shape[1])
            split_data.columns = column_names[:num_cols]

            df = split_data
            print(f"Parsed data into {df.shape[1]} columns")

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
                values = line.strip().split(',')
                rows.append(values)

        # Create DataFrame
        df = pd.DataFrame(rows)
        if len(df.columns) >= len(column_names):
            df.columns = column_names[:len(df.columns)]
        else:
            # Pad with NaN columns if needed
            for i in range(len(df.columns), len(column_names)):
                df[column_names[i]] = np.nan

        print(f"Manually parsed {len(df)} rows into {len(df.columns)} columns")

    # Print first few rows for debugging
    print("First 2 rows of the dataset:")
    print(df.head(2))

    # Convert timestamp to datetime and extract features
    try:
        df['ts'] = pd.to_numeric(df['ts'], errors='coerce')
        df['ts'] = pd.to_datetime(df['ts'], unit='s', errors='coerce')

        # Check for NaT values in timestamp
        nat_count = df['ts'].isna().sum()
        if nat_count > 0:
            print(f"Warning: {nat_count} invalid timestamp values found and converted to NaT")

        # Extract time features where timestamp is valid
        df['hour'] = df['ts'].dt.hour
        df['day'] = df['ts'].dt.day
        df['day_of_week'] = df['ts'].dt.dayofweek
    except Exception as e:
        print(f"Error processing timestamps: {str(e)}")
        # Create dummy time features if timestamp processing fails
        df['hour'] = 0
        df['day'] = 0
        df['day_of_week'] = 0

    # Extract IP features (convert IPs to numerical representations)
    try:
        df['src_ip_parsed'] = df['src_ip'].apply(lambda x: int(''.join([i.zfill(3) for i in str(x).split('.')]) if isinstance(x, str) and '.' in str(x) else 0))
        df['dst_ip_parsed'] = df['dst_ip'].apply(lambda x: int(''.join([i.zfill(3) for i in str(x).split('.')]) if isinstance(x, str) and '.' in str(x) else 0))
    except Exception as e:
        print(f"Error processing IP addresses: {str(e)}")
        df['src_ip_parsed'] = 0
        df['dst_ip_parsed'] = 0

    # Handle missing values
    # Identify numeric and categorical columns
    categorical_cols = ['proto', 'service', 'conn_state', 'dns_query', 'ssl_version',
                        'ssl_cipher', 'ssl_subject', 'ssl_issuer', 'http_method',
                        'http_uri', 'http_referrer', 'http_version', 'http_user_agent',
                        'http_orig_mime_types', 'http_resp_mime_types', 'weird_name',
                        'weird_addl']

    # Convert boolean-like columns to integers
    boolean_cols = ['dns_AA', 'dns_RD', 'dns_RA', 'dns_rejected', 'ssl_resumed',
                    'ssl_established', 'weird_notice']
    for col in boolean_cols:
        if col in df.columns:
            # Handle potential boolean values directly and coerce others
            df[col] = df[col].apply(lambda x: 1 if str(x).lower() in ['t', 'true', '1'] else 0)


    # Fill missing values
    for col in df.columns:
        if col in categorical_cols:
            # Ensure string type before filling NA, then fill NA
            df[col] = df[col].astype(str).fillna('-')
        elif col not in ['ts', 'src_ip', 'dst_ip', 'label', 'type']:
             # Ensure numeric conversion, filling errors and NaNs with 0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)


    # Convert label to numeric if it's not already
    df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int) # Ensure integer type

    # Ensure 'type' column has string values
    if 'type' in df.columns:
        df['type'] = df['type'].fillna('normal').astype(str)
        # If all values are NaN or empty, assign a default value
        if df['type'].str.strip().eq('').all() or df['type'].str.strip().str.lower().eq('nan').all():
            print("Warning: 'type' column contains only empty or NaN values. Assigning default value 'normal'.")
            df['type'] = 'normal'
    else:
        print("Warning: 'type' column not found. Creating a default 'normal' type column.")
        df['type'] = 'normal'

    # Extract features and target
    # Make sure to drop columns only if they exist
    cols_to_drop = [col for col in ['ts', 'src_ip', 'dst_ip', 'label', 'type'] if col in df.columns]
    X = df.drop(columns=cols_to_drop, errors='ignore')

    y_num = df['label'] if 'label' in df.columns else None # Numerical label
    y_cat = df['type'] if 'type' in df.columns else None # Categorical label

    if y_cat is None:
        raise ValueError("Target column 'type' could not be found or generated.")

    return X, y_num, y_cat, df

def preprocess_features(X, y_cat, train_indices=None):
    """Preprocess features for the model."""
    # Separate numerical and categorical features
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns # Include category type
    numerical_cols = X.select_dtypes(include=np.number).columns # Use numpy number types

    print(f"Identified {len(numerical_cols)} numerical columns and {len(categorical_cols)} categorical columns.")

    # Initialize encoders
    if train_indices is None:
        print("-" * 30)
        print("Features used for SCALING (Training):")
        print(list(numerical_cols)) # Print the list
        print(f"Total: {len(numerical_cols)}")
        print("-" * 30)
        print("Features used for ENCODING (Training):")
        print(list(categorical_cols)) # Print the list
        print(f"Total: {len(categorical_cols)}")
        print("-" * 30)
        # First time preprocessing (training data)
        print("Fitting scaler and encoders on training data.")
        scaler = StandardScaler()
        # Use handle_unknown='ignore' for robustness, sparse_output=False for dense array
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        # Fit and transform numerical features
        X_num = scaler.fit_transform(X[numerical_cols])
        print(f"Scaled numerical features shape: {X_num.shape}")

        # Fit and transform categorical features
        if not categorical_cols.empty:
            X_cat = encoder.fit_transform(X[categorical_cols])
            print(f"Encoded categorical features shape: {X_cat.shape}")
        else:
            X_cat = np.empty((X.shape[0], 0)) # Create empty array if no categorical features
            print("No categorical features found to encode.")


        # Also encode the target
        target_encoder = OneHotEncoder(sparse_output=False)
        y_encoded = target_encoder.fit_transform(y_cat.values.reshape(-1, 1))
        print(f"Encoded target variable shape: {y_encoded.shape}")
        print(f"Target classes found by encoder: {target_encoder.categories_}")


        return X_num, X_cat, scaler, encoder, target_encoder, y_encoded
    else:
        # Using pre-fitted encoders (validation/test data)
        print("Transforming validation/test data using fitted scaler and encoders.")
        scaler, encoder, target_encoder = train_indices

        # Transform numerical features
        X_num = scaler.transform(X[numerical_cols])

        # Transform categorical features
        if not categorical_cols.empty:
             # Check if encoder was fitted (i.e., if there were categorical cols in training)
            if hasattr(encoder, 'categories_') and encoder.categories_:
                X_cat = encoder.transform(X[categorical_cols])
            else: # If no encoder was fitted during training
                X_cat = np.empty((X.shape[0], 0))
        else:
            X_cat = np.empty((X.shape[0], 0))

        # Also encode the target
        y_encoded = target_encoder.transform(y_cat.values.reshape(-1, 1))

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
    ff_output = Dense(inputs.shape[-1])(ff_output) # Project back to input dim
    ff_output = Dropout(dropout)(ff_output) # Apply dropout after FF

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

    # Global average pooling - applies pooling across the sequence dimension
    # Result shape: (batch_size, features)
    x = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_last")(x)

    # MLP for classification
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)

    outputs = Dense(num_classes, activation="softmax")(x)

    return Model(inputs, outputs)

def prepare_data_for_transformer(X_num, X_cat):
    """Prepare data for the transformer model."""
    # Combine numerical and categorical features
    if X_cat.size > 0: # Check if X_cat is not empty
        X_combined = np.hstack([X_num, X_cat])
    else:
        X_combined = X_num # Only use numerical if no categorical

    # Reshape for transformer: (batch_size, sequence_length, features)
    # For tabular data, we treat the combined features as a single time step in a sequence.
    # Sequence length is 1, number of features is X_combined.shape[1]
    X_reshaped = X_combined.reshape(X_combined.shape[0], 1, X_combined.shape[1])
    print(f"Reshaped data for transformer: {X_reshaped.shape}")

    return X_reshaped

def train_model(model, X_train, y_train, X_val, y_val, output_dir, epochs=20, batch_size=64):
    """Train the transformer model."""
    # Early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
    )

    # Model checkpoint to save the best model
    model_checkpoint_path = os.path.join(output_dir, 'saved_transformer_model.h5')
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        model_checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1
    )
    print(f"Model checkpoints will be saved to: {model_checkpoint_path}")


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

def evaluate_model(model, X_test, y_test, y_test_cat, target_encoder, output_dir):
    """Evaluate the model performance."""
    # Predict on test data
    start_time = time.time()
    y_pred_proba = model.predict(X_test)
    detection_time = time.time() - start_time
    avg_detection_time = detection_time / len(X_test) if len(X_test) > 0 else 0

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
             # Ensure y_pred_proba matches shape requirements if needed
             if y_pred_proba.shape[1] != y_test.shape[1]:
                 print(f"Warning: Mismatch between prediction probability shape {y_pred_proba.shape} and test label shape {y_test.shape}. AUC might be inaccurate.")
                 # Pad prediction probabilities if needed (common issue if some classes are never predicted)
                 temp_proba = np.zeros_like(y_test, dtype=float)
                 max_pred_idx = y_pred_proba.shape[1]
                 cols_to_use = min(max_pred_idx, y_test.shape[1])
                 temp_proba[:,:cols_to_use] = y_pred_proba[:,:cols_to_use]
                 y_pred_proba_adjusted = temp_proba
                 auc_roc = roc_auc_score(y_test, y_pred_proba_adjusted, multi_class='ovr')
             else:
                auc_roc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')

        elif len(np.unique(y_test_indices)) > 1 and y_test.shape[1] == 1: # Binary case encoded as single column (unlikely with OneHotEncoder)
             auc_roc = roc_auc_score(y_test_indices, y_pred_proba[:, 1]) # Assuming positive class is index 1
        else:
            print("Warning: Only one class found in y_test_indices or y_test shape issue. AUC-ROC calculation skipped.")
            auc_roc = float('nan')
    except ValueError as e:
         print(f"Warning: ValueError calculating AUC-ROC: {str(e)}. Setting AUC-ROC to NaN.")
         auc_roc = float('nan')
    except Exception as e:
        print(f"Warning: Generic error calculating AUC-ROC: {str(e)}. Setting AUC-ROC to NaN.")
        auc_roc = float('nan')


    # --- Confusion Matrix ---
    cm = confusion_matrix(y_test_indices, y_pred, labels=range(len(class_names))) # Ensure labels match expected range
    print("\nConfusion Matrix:")
    # Plotting the confusion matrix for better visualization
    plt.figure(figsize=(10, 8)) # Adjusted size slightly
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right') # Rotate labels if they overlap
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"Confusion Matrix saved as '{cm_path}'")
    plt.close() # Close the plot

    # --- False Alarm Rate (FAR) ---
    # Calculated as the rate at which 'normal' instances are misclassified as 'attack'
    # FAR = False Positives for Attack Classes / Total Actual Normal = 1 - Recall of Normal Class
    try:
        # Find the index corresponding to the 'normal' class
        normal_class_name = 'normal' # Assuming this is the name
        if normal_class_name in class_names:
            normal_class_metrics = report.get(normal_class_name, None)
            if normal_class_metrics:
                recall_normal = normal_class_metrics['recall']
                false_alarm_rate = 1.0 - recall_normal
            else:
                 print(f"Warning: Metrics for '{normal_class_name}' class not found in report. Cannot calculate FAR.")
                 false_alarm_rate = float('nan')
        else:
            print(f"Warning: '{normal_class_name}' class not found in target encoder categories. Cannot calculate FAR.")
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
    # Define input file path and output directory
    file_path = 'ads/dataset_ami/test.csv' # Make sure this path is correct
    output_dir = 'ads/results'

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Ensured output directory exists: {output_dir}")

    print(f"Loading data from {file_path}...")

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
        print("\nPerforming undersampling to balance classes (heuristic: target ~3x smallest)...")
        if len(attack_types) > 1:
            min_class_count = class_counts.min()
            # Target sample size: aim for 3x the minority count for larger classes, but don't upsample minority
            target_sample_size_heuristic = min_class_count * 3
            print(f"Smallest class count: {min_class_count}. Aiming for samples around {target_sample_size_heuristic} for larger classes.")

            balanced_df_list = []
            for attack_type in attack_types:
                class_df = df[df['type'] == attack_type].copy()
                current_count = len(class_df)

                # Only undersample if the class is larger than the heuristic target size
                # AND significantly larger than the minimum count (to avoid over-sampling tiny majority classes)
                if current_count > target_sample_size_heuristic and current_count > min_class_count :
                    # Undersample to the heuristic size
                    sampled_df = class_df.sample(n=target_sample_size_heuristic, random_state=42)
                    balanced_df_list.append(sampled_df)
                    print(f"Undersampled '{attack_type}' from {current_count} to {len(sampled_df)}")
                else:
                    balanced_df_list.append(class_df)
                    print(f"Kept all {current_count} samples for '{attack_type}'")

            balanced_df = pd.concat(balanced_df_list)

            print(f"\nBalanced dataset shape: {balanced_df.shape} (original: {df.shape})")
            df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True) # Shuffle the balanced dataset

            # Update X, y_num, y_cat after balancing
            # Ensure all original columns exist before dropping
            cols_to_drop = [col for col in ['ts', 'src_ip', 'dst_ip', 'label', 'type'] if col in df.columns]
            X = df.drop(columns=cols_to_drop, errors='ignore')
            y_num = df['label'] if 'label' in df.columns else None
            y_cat = df['type'] if 'type' in df.columns else None

            # Show updated class distribution
            print("\nUpdated class distribution after balancing:")
            updated_counts = y_cat.value_counts()
            for cls, count in updated_counts.items():
                print(f"{cls}: {count} samples ({count/len(y_cat)*100:.2f}%)")
        else:
            print("Skipping undersampling as there are not enough classes.")


        # Use stratified split to maintain class distribution in train/val/test
        print("\nPerforming stratified train/validation/test split...")
        # Check if dataset size is too small for the split ratios
        if len(X) < 10: # Arbitrary small number check
             print("ERROR: Dataset too small for reliable train/val/test split after potential undersampling.")
             return

        # Ensure there are enough samples in the smallest class for stratification
        min_samples_per_class = y_cat.value_counts().min()
        required_samples_for_split = 2 # At least 1 for train, 1 for temp set

        # Adjust test_size if needed for stratification (especially relevant after undersampling)
        test_size_main = 0.3
        test_size_sub = 0.5 # for val/test split from temp
        if min_samples_per_class < required_samples_for_split / (1 - test_size_main):
            print(f"Warning: Smallest class ({min_samples_per_class} samples) might be too small for 30% test split stratification. Adjusting split if necessary or consider larger dataset/different balancing.")
            # Could potentially reduce test_size or handle differently, but for now proceed with warning.

        X_train, X_temp, y_cat_train, y_cat_temp = train_test_split(
            X, y_cat, test_size=test_size_main, random_state=42, stratify=y_cat
        )

        # Check if temp set is large enough for the next split
        min_samples_temp = y_cat_temp.value_counts().min()
        if min_samples_temp < required_samples_for_split:
             print(f"Warning: Smallest class in temporary set ({min_samples_temp} samples) is too small for 50% validation/test split stratification. Test set might be very small or lack some classes.")
             # Adjust sub-split if needed, e.g., use smaller test_size_sub or pool smallest classes if appropriate
             if len(X_temp) > 1: # Need at least 2 samples to split
                 X_val, X_test, y_cat_val, y_cat_test = train_test_split(
                     X_temp, y_cat_temp, test_size=test_size_sub, random_state=42, stratify=y_cat_temp
                 )
             else:
                 print("ERROR: Cannot split temporary set further due to insufficient samples.")
                 # Handle this case: maybe use temp set as validation only?
                 X_val, y_cat_val = X_temp, y_cat_temp
                 X_test, y_cat_test = pd.DataFrame(), pd.Series(dtype=str) # Empty test set
                 print("Using entire temporary set as validation, test set will be empty.")

        else:
            X_val, X_test, y_cat_val, y_cat_test = train_test_split(
                X_temp, y_cat_temp, test_size=test_size_sub, random_state=42, stratify=y_cat_temp
            )


        print(f"Train set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")

        # Verify class distribution in splits
        print("\nClass distribution in train set (%):")
        train_counts = y_cat_train.value_counts(normalize=True) * 100
        print(train_counts.round(2))

        print("\nClass distribution in validation set (%):")
        val_counts = y_cat_val.value_counts(normalize=True) * 100
        print(val_counts.round(2))

        if not X_test.empty:
            print("\nClass distribution in test set (%):")
            test_counts = y_cat_test.value_counts(normalize=True) * 100
            print(test_counts.round(2))
        else:
            print("\nTest set is empty.")
            return # Cannot proceed without a test set

        # Preprocess features and save scalers/encoders
        print("\nPreprocessing features...")
        X_num_train, X_cat_train, scaler, encoder, target_encoder, y_train_encoded = preprocess_features(X_train, y_cat_train)

        # Save the fitted scaler and encoders
        scaler_path = os.path.join(output_dir, 'scaler.joblib')
        encoder_path = os.path.join(output_dir, 'encoder.joblib')
        target_encoder_path = os.path.join(output_dir, 'target_encoder.joblib')

        joblib.dump(scaler, scaler_path)
        print(f"Saved scaler to {scaler_path}")
         # Check if encoder was actually fitted (i.e., categorical features existed)
        if hasattr(encoder, 'categories_') and encoder.categories_:
            joblib.dump(encoder, encoder_path)
            print(f"Saved feature encoder to {encoder_path}")
        else:
            print("No feature encoder fitted (no categorical columns in training data), skipping save.")

        joblib.dump(target_encoder, target_encoder_path)
        print(f"Saved target encoder to {target_encoder_path}")


        # Preprocess validation and test sets using the *fitted* objects
        X_num_val, X_cat_val, y_val_encoded = preprocess_features(X_val, y_cat_val, (scaler, encoder, target_encoder))
        X_num_test, X_cat_test, y_test_encoded = preprocess_features(X_test, y_cat_test, (scaler, encoder, target_encoder))

        # Prepare data for transformer
        print("\nPreparing data for transformer model...")
        X_train_transformer = prepare_data_for_transformer(X_num_train, X_cat_train)
        X_val_transformer = prepare_data_for_transformer(X_num_val, X_cat_val)
        X_test_transformer = prepare_data_for_transformer(X_num_test, X_cat_test)

        # Build and compile the model
        print("\nBuilding and compiling the transformer model...")
        input_shape = X_train_transformer.shape[1:] # Should be (1, num_features)
        num_classes = y_train_encoded.shape[1]
        print(f"Model input shape: {input_shape}, Number of output classes: {num_classes}")


        # Hyperparameters - Adjusted based on the potentially smaller dataset
        model = build_transformer_model(
            input_shape=input_shape,
            num_classes=num_classes,
            head_size=64,   # Reduced head size
            num_heads=2,    # Reduced number of heads
            ff_dim=128,     # Feed-forward dimension
            num_transformer_blocks=2, # Reduced number of blocks
            mlp_units=[64], # Simplified MLP head
            dropout=0.1,    # Slightly reduced dropout
            mlp_dropout=0.2 # Slightly reduced MLP dropout
        )

        # Print model summary
        model.summary()

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Train the model
        print("\nTraining the model...")
        model, history, training_time = train_model(
            model, X_train_transformer, y_train_encoded,
            X_val_transformer, y_val_encoded,
            output_dir=output_dir, # Pass output directory for checkpoint saving
            epochs=25,      # Increased epochs slightly, let early stopping decide
            batch_size=32   # Slightly larger batch size if memory allows
        )

        # Load the best model saved by ModelCheckpoint for evaluation
        print("\nLoading best model weights from checkpoint...")
        model_load_path = os.path.join(output_dir, 'saved_transformer_model.h5')
        try:
            # Only load weights if the file exists
            if os.path.exists(model_load_path):
                model.load_weights(model_load_path)
                print(f"Successfully loaded best weights from {model_load_path}")
            else:
                 print(f"Warning: Checkpoint file '{model_load_path}' not found. Using the model state from the end of training.")
        except Exception as e:
            print(f"Warning: Could not load weights from '{model_load_path}'. Using the model state from the end of training. Error: {e}")


        # Evaluate the model
        print("\nEvaluating the model on the test set...")
        metrics = evaluate_model(model, X_test_transformer, y_test_encoded, y_cat_test, target_encoder, output_dir) # Pass target_encoder and output_dir

        # Print results
        print("\n--- Evaluation Results ---")
        print(f"Overall Precision (Weighted): {metrics['precision']:.4f}")
        print(f"Overall Recall (Weighted): {metrics['recall']:.4f}")
        print(f"Overall F1 Score (Weighted): {metrics['f1']:.4f}")
        print(f"AUC-ROC (One-vs-Rest): {metrics['auc_roc']:.4f}")
        print(f"False Alarm Rate (Normal Misclassified): {metrics['false_alarm_rate']:.4f}")
        print(f"Average Detection Time per Sample: {metrics['avg_detection_time']*1000:.2f} ms")
        print(f"Total Training Time: {training_time:.2f} seconds")

        print("\n--- Metrics Per Class ---")
        # Find max length for alignment
        max_len = max(len(k) for k in metrics['attack_metrics'].keys()) if metrics['attack_metrics'] else 10 # Default length
        header = f"{'Attack Type'.ljust(max_len)} | {'Count'.rjust(7)} | {'Precision'.rjust(9)} | {'Recall'.rjust(6)} | {'F1-Score'.rjust(8)}"
        print(header)
        print("-" * len(header))
        for attack_type, attack_metric in metrics['attack_metrics'].items():
             print(f"{attack_type.ljust(max_len)} | {str(attack_metric['count']).rjust(7)} | {attack_metric['precision']:9.4f} | {attack_metric['recall']:6.4f} | {attack_metric['f1']:8.4f}")


        # Plot training history
        if history is not None:
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
            history_plot_path = os.path.join(output_dir, 'transformer_training_history.png')
            plt.savefig(history_plot_path)
            print(f"Plot saved as '{history_plot_path}'")
            plt.close() # Close the plot
        else:
             print("\nSkipping training history plot generation as training did not complete successfully.")


    except FileNotFoundError:
        print(f"ERROR: Dataset file not found at {file_path}")
    except KeyError as e:
        print(f"ERROR: Missing expected column in the dataset: {e}")
        print("Please ensure the CSV file has the correct columns and format, or adjust column names in the script.")
    except ValueError as e:
        print(f"ERROR: A data-related value error occurred: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging

if __name__ == "__main__":
    main()