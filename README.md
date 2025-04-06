      
# AMI Smart Grid SDN with Transformer-based Anomaly Detection

## Description

This project implements a Software-Defined Networking (SDN) solution tailored for Advanced Metering Infrastructure (AMI) in smart grids. It integrates a sophisticated anomaly detection system (ADS) utilizing a Transformer-based machine learning model to identify potential security threats within the network traffic.

The system simulates an AMI environment using Containernet, monitors network flows using a Ryu SDN controller, logs relevant flow statistics, and employs a trained Transformer model to classify traffic in near real-time as either normal or belonging to various attack categories.

## Features

*   **AMI Network Simulation:** Employs Containernet (Mininet with Docker support) to simulate houses (Docker containers) and AMI gateways (OpenFlow switches).
*   **SDN Control:** Uses a Ryu-based controller for basic L2 switching and, crucially, for periodic collection of OpenFlow flow statistics.
*   **Flow Logging:** The Ryu controller logs essential flow details (IPs, ports, protocol, byte/packet counts, duration) to a CSV file (`flow_stats.log`).
*   **Transformer Anomaly Detection:**
    *   Trains a Transformer neural network model (`ads/ads.py`) on a network traffic dataset (e.g., TON_IoT) to learn patterns of normal and malicious traffic.
    *   Includes preprocessing steps (scaling, encoding) consistent between training and detection.
    *   Supports multi-class classification to identify different types of anomalies (e.g., DoS, DDoS, Scanning, etc., depending on the training data labels).
*   **Real-time Monitoring:** The `anomaly_detector.py` script monitors `flow_stats.log` for new entries, preprocesses them, and uses the trained model to predict anomalies, printing detections to its log file.

## Visuals

*   **Project Structure:**
    ```
    ADS-AMI/
    ├── ads/
    │   ├── dataset_ami/
    │   │   └── test.csv         # Training/Evaluation Dataset
    │   ├── results/             # Output from training
    │   │   ├── confusion_matrix.png
    │   │   ├── encoder.joblib
    │   │   ├── saved_transformer_model.h5
    │   │   ├── scaler.joblib
    │   │   ├── target_encoder.joblib
    │   │   └── transformer_training_history.png
    │   └── ads.py               # Model training script
    ├── containernet/            # Containernet library files
    ├── logs/                    # Runtime logs from components
    │   ├── anomaly_detector.log
    │   ├── containernet_topology.log
    │   └── ryu_controller.log
    ├── sdn_topology/
    │   ├── __pycache__/
    │   ├── ami_controller.py    # Ryu controller logic
    │   └── ami_topology.py      # Containernet topology definition
    ├── .gitignore
    ├── anomaly_detector.py      # Real-time detection script
    ├── flow_stats.log           # Log file from Ryu controller
    ├── README.md                
    ├── requirements.txt         # Python dependencies
    └── run_ami_sdn.py           # Main script to start simulation
    ```
*   **Training Results:** After running `ads.py`, check the `ads/results/` directory for:
    *   `confusion_matrix.png`: Visualizes the model's classification performance.
    *   `transformer_training_history.png`: Shows training/validation loss and accuracy over epochs.

## Installation

Follow these steps carefully to set up the project environment.

**1. Prerequisites:**

*   **Operating System:** Linux (Ubuntu recommended, as Mininet/Ryu work best here).
*   **Python:** Python 3.9
*   **Pyenv (Optional but recommended):** For managing Python versions. [pyenv installation](https://github.com/pyenv/pyenv#installation)
*   **Mininet & Containernet:** Install Mininet and its Docker-based extension Containernet.
    ```bash
    # Install Mininet core and dependencies
    git clone https://github.com/mininet/mininet
    mininet/util/install.sh -a

    # Install Containernet (follow official instructions)
    # Typically involves installing Docker and then Containernet
    # See: https://containernet.github.io/#installation
    sudo apt-get update
    sudo apt-get install -y docker.io
    git clone https://github.com/containernet/containernet.git
    cd containernet
    sudo make install
    cd ..
    ```
*   **Open vSwitch:** Usually installed with Mininet. Verify installation and ensure the service is running.
    ```bash
    sudo ovs-vsctl --version
    sudo systemctl status openvswitch-switch  # Check status
    sudo systemctl start openvswitch-switch   # Start if not running
    ```
*   **Ryu SDN Framework:** Install the Ryu controller framework.
    ```bash
    # Using pip within your chosen Python environment is recommended
    pip install ryu
    ```
*   **Build Essentials (for hping3 etc.):**
    ```bash
    sudo apt-get install -y build-essential
    ```

**2. Environment Setup:**

*   (If using pyenv) Activate your desired Python environment:
    ```bash
    # Example for my setup
    source ~/.pyenv/versions/myenv-3.9.0/bin/activate
    # Or create and activate a new one
    # pyenv virtualenv 3.9.0 ami-sdn-env
    # pyenv activate ami-sdn-env
    ```

**3. Clone Repository:**

```bash
git clone <repository-url>
cd ADS-AMI
```

**4. Install Python Dependencies:**

```bash  
pip install -r requirements.txt
```
    
**5. Prepare Dataset:**

*    Place your training dataset (subset of TON_IoT) named test.csv inside the ```ads/dataset_ami/``` directory. Ensure it has the expected columns as defined in ```ads/ads.py```.

**6. Train the Anomaly Detection Model:**

*    Run the training script. This will process the dataset, train the Transformer model, and save the model (```.h5```), scaler (```.joblib```), encoders (```.joblib```), and plots (```.png```) into the ```ads/results/``` directory.

```bash
python ads/ads.py
```

*    Important: Ensure the ```ads/results/``` directory and its contents (saved_transformer_model.h5, scaler.joblib, etc.) are present before running the simulation.

**7. Verify/Update Hardcoded Paths:**

*    CRITICAL: The ```run_ami_sdn.py``` script contains hardcoded paths for the Ryu manager and Python executables based on your specific pyenv setup.

*    Open ```run_ami_sdn.py``` and verify/update the following variables to match your system's paths:

```bash
RYU_MANAGER_CMD (== /home/atharv/.pyenv/versions/3.9.0/envs/myenv-3.9.0/bin/ryu-manager)
```
    
```bash
PYTHON_CMD (== /home/atharv/.pyenv/versions/3.9.0/envs/myenv-3.9.0/bin/python)
```

## Usage

**0. Cleanup:**

* Clean unnecessary docker containers/mininet process:

```bash
# clear mininet environment
sudo mn -c
```

```bash
# if getting docker container already running
docker ps -a
docker system prune -a
```

**1. Start the Simulation:**

*    Run the main orchestration script using sudo (required for Containernet/Mininet):

```bash      
sudo python3 run_ami_sdn.py
```

* This script will:

    * Start the Ryu Controller (```ami_controller.py```).

    * Start the Containernet topology (```ami_topology.py```).

    * Wait for ```flow_stats.log``` to be created.

    * Start the Anomaly Detector (```anomaly_detector.py```).

**2. Monitor the System:**

*    Anomaly Detections: Check the output log of the anomaly detector:
        * Open ```logs/anomaly_detector.log```
            (This will show Normal traffic detected or ```--- ANOMALY DETECTED ---``` messages with the predicted type and flow details.)
            
            OR
            
            ```bash
            tail -f logs/anomaly_detector.log
            ```

* Raw Flow Data: View the raw flow statistics being logged by the controller:
    * Open ```flow_stats.log```

        OR

        ```bash
        tail -f flow_stats.log
        ```

* Component Logs: Check individual logs in the ```logs/``` directory for detailed information or errors from Ryu (```ryu_controller.log```) and Containernet (```containernet_topology.log```).

**3. Generating Traffic and Simulating Attacks:**

* You need to execute commands inside the Docker containers representing the houses (e.g., ```mn.h1_1, mn.h1_2, mn.h2_1, mn.h2_2```). Use

        sudo docker exec -it <container_name> <command>.

* Install Tools in Containers (One-time per container):

```bash
      
# Example for h1_1
sudo docker exec -it mn.h1_1 bash
# --- Inside the container ---
apt-get update
apt-get install -y iputils-ping net-tools nmap hping3 netcat dnsutils curl wget python3
exit
# --- End of container ---
# Repeat for other containers as needed (e.g., mn.h1_2)
```

* Generate NORMAL traffic:
```bash
sudo docker exec -it mn.h1_1 ping -c 5 10.0.0.3 # Ping from h1_1 to h2_1
sudo docker exec -it mn.h2_2 ping -c 5 10.0.0.2 # Ping from h2_2 to h1_2
```

* Port Scanning (TCP SYN Scan):
```bash
sudo docker exec -it mn.h1_1 nmap -Pn -sS -p 1-1024 10.0.0.3
```

* XSS:
```bash
sudo docker exec -it mn.h1_1 hping3 --flood --syn -p 80 10.0.0.3 #STOP THIS COMMAND AFTER 5 SECONDS
```
or
```bash
sudo ping -f 10.0.0.4
```

* PASSWORD attack (in bash):
```bash
sudo docker exec -it mn.h1_1 bash
sudo hping3 --syn -p 80 --flood --rand-source 10.0.0.3
```

* DDOS (in bash):
```bash
sudo docker exec -it mn.h1_2 bash
sudo hping3 --udp -p 53 --flood --rand-source -d 1000 10.0.0.3
```

**4. Stop the Simulation:**

* Press Ctrl+C in the terminal where run_ami_sdn.py is running. 

## Troubleshooting

* Open vSwitch: Ensure the service is running: ```sudo systemctl status openvswitch-switch.``` Start it if needed: ```sudo systemctl start openvswitch-switch.```

* Permissions: Most commands involving Mininet/Containernet or modifying network state require ```sudo```.

* Hardcoded Paths: Double-check the ```RYU_MANAGER_CMD``` and ```PYTHON_CMD``` in ```run_ami_sdn.py```.

* Model/Scaler Not Found: Ensure you have run ```python ads/ads.py``` successfully and the files exist in ```ads/results/```.

## Roadmap

*    Integrate more sophisticated attack simulation tools (e.g., Scapy).

*    Develop a web-based UI for visualization and monitoring.

*    Optimize the anomaly detection model and preprocessing pipeline for performance.

*    Experiment with different ML models (LSTM, GRU, etc.).

*    Improve flow tracking in the controller for more accurate duration and bidirectional statistics.