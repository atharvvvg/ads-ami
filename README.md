      
# AMI Smart Grid SDN with Transformer-based Anomaly Detection

## Overview

This project presents an integrated system for enhancing the cybersecurity posture of Advanced Metering Infrastructure (AMI) within smart grids. Traditional Intrusion Detection Systems (IDS) often struggle with the dynamic nature and specific threat landscape of AMI networks. This solution addresses these challenges by leveraging the synergy between Software-Defined Networking (SDN) for granular network visibility and control, and advanced Artificial Intelligence (AI) for intelligent threat detection.

Specifically, it utilizes:
*   **Containernet:** To create a realistic, yet controlled, simulation of an AMI network topology with Docker containers representing consumer houses and Open vSwitch instances acting as AMI gateways.
*   **Ryu SDN Controller:** To manage the network, enforce basic connectivity, and systematically collect detailed flow statistics (IPs, ports, protocols, byte/packet counts, duration) from the AMI gateways via OpenFlow.
*   **Transformer-based Deep Learning Model:** A sophisticated AI model trained on the `TON_IoT` dataset (a relevant IoT/IIoT dataset) to learn complex temporal patterns and differentiate between normal AMI traffic and various malicious activities like Denial of Service (DoS), Distributed DoS (DDoS), and network scanning.
*   **Near Real-Time Detection:** A dedicated monitoring script analyzes the collected flow statistics, preprocesses them, and uses the trained Transformer model to classify network flows, logging detections promptly.

The project demonstrates a complete workflow from simulation setup and data acquisition to AI-powered analysis and threat identification, showcasing a robust and adaptable approach to AMI security.


## Features

*   **AMI Network Simulation:** Employs Containernet (Mininet with Docker support) to simulate houses (Docker containers) and AMI gateways (OpenFlow switches).
*   **SDN Control:** Uses a Ryu-based controller for basic L2 switching and, crucially, for periodic collection of OpenFlow flow statistics.
*   **Flow Logging:** The Ryu controller logs essential flow details (IPs, ports, protocol, byte/packet counts, duration) to a CSV file (`flow_stats.log`).
*   **Transformer Anomaly Detection:**
    *   Trains a Transformer neural network model (`ads/ads.py`) on a network traffic dataset (e.g., TON_IoT) to learn patterns of normal and malicious traffic.
    *   Includes preprocessing steps (scaling, encoding) consistent between training and detection.
    *   Supports multi-class classification to identify different types of anomalies (e.g., DoS, DDoS, Scanning, etc., depending on the training data labels).
*   **Real-time Monitoring:** The `anomaly_detector.py` script monitors `flow_stats.log` for new entries, preprocesses them, and uses the trained model to predict anomalies, printing detections to its log file.

## System Architecture
<br />
<p align="center">
  <img src="https://github.com/user-attachments/assets/095346df-9b62-4b44-8661-7b9a3aee7bd0" alt="image" />
  <br />
  <em>Fig. 1 System Architecture</em>
</p>
<br />
The system follows a layered architecture:
1.  **Simulation Layer (Containernet):** Emulates the physical AMI network (`ami_topology.py`).
2.  **SDN Control Layer (Ryu):** Manages switches, collects stats (`ami_controller.py`).
3.  **Data Logging Layer:** Persists flow statistics (`flow_stats.log`).
4.  **Analysis & Detection Layer (Transformer ADS):**
    *   *Offline Training:* Trains the Transformer model (`ads/ads.py`).
    *   *Near Real-Time Detection:* Monitors logs and classifies flows (`anomaly_detector.py`).
5.  **Orchestration Layer:** Manages the startup and coordination of all components (`run_ami_sdn.py`).
<br />
<p align="center">
  <img src="https://github.com/user-attachments/assets/87d2f3fd-17ad-49a8-a7e2-d23bc4b4f3ee" alt="image" 
  <br />
  <em>Fig. 2 Simulation Network Topology</em>
</p>
<br />

## Technology Stack

*   **Simulation:** Containernet, Mininet, Docker, Open vSwitch (OVS)
*   **SDN Controller:** Ryu SDN Framework
*   **AI/ML:** Python 3.9, TensorFlow/Keras, Scikit-learn, Pandas, NumPy
*   **Operating System:** Linux (Ubuntu Recommended)


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
*   **Open vSwitch (OVS):** Usually installed with Mininet. Verify installation and ensure the service is running.
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
<br />
<p align="center">
  <img src="https://github.com/user-attachments/assets/a1a46f8f-3ee6-4432-a743-3834f8d7d6ce" alt="image" />
  <br />
  <em>Fig. 3 System Initialization via run_ami_sdn.py</em>
</p>
<br />

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
<br />
<p align="center">
  <img src="https://github.com/user-attachments/assets/1ee91e9f-e038-4a98-bd7e-ff306258b686" alt="image" />
  <br />
  <em>Fig. 4 Sample Content of flow_stats.log (Raw Flow Data)</em>
</p>
<br />

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
<br />
<p align="center">
  <img src="https://github.com/user-attachments/assets/9c7b3143-a8da-4055-b9b5-1352788a12c2" alt="image" />
  <br />
  <em>Fig. 5 Detecting Normal Traffic (Ping)</em>
</p>
<br />
<p align="center">
  <img src="https://github.com/user-attachments/assets/863e1c0b-6293-4764-be08-80f34a5648c2" alt="image" />
  <br />
  <em>Fig. 6 Detecting Password Attack Traffic</em>
</p>
<br />
<p align="center">
  <img src="https://github.com/user-attachments/assets/aaf5cf38-e72a-46fd-8de7-4d9d8be7e227" alt="image" />
  <br />
  <em>Fig. 7 Detecting DDoS UDP Flood Attack Traffic</em>
</p>
<br />

**4. Stop the Simulation:**

* Press Ctrl+C in the terminal where run_ami_sdn.py is running. 

## Results Highlights
<br />
<p align="center">
  <img src="https://github.com/user-attachments/assets/abc55e62-8a9c-4601-a817-b57caf401af9" alt="image" />
  <br />
  <em>Fig. 8 Model evaluation graph</em>
</p>
<br />

The Transformer-based model demonstrated strong performance in classifying network traffic within the simulated AMI environment:

*   **Overall F1 Score (Weighted):** ~0.9144
*   **Overall Precision (Weighted):** ~0.9257
*   **Overall Recall (Weighted):** ~0.9167
*   **AUC-ROC (One-vs-Rest):** ~0.9893 (indicating excellent class discrimination)
*   **Average Inference Time:** ~20.52 ms per flow sample (demonstrating near real-time capability)
*   **False Alarm Rate (Normal Misclassified):** ~0.2727 (Area for potential improvement/tuning)

Crucially, the Transformer model outperformed a baseline LSTM-Autoencoder approach on similar tasks, due to the effectiveness of its attention mechanism in capturing complex flow patterns. Detailed results, including the confusion matrix and training history plots, can be found in the `ads/results/` directory after running the training script.
<br />
<p align="center">
  <img src="https://github.com/user-attachments/assets/7a850ee0-6de6-44a9-9bbb-8ea25a3854b4" alt="image" />
  <br />
  <em>Fig. 9 Results of LSTM-Autoencoder Model</em>
</p>
<br />

## Troubleshooting

* Open vSwitch: Ensure the service is running: ```sudo systemctl status openvswitch-switch.``` Start it if needed: ```sudo systemctl start openvswitch-switch.```

* Permissions: Most commands involving Mininet/Containernet or modifying network state require ```sudo```.

* Hardcoded Paths: Double-check the ```RYU_MANAGER_CMD``` and ```PYTHON_CMD``` in ```run_ami_sdn.py```.

* Model/Scaler Not Found: Ensure you have run ```python ads/ads.py``` successfully and the files exist in ```ads/results/```.

## Future Enhancements

*   **Real-World Validation:** Test on physical AMI testbeds or using extensive real-world trace data.
*   **IPS Integration:** Implement closed-loop mitigation by allowing the detector to trigger response actions (e.g., installing blocking rules) via the SDN controller.
*   **Dataset Enrichment:** Augment training data with more diverse and AMI-specific datasets, potentially using synthetic data generation.
*   **Energy Theft Detection:** Extend anomaly detection beyond network patterns to potentially identify anomalies related to energy consumption data itself.
*   **Scalability Improvements:** Investigate distributed architectures for the controller and detection components.
*   **UI Development:** Create a web-based dashboard for easier monitoring and visualization.

## Acknowledgements

This project was developed as part of the BCSE498J Capstone Project at Vellore Institute of Technology (VIT).
