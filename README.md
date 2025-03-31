# AMI Smart Grid SDN with Anomaly Detection

This project implements a Software-Defined Networking (SDN) solution for Advanced Metering Infrastructure (AMI) smart grids with integrated anomaly detection using a transformer-based machine learning model.

## Project Overview

The system consists of three main components:

1. **Network Topology**: A Containernet-based simulation of an AMI smart grid with smart meters and houses connected to an SDN controller.
2. **SDN Controller**: A Ryu-based controller that monitors network traffic and collects flow statistics.
3. **Anomaly Detection System**: A transformer-based machine learning model that analyzes network flows to detect potential threats.

## System Architecture

The architecture follows the diagram provided, with:

- An SDN controller at the core
- Smart meters for each user group
- Multiple houses connected to each smart meter

The system uses the following technologies:

- **Containernet**: For network simulation (extends Mininet with Docker container support)
- **Ryu**: For SDN controller implementation
- **TensorFlow**: For the transformer-based anomaly detection model
- **Pandas/NumPy**: For data processing and analysis

## Installation

### Prerequisites

- Ubuntu operating system
- Python 3.8 or higher
- Docker

### Setup

1. Clone this repository:

   ```
   git clone <repository-url>
   cd new_capstone
   ```

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Install Containernet:

   ```
   git clone https://github.com/containernet/containernet.git
   cd containernet
   sudo ./install.sh
   ```

4. Install Ryu:
   ```
   pip install ryu
   ```

## Usage

### Running the Complete System

To run the entire AMI Smart Grid SDN system with anomaly detection:

```
python run_ami_sdn.py
```

This will:

1. Start the Ryu SDN controller with the AMI controller application
2. Create the network topology with smart meters and houses
3. Begin monitoring network flows for anomalies

### Running Components Separately

You can also run individual components:

1. Start only the SDN controller:

   ```
   python run_ami_sdn.py --controller-only
   ```

2. Start only the network topology (requires controller to be running):

   ```
   python run_ami_sdn.py --topology-only
   ```

3. Only monitor logs for anomaly detection:
   ```
   python run_ami_sdn.py --monitor-only
   ```

### Testing the Anomaly Detection System

To test the anomaly detection system independently:

```
python anomaly_detector.py
```

This will run the anomaly detector on the test dataset and display the results.

## Project Structure

```
├── ads/                        # Anomaly Detection System
│   ├── ads.py                  # Main ADS implementation
│   ├── dataset_ami/            # Dataset directory
│   │   └── test.csv            # Test dataset
│   └── saved_transformer_model.h5  # Trained model
├── sdn_topology/               # SDN Topology
│   ├── ami_topology.py         # Network topology implementation
│   └── ami_controller.py       # SDN controller implementation
├── anomaly_detector.py         # Interface between ADS and SDN
├── run_ami_sdn.py              # Main script to run the system
└── requirements.txt            # Project dependencies
```

## Anomaly Detection

The system uses a transformer-based model to detect the following types of anomalies:

- Normal traffic
- Backdoor attacks
- DDoS attacks
- Injection attacks
- Man-in-the-middle attacks
- Password attacks
- Ransomware
- Port scanning
- Cross-site scripting (XSS)

When an anomaly is detected, the SDN controller takes appropriate mitigation actions, such as blocking the malicious traffic.

## Extending the System

### Adding New Devices

To add new devices to the topology, modify the `ami_topology.py` file and add new hosts and links as needed.

### Customizing Anomaly Detection

The anomaly detection model can be retrained with custom data by modifying the `ads.py` file and running the training process.

## Troubleshooting

### Common Issues

1. **Controller Connection Failure**:

   - Ensure the Ryu controller is running before starting the topology
   - Check if port 6653 is available and not blocked by a firewall

2. **Anomaly Detection Issues**:

   - Verify that the model file `saved_transformer_model.h5` exists
   - Check the format of flow data matches the expected input format

3. **Containernet Issues**:
   - Ensure Docker is running
   - Run with sudo if permission issues occur

## License

This project is licensed under the MIT License - see the LICENSE file for details.
