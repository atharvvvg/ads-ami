#!/usr/bin/env python

"""
AMI Smart Grid SDN with Anomaly Detection
This script runs the complete AMI smart grid SDN system with anomaly detection.
It starts the Ryu controller, creates the network topology, and monitors for anomalies.
"""

import os
import sys
import time
import subprocess
import signal
import argparse
import threading

def start_ryu_controller():
    """Start the Ryu SDN controller with the AMI controller application"""
    print("Starting Ryu SDN controller...")
    controller_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                 'sdn_topology', 'ami_controller.py')
    
    # Start Ryu controller as a subprocess
    cmd = ['ryu-manager', controller_path, '--verbose']
    controller_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for controller to initialize
    print("Waiting for controller to initialize...")
    time.sleep(5)
    
    return controller_process

def start_network_topology():
    """Start the AMI network topology using Containernet"""
    print("Starting AMI network topology...")
    topology_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                               'sdn_topology', 'ami_topology.py')
    
    # Start topology as a subprocess
    cmd = ['python', topology_path]
    topology_process = subprocess.Popen(cmd)
    
    return topology_process

def monitor_logs():
    """Monitor log files for anomaly detection results"""
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                          'sdn_topology', 'flow_logs')
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    print(f"Monitoring logs in {log_dir}")
    
    # This is a placeholder for more sophisticated log monitoring
    # In a real implementation, you might want to use a log aggregation tool
    while True:
        time.sleep(10)
        log_files = [f for f in os.listdir(log_dir) if f.endswith('.csv')]
        if log_files:
            print(f"Found {len(log_files)} log files")
            # Sort by modification time (newest first)
            log_files.sort(key=lambda x: os.path.getmtime(os.path.join(log_dir, x)), reverse=True)
            newest_log = log_files[0]
            print(f"Newest log file: {newest_log}")

def main():
    """Main function to run the AMI SDN system"""
    parser = argparse.ArgumentParser(description='Run AMI Smart Grid SDN with Anomaly Detection')
    parser.add_argument('--controller-only', action='store_true', help='Start only the controller')
    parser.add_argument('--topology-only', action='store_true', help='Start only the topology')
    parser.add_argument('--monitor-only', action='store_true', help='Only monitor logs')
    args = parser.parse_args()
    
    controller_process = None
    topology_process = None
    monitor_thread = None
    
    try:
        # Start components based on arguments
        if args.controller_only:
            controller_process = start_ryu_controller()
        elif args.topology_only:
            topology_process = start_network_topology()
        elif args.monitor_only:
            monitor_logs()
        else:
            # Start all components
            controller_process = start_ryu_controller()
            topology_process = start_network_topology()
            
            # Start log monitoring in a separate thread
            monitor_thread = threading.Thread(target=monitor_logs)
            monitor_thread.daemon = True
            monitor_thread.start()
            
            # Wait for user to press Ctrl+C
            print("\nAMI Smart Grid SDN system is running")
            print("Press Ctrl+C to stop")
            while True:
                time.sleep(1)
    
    except KeyboardInterrupt:
        print("\nStopping AMI Smart Grid SDN system...")
    finally:
        # Clean up processes
        if controller_process:
            print("Stopping Ryu controller...")
            controller_process.terminate()
            controller_process.wait()
        
        if topology_process:
            print("Stopping network topology...")
            topology_process.terminate()
            topology_process.wait()
        
        print("AMI Smart Grid SDN system stopped")

if __name__ == '__main__':
    main()