#!/usr/bin/env python3
"""
Main script to run the AMI Smart Grid SDN simulation with Anomaly Detection.

Starts the Ryu controller, Containernet topology, and the anomaly detector script.
Uses hardcoded paths for Python and Ryu executables based on user environment.
"""

import subprocess
import time
import os
import signal
import sys

# --- Configuration ---
RYU_CONTROLLER_SCRIPT = os.path.join('sdn_topology', 'ami_controller.py')
CONTAINERNET_TOPOLOGY_SCRIPT = os.path.join('sdn_topology', 'ami_topology.py')
ANOMALY_DETECTOR_SCRIPT = 'anomaly_detector.py'

# --- Hardcoded Paths (as requested by user) ---
# Ensure these paths are correct for your environment
RYU_MANAGER_CMD = '/home/atharv/.pyenv/versions/3.9.0/envs/myenv-3.9.0/bin/ryu-manager'
PYTHON_CMD = '/home/atharv/.pyenv/versions/3.9.0/envs/myenv-3.9.0/bin/python'
# --- --- --- --- --- --- --- --- --- --- --- ---

LOG_DIR = 'logs' # Directory to store logs from processes
FLOW_LOG_FILE = 'flow_stats.log' # Make sure this matches controller/detector

# --- Global Process Variables ---
ryu_process = None
containernet_process = None
detector_process = None

def start_ryu_controller():
    """Starts the Ryu controller in a separate process."""
    global ryu_process
    print(f"Starting Ryu controller: {RYU_CONTROLLER_SCRIPT}")
    print(f"Using Ryu executable: {RYU_MANAGER_CMD}") # Log the path being used
    if not os.path.exists(RYU_CONTROLLER_SCRIPT):
        print(f"ERROR: Ryu controller script not found at {RYU_CONTROLLER_SCRIPT}")
        return False
    if not os.path.exists(RYU_MANAGER_CMD):
         print(f"ERROR: Ryu executable not found at hardcoded path: {RYU_MANAGER_CMD}")
         return False

    # Ensure log directory exists
    os.makedirs(LOG_DIR, exist_ok=True)
    ryu_log_file = os.path.join(LOG_DIR, 'ryu_controller.log')

    try:
        # Start Ryu manager using the hardcoded path
        with open(ryu_log_file, 'w') as log_f:
            ryu_process = subprocess.Popen(
                [RYU_MANAGER_CMD, '--verbose', RYU_CONTROLLER_SCRIPT],
                stdout=log_f,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid # Create a new process group
            )
        print(f"Ryu controller started (PID: {ryu_process.pid}). Logging to {ryu_log_file}")
        # Give Ryu some time to start up
        time.sleep(7) # Increased sleep slightly
        return True
    except FileNotFoundError:
        # This might occur if the path is wrong despite the exists check (rare)
        print(f"ERROR: Command '{RYU_MANAGER_CMD}' failed with FileNotFoundError.")
        return False
    except Exception as e:
        print(f"ERROR: Failed to start Ryu controller: {e}")
        return False

def start_containernet_topology():
    """Starts the Containernet topology simulation."""
    global containernet_process
    print(f"Starting Containernet topology: {CONTAINERNET_TOPOLOGY_SCRIPT}")
    print(f"Using Python executable: {PYTHON_CMD}") # Log the path being used
    if not os.path.exists(CONTAINERNET_TOPOLOGY_SCRIPT):
        print(f"ERROR: Containernet topology script not found at {CONTAINERNET_TOPOLOGY_SCRIPT}")
        return False
    if not os.path.exists(PYTHON_CMD):
         print(f"ERROR: Python executable not found at hardcoded path: {PYTHON_CMD}")
         return False

    # Ensure log directory exists
    os.makedirs(LOG_DIR, exist_ok=True)
    topo_log_file = os.path.join(LOG_DIR, 'containernet_topology.log')

    try:
        # Containernet often requires sudo
        # Use the hardcoded Python path
        cmd = ['sudo', PYTHON_CMD, CONTAINERNET_TOPOLOGY_SCRIPT]
        with open(topo_log_file, 'w') as log_f:
            containernet_process = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid # Create a new process group
            )
        print(f"Containernet topology started (PID: {containernet_process.pid}). Logging to {topo_log_file}")
        # Give topology time to set up links with controller
        time.sleep(12) # Increased sleep slightly
        return True
    except FileNotFoundError:
         # This might occur if sudo isn't found or PYTHON_CMD path is wrong despite check
         print(f"ERROR: Command 'sudo' or '{PYTHON_CMD}' failed with FileNotFoundError.")
         return False
    except Exception as e:
        print(f"ERROR: Failed to start Containernet topology: {e}")
        # Check if it's a permission error
        if isinstance(e, PermissionError) or 'sudo' in str(e):
             print("Hint: This script might need to be run with sudo itself, or ensure your user has sudo rights without password.")
        return False


def start_anomaly_detector():
    """Starts the anomaly detection script."""
    global detector_process
    print(f"Starting Anomaly Detector: {ANOMALY_DETECTOR_SCRIPT}")
    print(f"Using Python executable: {PYTHON_CMD}") # Log the path being used
    if not os.path.exists(ANOMALY_DETECTOR_SCRIPT):
        print(f"ERROR: Anomaly detector script not found at {ANOMALY_DETECTOR_SCRIPT}")
        return False
    if not os.path.exists(os.path.join('ads', 'saved_transformer_model.h5')):
         print(f"ERROR: Model file 'ads/saved_transformer_model.h5' not found.")
         return False
    if not os.path.exists(PYTHON_CMD):
         print(f"ERROR: Python executable not found at hardcoded path: {PYTHON_CMD}")
         return False

    # Ensure log directory exists
    os.makedirs(LOG_DIR, exist_ok=True)
    detector_log_file = os.path.join(LOG_DIR, 'anomaly_detector.log')

    # Wait for the flow log file to be created by the controller
    print(f"Waiting for flow log file '{FLOW_LOG_FILE}' to be created...")
    wait_time = 0
    max_wait = 30 # seconds
    while not os.path.exists(FLOW_LOG_FILE) and wait_time < max_wait:
        time.sleep(1)
        wait_time += 1
    if not os.path.exists(FLOW_LOG_FILE):
        # Check if Ryu process died, which might explain missing log file
        if ryu_process and ryu_process.poll() is not None:
             print(f"ERROR: Ryu controller process (PID: {ryu_process.pid}) seems to have terminated. Check logs/{ryu_log_file}.")
        print(f"ERROR: Flow log file '{FLOW_LOG_FILE}' was not created by the controller after {max_wait} seconds.")
        return False
    print(f"Flow log file found.")


    try:
        # Use the hardcoded Python path
        cmd = [PYTHON_CMD, ANOMALY_DETECTOR_SCRIPT]
        with open(detector_log_file, 'w') as log_f:
            detector_process = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid # Create a new process group
            )
        print(f"Anomaly detector started (PID: {detector_process.pid}). Logging to {detector_log_file}")
        return True
    except FileNotFoundError:
         print(f"ERROR: Command '{PYTHON_CMD}' failed with FileNotFoundError.")
         return False
    except Exception as e:
        print(f"ERROR: Failed to start anomaly detector: {e}")
        return False

def stop_simulation():
    """Stops all running processes gracefully."""
    print("\nStopping simulation...")

    # Stop processes in reverse order of start
    if detector_process and detector_process.poll() is None: # Check if process exists and is running
        print(f"Stopping Anomaly Detector (PID: {detector_process.pid})...")
        try:
            pgid = os.getpgid(detector_process.pid)
            os.killpg(pgid, signal.SIGTERM)
            detector_process.wait(timeout=5)
        except ProcessLookupError:
            print("Anomaly Detector process already finished.")
        except subprocess.TimeoutExpired:
            print("Anomaly detector did not terminate gracefully, killing...")
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                 print("Anomaly Detector process already finished before kill.") # Handle race condition
            except Exception as kill_e:
                 print(f"Error force killing detector: {kill_e}")
        except Exception as e:
            print(f"Error stopping detector: {e}")
    elif detector_process:
         print("Anomaly Detector process already terminated.")


    if containernet_process and containernet_process.poll() is None:
        print(f"Stopping Containernet Topology (PID: {containernet_process.pid})...")
        # Containernet cleanup often needs sudo kill
        try:
             pgid = os.getpgid(containernet_process.pid)
             print(f"Sending SIGTERM to Containernet process group {pgid}")
             # Use Popen to avoid blocking indefinitely if sudo asks for password
             kill_proc = subprocess.Popen(['sudo', 'kill', '-SIGTERM', f'-{pgid}'], stderr=subprocess.PIPE)
             _, stderr = kill_proc.communicate(timeout=10)
             if kill_proc.returncode != 0:
                 print(f"Error sending SIGTERM to Containernet group: {stderr.decode()}")

             containernet_process.wait(timeout=10) # Wait for main process
        except ProcessLookupError:
             print("Containernet process already finished.")
        except subprocess.TimeoutExpired:
             print("Containernet did not terminate gracefully after SIGTERM, killing group...")
             try:
                 # Use Popen for SIGKILL as well
                 kill_proc = subprocess.Popen(['sudo', 'kill', '-SIGKILL', f'-{pgid}'], stderr=subprocess.PIPE)
                 _, stderr = kill_proc.communicate(timeout=5)
                 if kill_proc.returncode != 0:
                     print(f"Error sending SIGKILL to Containernet group: {stderr.decode()}")
             except ProcessLookupError:
                  print("Containernet process already finished before kill.")
             except Exception as kill_e:
                  print(f"Error force killing containernet group: {kill_e}")
        except Exception as e:
            print(f"Error stopping Containernet (may need manual cleanup 'sudo mn -c'): {e}")
    elif containernet_process:
        print("Containernet process already terminated.")


    if ryu_process and ryu_process.poll() is None:
        print(f"Stopping Ryu Controller (PID: {ryu_process.pid})...")
        try:
            pgid = os.getpgid(ryu_process.pid)
            os.killpg(pgid, signal.SIGTERM)
            ryu_process.wait(timeout=5)
        except ProcessLookupError:
            print("Ryu process already finished.")
        except subprocess.TimeoutExpired:
            print("Ryu controller did not terminate gracefully, killing...")
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                 print("Ryu process already finished before kill.")
            except Exception as kill_e:
                 print(f"Error force killing Ryu: {kill_e}")
        except Exception as e:
            print(f"Error stopping Ryu: {e}")
    elif ryu_process:
        print("Ryu process already terminated.")


    # Final Mininet cleanup (just in case)
    print("Running 'sudo mn -c' for final cleanup...")
    try:
        # Use Popen to avoid blocking
        mn_clean_proc = subprocess.Popen(['sudo', 'mn', '-c'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = mn_clean_proc.communicate(timeout=15)
        if mn_clean_proc.returncode != 0:
             print(f"Warning: 'sudo mn -c' failed. Stderr: {stderr.decode()}")
        else:
             print("'sudo mn -c' completed.")
             # print(f"Stdout: {stdout.decode()}") # Optional: show output
    except subprocess.TimeoutExpired:
         print("Warning: 'sudo mn -c' timed out.")
    except Exception as e:
        print(f"Warning: 'sudo mn -c' failed: {e}")

    print("Simulation stopped.")

def handle_interrupt(sig, frame):
    """Handles Ctrl+C interrupt."""
    print("\nCtrl+C detected.")
    stop_simulation()
    sys.exit(0)

if __name__ == "__main__":
    # Register interrupt handler
    signal.signal(signal.SIGINT, handle_interrupt)

    # --- Start Simulation ---
    if not start_ryu_controller():
        print("Failed to start Ryu controller. Aborting.")
        stop_simulation() # Attempt cleanup just in case
        sys.exit(1)

    if not start_containernet_topology():
        print("Failed to start Containernet topology. Aborting.")
        stop_simulation()
        sys.exit(1)

    if not start_anomaly_detector():
        print("Failed to start Anomaly Detector. Aborting.")
        stop_simulation()
        sys.exit(1)

    # --- Keep Running ---
    print("\nSimulation is running.")
    print("Press Ctrl+C to stop.")
    try:
        # Keep the main script alive while processes run
        # Monitor processes and exit if one fails unexpectedly
        while True:
             ryu_rc = ryu_process.poll() if ryu_process else None
             cn_rc = containernet_process.poll() if containernet_process else None
             ad_rc = detector_process.poll() if detector_process else None

             if ryu_rc is not None:
                  print(f"ERROR: Ryu controller process terminated unexpectedly (Return Code: {ryu_rc}). Check logs/{LOG_DIR}/ryu_controller.log")
                  break
             if cn_rc is not None:
                  print(f"ERROR: Containernet topology process terminated unexpectedly (Return Code: {cn_rc}). Check logs/{LOG_DIR}/containernet_topology.log")
                  break
             if ad_rc is not None:
                  print(f"ERROR: Anomaly detector process terminated unexpectedly (Return Code: {ad_rc}). Check logs/{LOG_DIR}/anomaly_detector.log")
                  break

             time.sleep(5) # Check every 5 seconds


    except KeyboardInterrupt:
         # This part is handled by the signal handler now
         print("\nKeyboardInterrupt received by main loop.")
         pass
    finally:
        # Ensure cleanup happens even if loop breaks unexpectedly
        print("\nExiting main loop or encountered error. Initiating cleanup...")
        stop_simulation()

