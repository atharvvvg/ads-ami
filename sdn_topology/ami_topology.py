#!/usr/bin/python
"""
Defines the AMI Smart Grid topology using Containernet.

Topology:
    - Remote Ryu Controller
    - 2 Switches (AMI Gateways)
    - 4 Docker Hosts (Houses), 2 connected to each switch.
"""

import os
from mininet.net import Containernet
from mininet.node import Controller, RemoteController
from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import info, setLogLevel

# Set log level
setLogLevel('info')

# --- Configuration ---
CONTROLLER_IP = '127.0.0.1' # IP address of the Ryu controller
CONTROLLER_PORT = 6653      # Default OpenFlow port for Ryu
DOCKER_IMAGE = 'ubuntu:trusty' # Docker image for house containers

def create_topology():
    """Creates and configures the AMI smart grid topology."""

    info('*** Creating network\n')
    # Use Containernet to allow Docker containers as hosts
    net = Containernet(controller=None) # Controller will be added manually

    info('*** Adding controller\n')
    # Add a remote controller running Ryu
    c0 = net.addController(
        name='c0',
        controller=RemoteController,
        ip=CONTROLLER_IP,
        port=CONTROLLER_PORT
    )

    info('*** Adding switches (AMI Gateways)\n')
    # Add two switches to represent AMI gateways
    s1 = net.addSwitch('s1', dpid='0000000000000001') # Use specific DPIDs
    s2 = net.addSwitch('s2', dpid='0000000000000002')

    info('*** Adding hosts (Houses) using Docker containers\n')
    # Define Docker resource limits if needed (optional)
    dargs = {'dimage': DOCKER_IMAGE,
             'cpu_shares': 20, # Example: limit CPU shares
             'mem_limit': '128m'} # Example: limit memory

    # Add hosts for AMI Gateway 1
    h1_1 = net.addDocker('h1_1', ip='10.0.0.1/24', dpid='s1', **dargs)
    h1_2 = net.addDocker('h1_2', ip='10.0.0.2/24', dpid='s1', **dargs)

    # Add hosts for AMI Gateway 2
    h2_1 = net.addDocker('h2_1', ip='10.0.0.3/24', dpid='s2', **dargs)
    h2_2 = net.addDocker('h2_2', ip='10.0.0.4/24', dpid='s2', **dargs)

    info('*** Creating links\n')
    # Link AMI gateways (switches) to the controller (implicitly handled by setting controller)

    # Link houses (hosts) to their respective AMI gateways (switches)
    net.addLink(h1_1, s1, cls=TCLink, bw=10) # Example bandwidth limit
    net.addLink(h1_2, s1, cls=TCLink, bw=10)
    net.addLink(h2_1, s2, cls=TCLink, bw=10)
    net.addLink(h2_2, s2, cls=TCLink, bw=10)

    # Link the AMI gateways (switches) together (optional, depends on desired routing)
    # If AMI gateways need to communicate directly or route between houses on different gateways
    # net.addLink(s1, s2, cls=TCLink, bw=100)

    info('*** Starting network\n')
    net.build()
    # Start the switches and link them to the controller
    s1.start([c0])
    s2.start([c0])

    info('*** Running CLI\n')
    # Open the Mininet command-line interface for interaction
    CLI(net)

    info('*** Stopping network\n')
    net.stop()

if __name__ == '__main__':
    # Ensure script runs with sudo
    if os.getuid() != 0:
        print("This script requires sudo privileges!")
    else:
        create_topology()
