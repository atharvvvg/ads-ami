#!/usr/bin/env python

"""
AMI Smart Grid Topology with SDN Controller
This script creates a network topology that mimics an Advanced Metering Infrastructure (AMI)
with smart meters and houses connected to an SDN controller.
"""

from mininet.net import Containernet
from mininet.node import Controller, OVSSwitch, RemoteController
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink
import time
import os

def create_ami_topology():
    """Create an AMI smart grid topology with SDN controller"""
    
    # Initialize Containernet
    net = Containernet(controller=RemoteController)
    
    # Add remote controller (Ryu)
    info('*** Adding controller\n')
    c0 = net.addController('c0', controller=RemoteController, ip='127.0.0.1', port=6653)
    
    # Add switches for the SDN infrastructure
    info('*** Adding switches\n')
    s0 = net.addSwitch('s0')  # Main switch connected to the controller
    
    # Add switches for each smart meter
    s1 = net.addSwitch('s1')  # Switch for User 1's smart meter
    s2 = net.addSwitch('s2')  # Switch for User 2's smart meter
    s3 = net.addSwitch('s3')  # Switch for User 3's smart meter
    
    # Add smart meters as hosts
    info('*** Adding smart meters\n')
    sm1 = net.addHost('sm1', ip='10.0.0.10/24')
    sm2 = net.addHost('sm2', ip='10.0.0.20/24')
    sm3 = net.addHost('sm3', ip='10.0.0.30/24')
    
    # Add houses as hosts
    info('*** Adding houses\n')
    # User 1 houses
    h1_1 = net.addHost('h1_1', ip='10.0.0.11/24')
    h1_2 = net.addHost('h1_2', ip='10.0.0.12/24')
    h1_3 = net.addHost('h1_3', ip='10.0.0.13/24')
    
    # User 2 houses
    h2_1 = net.addHost('h2_1', ip='10.0.0.21/24')
    h2_2 = net.addHost('h2_2', ip='10.0.0.22/24')
    h2_3 = net.addHost('h2_3', ip='10.0.0.23/24')
    
    # User 3 houses
    h3_1 = net.addHost('h3_1', ip='10.0.0.31/24')
    h3_2 = net.addHost('h3_2', ip='10.0.0.32/24')
    h3_3 = net.addHost('h3_3', ip='10.0.0.33/24')
    
    # Add links between SDN controller switch and smart meter switches
    info('*** Creating links\n')
    net.addLink(s0, s1, cls=TCLink, delay='5ms', bw=10)
    net.addLink(s0, s2, cls=TCLink, delay='5ms', bw=10)
    net.addLink(s0, s3, cls=TCLink, delay='5ms', bw=10)
    
    # Connect smart meters to their respective switches
    net.addLink(s1, sm1, cls=TCLink, delay='2ms', bw=5)
    net.addLink(s2, sm2, cls=TCLink, delay='2ms', bw=5)
    net.addLink(s3, sm3, cls=TCLink, delay='2ms', bw=5)
    
    # Connect houses to their respective smart meters via switches
    # User 1 houses
    net.addLink(s1, h1_1, cls=TCLink, delay='1ms', bw=2)
    net.addLink(s1, h1_2, cls=TCLink, delay='1ms', bw=2)
    net.addLink(s1, h1_3, cls=TCLink, delay='1ms', bw=2)
    
    # User 2 houses
    net.addLink(s2, h2_1, cls=TCLink, delay='1ms', bw=2)
    net.addLink(s2, h2_2, cls=TCLink, delay='1ms', bw=2)
    net.addLink(s2, h2_3, cls=TCLink, delay='1ms', bw=2)
    
    # User 3 houses
    net.addLink(s3, h3_1, cls=TCLink, delay='1ms', bw=2)
    net.addLink(s3, h3_2, cls=TCLink, delay='1ms', bw=2)
    net.addLink(s3, h3_3, cls=TCLink, delay='1ms', bw=2)
    
    # Start the network
    info('*** Starting network\n')
    net.start()
    
    # Wait for the controller to connect
    info('*** Waiting for controller to connect...\n')
    time.sleep(5)
    
    # Configure hosts
    info('*** Configuring hosts\n')
    
    # Set default routes for all hosts
    for host in net.hosts:
        host.cmd('route add default gw 10.0.0.1')
    
    # Return the network object for CLI or further configuration
    return net

def run_network_with_traffic_generation():
    """Run the network and generate some sample traffic"""
    net = create_ami_topology()
    
    # Generate some sample traffic
    info('*** Generating sample traffic\n')
    
    # Ping between houses and smart meters
    info('*** Testing connectivity between houses and smart meters\n')
    net.pingAll()
    
    # Start iperf servers on smart meters
    info('*** Starting iperf servers on smart meters\n')
    for sm in ['sm1', 'sm2', 'sm3']:
        net.getNodeByName(sm).cmd('iperf -s &')
    
    # Run iperf clients from houses to their respective smart meters
    info('*** Running iperf clients from houses to smart meters\n')
    # User 1 houses to smart meter 1
    for h in ['h1_1', 'h1_2', 'h1_3']:
        net.getNodeByName(h).cmd(f'iperf -c 10.0.0.10 -t 10 -i 1 &')
    
    # User 2 houses to smart meter 2
    for h in ['h2_1', 'h2_2', 'h2_3']:
        net.getNodeByName(h).cmd(f'iperf -c 10.0.0.20 -t 10 -i 1 &')
    
    # User 3 houses to smart meter 3
    for h in ['h3_1', 'h3_2', 'h3_3']:
        net.getNodeByName(h).cmd(f'iperf -c 10.0.0.30 -t 10 -i 1 &')
    
    # Generate some malicious traffic (for testing anomaly detection)
    info('*** Generating some anomalous traffic patterns\n')
    # Simulate a port scan from h1_1 to sm2
    net.getNodeByName('h1_1').cmd('nmap -p 1-100 10.0.0.20 &')
    
    # Simulate a SYN flood from h2_2 to sm3
    net.getNodeByName('h2_2').cmd('hping3 -S --flood -p 80 10.0.0.30 &')
    
    # Open Mininet CLI for manual interaction
    info('*** Running CLI\n')
    CLI(net)
    
    # Clean up
    info('*** Stopping network\n')
    net.stop()

if __name__ == '__main__':
    # Tell mininet to print useful information
    setLogLevel('info')
    run_network_with_traffic_generation()