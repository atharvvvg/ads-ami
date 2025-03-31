#!/usr/bin/env python

"""
AMI Smart Grid SDN Controller
This script implements a Ryu SDN controller for the AMI smart grid topology.
It monitors network traffic, collects flow statistics, and integrates with the
anomaly detection system to identify potential threats.
"""

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ether_types, ipv4, tcp, udp
from ryu.lib import hub

import time
import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime

# Add the parent directory to sys.path to import the anomaly detection module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from anomaly_detector import AnomalyDetector

class AMIController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(AMIController, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.mac_to_port = {}
        self.flow_stats = []
        self.anomaly_detector = AnomalyDetector()
        
        # Start background threads for monitoring
        self.monitor_thread = hub.spawn(self._monitor)
        self.anomaly_detection_thread = hub.spawn(self._detect_anomalies)
        
        # Create directory for flow logs if it doesn't exist
        self.log_dir = os.path.join(os.path.dirname(__file__), 'flow_logs')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        self.logger.info("AMI Smart Grid SDN Controller started")

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """Handle switch connection and install table-miss flow entry"""
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # Install table-miss flow entry
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)
        self.logger.info(f"Switch {datapath.id} connected")

    def add_flow(self, datapath, priority, match, actions, buffer_id=None, idle_timeout=0, hard_timeout=0):
        """Add a flow entry to the switch"""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                             actions)]
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                    priority=priority, match=match,
                                    instructions=inst, idle_timeout=idle_timeout,
                                    hard_timeout=hard_timeout)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                    match=match, instructions=inst,
                                    idle_timeout=idle_timeout,
                                    hard_timeout=hard_timeout)
        datapath.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        """Handle packet-in events and learn MAC addresses"""
        # If you hit this you might want to increase
        # the "miss_send_length" of your switch
        if ev.msg.msg_len < ev.msg.total_len:
            self.logger.debug("packet truncated: only %s of %s bytes",
                              ev.msg.msg_len, ev.msg.total_len)
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            # Ignore LLDP packets
            return
        
        dst = eth.dst
        src = eth.src
        dpid = datapath.id

        # Learn MAC addresses to avoid FLOOD
        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src] = in_port

        # Extract packet information for flow statistics
        ip_proto = None
        src_ip = None
        dst_ip = None
        src_port = None
        dst_port = None
        service = '-'  # Default value

        # Extract IP information if present
        ip_pkt = pkt.get_protocol(ipv4.ipv4)
        if ip_pkt:
            src_ip = ip_pkt.src
            dst_ip = ip_pkt.dst
            ip_proto = ip_pkt.proto

            # Extract TCP/UDP information if present
            tcp_pkt = pkt.get_protocol(tcp.tcp)
            if tcp_pkt:
                src_port = tcp_pkt.src_port
                dst_port = tcp_pkt.dst_port
                # Identify common services
                if dst_port == 80 or dst_port == 8080:
                    service = 'http'
                elif dst_port == 443:
                    service = 'https'
                elif dst_port == 22:
                    service = 'ssh'
                elif dst_port == 23:
                    service = 'telnet'
                elif dst_port == 21:
                    service = 'ftp'
                else:
                    service = f'tcp/{dst_port}'
            else:
                udp_pkt = pkt.get_protocol(udp.udp)
                if udp_pkt:
                    src_port = udp_pkt.src_port
                    dst_port = udp_pkt.dst_port
                    # Identify common UDP services
                    if dst_port == 53:
                        service = 'dns'
                    elif dst_port == 67 or dst_port == 68:
                        service = 'dhcp'
                    elif dst_port == 123:
                        service = 'ntp'
                    else:
                        service = f'udp/{dst_port}'

            # Log flow information for anomaly detection
            if src_ip and dst_ip:
                flow_data = {
                    'ts': time.time(),
                    'src_ip': src_ip,
                    'src_port': src_port if src_port else 0,
                    'dst_ip': dst_ip,
                    'dst_port': dst_port if dst_port else 0,
                    'proto': 'tcp' if tcp_pkt else ('udp' if udp_pkt else 'other'),
                    'service': service,
                    'duration': 0,  # Will be updated when flow ends
                    'src_bytes': len(msg.data),
                    'dst_bytes': 0,  # Will be updated with response
                    'conn_state': 'S0',  # Initial state
                    'src_pkts': 1,
                    'dst_pkts': 0
                }
                self.flow_stats.append(flow_data)

        # Forward packet based on learned MAC addresses
        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port)]

        # Install a flow to avoid packet_in next time
        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            # Verify if we have a valid buffer_id, if yes avoid sending both
            # flow_mod & packet_out
            if msg.buffer_id != ofproto.OFP_NO_BUFFER:
                self.add_flow(datapath, 1, match, actions, msg.buffer_id, idle_timeout=10)
                return
            else:
                self.add_flow(datapath, 1, match, actions, idle_timeout=10)

        # Send packet out
        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                  in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        """Track active datapaths"""
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.logger.info(f'Datapath {datapath.id} registered')
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.info(f'Datapath {datapath.id} unregistered')
                del self.datapaths[datapath.id]

    def _monitor(self):
        """Background thread for monitoring flow statistics"""
        while True:
            # Request flow statistics from all switches
            for dp in self.datapaths.values():
                self._request_stats(dp)
            
            # Save flow statistics to CSV file every 60 seconds
            if self.flow_stats:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(self.log_dir, f'flow_stats_{timestamp}.csv')
                
                # Convert flow stats to DataFrame and save to CSV
                df = pd.DataFrame(self.flow_stats)
                df.to_csv(filename, index=False)
                self.logger.info(f'Saved {len(self.flow_stats)} flow records to {filename}')
            
            # Sleep for monitoring interval
            hub.sleep(60)

    def _request_stats(self, datapath):
        """Request flow statistics from a switch"""
        self.logger.debug(f'Sending stats request to datapath {datapath.id}')
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        """Handle flow statistics reply from switches"""
        body = ev.msg.body
        datapath = ev.msg.datapath

        self.logger.debug(f'Received flow stats from datapath {datapath.id}')
        for stat in body:
            # Update flow statistics with duration and byte counts
            for flow in self.flow_stats:
                # Try to match the flow based on IP and port information
                if 'src_ip' in flow and 'dst_ip' in flow and hasattr(stat.match, 'ipv4_src') and hasattr(stat.match, 'ipv4_dst'):
                    if (flow['src_ip'] == stat.match.ipv4_src and 
                        flow['dst_ip'] == stat.match.ipv4_dst):
                        # Update flow statistics
                        flow['duration'] = stat.duration_sec + (stat.duration_nsec / 1e9)
                        flow['dst_bytes'] = stat.byte_count - flow['src_bytes']
                        flow['conn_state'] = 'SF'  # Assume successful flow
                        flow['dst_pkts'] = stat.packet_count - flow['src_pkts']

    def _detect_anomalies(self):
        """Background thread for anomaly detection"""
        # Wait for initial flow data collection
        hub.sleep(120)
        
        while True:
            if len(self.flow_stats) > 10:  # Only run detection if we have enough flow data
                self.logger.info(f'Running anomaly detection on {len(self.flow_stats)} flows')
                
                # Convert flow stats to DataFrame for anomaly detection
                df = pd.DataFrame(self.flow_stats)
                
                # Run anomaly detection
                results = self.anomaly_detector.detect_anomalies(df)
                
                # Log anomalies
                if results['anomalies']:
                    self.logger.warning(f"Detected {len(results['anomalies'])} anomalies")
                    for anomaly in results['anomalies']:
                        self.logger.warning(f"Anomaly: {anomaly}")
                        
                        # Take action on detected anomalies (e.g., block traffic)
                        self._mitigate_anomaly(anomaly)
                else:
                    self.logger.info("No anomalies detected")
            
            # Sleep before next detection cycle
            hub.sleep(60)

    def _mitigate_anomaly(self, anomaly):
        """Take action to mitigate detected anomalies"""
        # Extract source IP from the anomaly
        src_ip = anomaly.get('src_ip')
        dst_ip = anomaly.get('dst_ip')
        anomaly_type = anomaly.get('type')
        
        if not src_ip or not dst_ip:
            self.logger.warning(f"Cannot mitigate anomaly without source/destination IP: {anomaly}")
            return
        
        self.logger.info(f"Mitigating {anomaly_type} anomaly from {src_ip} to {dst_ip}")
        
        # Block traffic from the source IP to the destination IP on all switches
        for dp in self.datapaths.values():
            ofproto = dp.ofproto
            parser = dp.ofproto_parser
            
            # Create a match for the source IP
            match = parser.OFPMatch(eth_type=ether_types.ETH_TYPE_IP,
                                   ipv4_src=src_ip,
                                   ipv4_dst=dst_ip)
            
            # Drop the matching packets
            actions = []  # Empty action list means drop
            
            # Add flow with high priority and timeout
            self.add_flow(dp, 100, match, actions, idle_timeout=300, hard_timeout=600)
            
            self.logger.info(f"Blocked traffic from {src_ip} to {dst_ip} on switch {dp.id} for 10 minutes")