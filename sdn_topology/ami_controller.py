# -*- coding: UTF-8 -*-
"""
Ryu SDN Controller for AMI Smart Grid. (Version 2)

Features:
- Simple Layer 2 Learning Switch functionality.
- Periodically requests and logs flow statistics from connected switches.
- Logs flow data to a CSV file for the Anomaly Detection System.
- Corrected handling of flow duration from stats replies.
"""

import datetime
import csv
import os
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet
from ryu.lib.packet import ethernet
from ryu.lib.packet import ether_types
from ryu.lib.packet import ipv4
from ryu.lib.packet import tcp
from ryu.lib.packet import udp
from ryu.lib import hub

# --- Configuration ---
FLOW_STATS_LOG_FILE = 'flow_stats.log' # File to log flow statistics
MONITORING_INTERVAL = 10 # Interval in seconds to request flow stats

class AmiSdnController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(AmiSdnController, self).__init__(*args, **kwargs)
        # MAC address table: dpid -> {mac -> port}
        self.mac_to_port = {}
        # Dictionary to store datapath objects: dpid -> datapath
        self.datapaths = {}
        # Start the monitoring thread
        self.monitor_thread = hub.spawn(self._monitor)
        # File handler for logging
        self.log_file = None
        self.csv_writer = None
        self._setup_logging()
        self.logger.info("AMI SDN Controller Started")
        # Store flow start times to calculate duration approx.
        # (dpid, src_ip, dst_ip, src_port, dst_port, proto) -> timestamp
        self.flow_start_times = {}

    def _setup_logging(self):
        """Initializes the CSV log file and writer."""
        self.column_names = [
            'ts', 'datapath_id', 'src_ip', 'src_port', 'dst_ip', 'dst_port', 'proto',
            'duration', 'src_bytes', 'dst_bytes', # Mapped from byte_count
            'src_pkts', 'dst_pkts', # Mapped from packet_count
            'service', 'conn_state', 'missed_bytes', 'src_ip_bytes', 'dst_ip_bytes',
            'dns_query', 'dns_qclass', 'dns_qtype', 'dns_rcode', 'dns_AA', 'dns_RD', 'dns_RA',
            'dns_rejected', 'ssl_version', 'ssl_cipher', 'ssl_resumed', 'ssl_established',
            'ssl_subject', 'ssl_issuer', 'http_trans_depth', 'http_method', 'http_uri',
            'http_referrer', 'http_version', 'http_request_body_len', 'http_response_body_len',
            'http_status_code', 'http_user_agent', 'http_orig_mime_types', 'http_resp_mime_types',
            'weird_name', 'weird_addl', 'weird_notice'
        ]
        file_exists = os.path.isfile(FLOW_STATS_LOG_FILE)
        try:
            self.log_file = open(FLOW_STATS_LOG_FILE, 'a', newline='')
            self.csv_writer = csv.DictWriter(self.log_file, fieldnames=self.column_names)
            if not file_exists or os.path.getsize(FLOW_STATS_LOG_FILE) == 0:
                self.csv_writer.writeheader()
                self.log_file.flush()
            self.logger.info(f"Logging flow stats to {FLOW_STATS_LOG_FILE}")
        except IOError as e:
            self.logger.error(f"Error opening log file {FLOW_STATS_LOG_FILE}: {e}")
            self.log_file = None
            self.csv_writer = None

    def __del__(self):
        """Cleanup resources."""
        if self.log_file:
            self.log_file.close()

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """Handle switch features event."""
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)
        self.logger.info(f"Switch {datapath.id:016x} connected")

    def add_flow(self, datapath, priority, match, actions, buffer_id=None, idle_timeout=0, hard_timeout=0):
        """Add a flow entry to the switch."""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                    priority=priority, match=match,
                                    instructions=inst,
                                    idle_timeout=idle_timeout,
                                    hard_timeout=hard_timeout)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                    match=match, instructions=inst,
                                    idle_timeout=idle_timeout,
                                    hard_timeout=hard_timeout)
        datapath.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        """Handle packet_in events (simple L2 learning switch)."""
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']
        dpid = datapath.id

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        dst_mac = eth.dst
        src_mac = eth.src

        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src_mac] = in_port

        if dst_mac in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst_mac]
        else:
            out_port = ofproto.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port)]

        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst_mac, eth_src=src_mac)
            # --- Track flow start time for duration calculation ---
            # Extract L3/L4 details to create a unique flow key
            ip_pkt = pkt.get_protocol(ipv4.ipv4)
            tcp_pkt = pkt.get_protocol(tcp.tcp)
            udp_pkt = pkt.get_protocol(udp.udp)

            flow_key = None
            if ip_pkt:
                src_ip = ip_pkt.src
                dst_ip = ip_pkt.dst
                proto = ip_pkt.proto
                src_port = 0
                dst_port = 0
                if tcp_pkt:
                    src_port = tcp_pkt.src_port
                    dst_port = tcp_pkt.dst_port
                elif udp_pkt:
                    src_port = udp_pkt.src_port
                    dst_port = udp_pkt.dst_port
                # Create the key used for tracking start times
                flow_key = (dpid, src_ip, dst_ip, src_port, dst_port, proto)

            # Add timestamp only if it's a new flow we haven't seen before in packet-ins
            if flow_key and flow_key not in self.flow_start_times:
                 self.flow_start_times[flow_key] = datetime.datetime.now().timestamp()
                 # Add flow rule with appropriate match fields
                 if ip_pkt and proto == 6: # TCP
                      match = parser.OFPMatch(in_port=in_port, eth_type=ether_types.ETH_TYPE_IP,
                                              ipv4_src=src_ip, ipv4_dst=dst_ip, ip_proto=proto,
                                              tcp_src=src_port, tcp_dst=dst_port)
                 elif ip_pkt and proto == 17: # UDP
                      match = parser.OFPMatch(in_port=in_port, eth_type=ether_types.ETH_TYPE_IP,
                                              ipv4_src=src_ip, ipv4_dst=dst_ip, ip_proto=proto,
                                              udp_src=src_port, udp_dst=dst_port)
                 elif ip_pkt: # Other IP protocols (like ICMP)
                      match = parser.OFPMatch(in_port=in_port, eth_type=ether_types.ETH_TYPE_IP,
                                              ipv4_src=src_ip, ipv4_dst=dst_ip, ip_proto=proto)
                 else: # Non-IP
                      match = parser.OFPMatch(in_port=in_port, eth_dst=dst_mac, eth_src=src_mac)

                 # Install the flow rule
                 if msg.buffer_id != ofproto.OFP_NO_BUFFER:
                     self.add_flow(datapath, 1, match, actions, msg.buffer_id, idle_timeout=20, hard_timeout=100)
                     # Don't send packet out if we added flow with buffer_id
                     return
                 else:
                     self.add_flow(datapath, 1, match, actions, idle_timeout=20, hard_timeout=100)


        # Send packet out if no flow was added or if flooding
        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data
        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                  in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)


    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        """Handle switch connection/disconnection events."""
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.logger.info(f'Register datapath: {datapath.id:016x}')
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.info(f'Unregister datapath: {datapath.id:016x}')
                del self.datapaths[datapath.id]
                if datapath.id in self.mac_to_port:
                    del self.mac_to_port[datapath.id]
                keys_to_remove = [k for k in self.flow_start_times if k[0] == datapath.id]
                for key in keys_to_remove:
                    # Use try-except in case key was somehow removed already
                    try:
                        del self.flow_start_times[key]
                    except KeyError:
                        pass


    def _monitor(self):
        """Periodically request flow statistics from switches."""
        while True:
            for dp in list(self.datapaths.values()): # Use list copy in case datapaths change during iteration
                self._request_stats(dp)
            hub.sleep(MONITORING_INTERVAL)

    def _request_stats(self, datapath):
        """Send flow statistics request to a datapath."""
        self.logger.debug(f'Sending stats request: {datapath.id:016x}')
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        # Request flow stats for table 0
        req = parser.OFPFlowStatsRequest(datapath, 0)
        try:
            datapath.send_msg(req)
        except Exception as e:
            self.logger.error(f"Error sending stats request to {datapath.id:016x}: {e}")
            # Consider removing datapath if send fails repeatedly
            # if datapath.id in self.datapaths:
            #     del self.datapaths[datapath.id]


    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        """Handle flow statistics replies and log them."""
        body = ev.msg.body
        dpid = ev.msg.datapath.id
        timestamp = datetime.datetime.now().timestamp()

        if not self.csv_writer:
            self.logger.warning("CSV writer not initialized. Skipping log.")
            return

        self.logger.debug(f'Received Flow Stats Reply from DPID: {dpid:016x}')

        flows_logged = 0
        for stat in sorted([flow for flow in body if flow.priority == 1], # Only log flows installed by controller
                           key=lambda flow: (flow.match.get('in_port', 0), flow.match.get('eth_dst', ''))):

            # Extract basic flow information from the match fields
            src_ip, dst_ip, src_port, dst_port, proto = "0.0.0.0", "0.0.0.0", 0, 0, 0 # Defaults

            if 'ipv4_src' in stat.match:
                src_ip = stat.match['ipv4_src']
            if 'ipv4_dst' in stat.match:
                dst_ip = stat.match['ipv4_dst']
            if 'ip_proto' in stat.match:
                proto = stat.match['ip_proto']
                if proto == 6 and 'tcp_src' in stat.match: # TCP
                    src_port = stat.match['tcp_src']
                    dst_port = stat.match['tcp_dst']
                elif proto == 17 and 'udp_src' in stat.match: # UDP
                    src_port = stat.match['udp_src']
                    dst_port = stat.match['udp_dst']
                # Add handling for ICMP if needed
                # elif proto == 1: # ICMP
                #    src_port = stat.match.get('icmpv4_type', 0)
                #    dst_port = stat.match.get('icmpv4_code', 0)


            # --- Corrected Duration Calculation ---
            flow_key = (dpid, src_ip, dst_ip, src_port, dst_port, proto)
            flow_duration = 0.0

            # Prefer duration calculated from tracked start time
            if flow_key in self.flow_start_times:
                start_time = self.flow_start_times[flow_key]
                flow_duration = timestamp - start_time
                # Optional: Remove old entries if flow duration exceeds hard timeout?
                # if stat.hard_timeout > 0 and flow_duration > stat.hard_timeout:
                #     del self.flow_start_times[flow_key]
            else:
                # If start time wasn't tracked (e.g., controller restarted),
                # use the duration reported by the switch.
                # This is the time since the flow was installed or last matched.
                flow_duration = stat.duration_sec + stat.duration_nsec * 1e-9
                # Log a warning if we have to rely on switch duration?
                # self.logger.warning(f"Flow key {flow_key} not found in start times. Using switch duration.")

            # --- Create log entry dictionary ---
            log_entry = {
                'ts': timestamp,
                'datapath_id': f"{dpid:016x}",
                'src_ip': src_ip,
                'src_port': src_port,
                'dst_ip': dst_ip,
                'dst_port': dst_port,
                'proto': proto,
                'duration': round(flow_duration, 6), # Use calculated duration, rounded
                'src_bytes': stat.byte_count,
                'dst_bytes': 0, # Approximation
                'src_pkts': stat.packet_count,
                'dst_pkts': 0, # Approximation
                'service': '-',
                'conn_state': 'SF', # Approximation
                'missed_bytes': 0,
                'src_ip_bytes': stat.byte_count, # Approximation
                'dst_ip_bytes': 0, # Approximation
                'dns_query': '-', 'dns_qclass': '-', 'dns_qtype': '-', 'dns_rcode': '-',
                'dns_AA': '-', 'dns_RD': '-', 'dns_RA': '-', 'dns_rejected': '-',
                'ssl_version': '-', 'ssl_cipher': '-', 'ssl_resumed': '-', 'ssl_established': '-',
                'ssl_subject': '-', 'ssl_issuer': '-',
                'http_trans_depth': '-', 'http_method': '-', 'http_uri': '-', 'http_referrer': '-',
                'http_version': '-', 'http_request_body_len': 0, 'http_response_body_len': 0,
                'http_status_code': '-', 'http_user_agent': '-', 'http_orig_mime_types': '-',
                'http_resp_mime_types': '-',
                'weird_name': '-', 'weird_addl': '-', 'weird_notice': '-'
            }

            # Ensure all columns are present, even if just with defaults
            # (This loop might be redundant if defaults are set above, but safe)
            for col in self.column_names:
                if col not in log_entry:
                    log_entry[col] = 0 if col in ['missed_bytes', 'dst_bytes', 'dst_pkts', 'dst_ip_bytes', 'http_request_body_len', 'http_response_body_len'] else '-'

            # Write to CSV
            try:
                ordered_row = {col: log_entry.get(col) for col in self.column_names}
                self.csv_writer.writerow(ordered_row)
                flows_logged += 1
            except Exception as e:
                 self.logger.error(f"Error writing log entry: {e} - Data: {log_entry}")

        self.logger.debug(f"Logged {flows_logged} flow entries for DPID {dpid:016x}")

        # Flush the buffer to ensure data is written to disk
        if self.log_file:
            try:
                self.log_file.flush()
            except Exception as e:
                 self.logger.error(f"Error flushing log file: {e}")

