# -*- coding: UTF-8 -*-
"""
Ryu SDN Controller for AMI Smart Grid.

Features:
- Simple Layer 2 Learning Switch functionality.
- Periodically requests and logs flow statistics from connected switches.
- Logs flow data to a CSV file for the Anomaly Detection System.
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
        self.flow_start_times = {} # (dpid, src_ip, dst_ip, src_port, dst_port, proto) -> timestamp

    def _setup_logging(self):
        """Initializes the CSV log file and writer."""
        # Define the header based on available stats and target features
        # IMPORTANT: These columns MUST align with what the anomaly_detector.py expects
        # We can only reliably get L2-L4 info here. Other fields need defaults.
        self.column_names = [
            'ts', 'datapath_id', 'src_ip', 'src_port', 'dst_ip', 'dst_port', 'proto',
            'duration', 'src_bytes', 'dst_bytes', # Mapped from byte_count
            'src_pkts', 'dst_pkts', # Mapped from packet_count
            # --- Fields below need default values as Ryu doesn't provide them directly ---
            'service', 'conn_state', 'missed_bytes', 'src_ip_bytes', 'dst_ip_bytes',
            'dns_query', 'dns_qclass', 'dns_qtype', 'dns_rcode', 'dns_AA', 'dns_RD', 'dns_RA',
            'dns_rejected', 'ssl_version', 'ssl_cipher', 'ssl_resumed', 'ssl_established',
            'ssl_subject', 'ssl_issuer', 'http_trans_depth', 'http_method', 'http_uri',
            'http_referrer', 'http_version', 'http_request_body_len', 'http_response_body_len',
            'http_status_code', 'http_user_agent', 'http_orig_mime_types', 'http_resp_mime_types',
            'weird_name', 'weird_addl', 'weird_notice'
            # 'label', 'type' - These are usually added during analysis/labeling, not logged here
        ]
        # Check if file exists to avoid writing header multiple times
        file_exists = os.path.isfile(FLOW_STATS_LOG_FILE)
        try:
            # Use 'a' to append if file exists, 'w' to write new otherwise
            self.log_file = open(FLOW_STATS_LOG_FILE, 'a', newline='')
            self.csv_writer = csv.DictWriter(self.log_file, fieldnames=self.column_names)
            if not file_exists or os.path.getsize(FLOW_STATS_LOG_FILE) == 0:
                self.csv_writer.writeheader()
                self.log_file.flush() # Ensure header is written immediately
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

        # Install table-miss flow entry
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
            # Ignore LLDP packets
            return

        dst_mac = eth.dst
        src_mac = eth.src

        self.mac_to_port.setdefault(dpid, {})

        # Learn source MAC address to avoid FLOOD next time.
        self.mac_to_port[dpid][src_mac] = in_port
        # self.logger.info(f"Packet in dpid={dpid:016x} src={src_mac} dst={dst_mac} in_port={in_port}")


        # Determine output port
        if dst_mac in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst_mac]
        else:
            out_port = ofproto.OFPP_FLOOD # Flood if destination MAC is unknown

        actions = [parser.OFPActionOutput(out_port)]

        # Install a flow to avoid packet_in next time if destination is known
        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst_mac, eth_src=src_mac)
            # Verify if we should install flows based on buffer_id
            if msg.buffer_id != ofproto.OFP_NO_BUFFER:
                self.add_flow(datapath, 1, match, actions, msg.buffer_id, idle_timeout=20, hard_timeout=100)
                return
            else:
                self.add_flow(datapath, 1, match, actions, idle_timeout=20, hard_timeout=100)

        # Send packet out
        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                  in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)

        # --- Track flow start time for duration calculation ---
        ip_pkt = pkt.get_protocol(ipv4.ipv4)
        tcp_pkt = pkt.get_protocol(tcp.tcp)
        udp_pkt = pkt.get_protocol(udp.udp)

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

            flow_key = (dpid, src_ip, dst_ip, src_port, dst_port, proto)
            if flow_key not in self.flow_start_times:
                 self.flow_start_times[flow_key] = datetime.datetime.now().timestamp()


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
                # Clean up MAC table for the disconnected switch
                if datapath.id in self.mac_to_port:
                    del self.mac_to_port[datapath.id]
                 # Clean up flow start times for the disconnected switch
                keys_to_remove = [k for k in self.flow_start_times if k[0] == datapath.id]
                for key in keys_to_remove:
                    del self.flow_start_times[key]


    def _monitor(self):
        """Periodically request flow statistics from switches."""
        while True:
            for dp in self.datapaths.values():
                self._request_stats(dp)
            hub.sleep(MONITORING_INTERVAL)

    def _request_stats(self, datapath):
        """Send flow statistics request to a datapath."""
        self.logger.debug(f'Sending stats request: {datapath.id:016x}')
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # Request flow stats
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

        # Request port stats (optional, could add port-related features if needed)
        # req = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)
        # datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        """Handle flow statistics replies and log them."""
        body = ev.msg.body
        dpid = ev.msg.datapath.id
        timestamp = datetime.datetime.now().timestamp()

        if not self.csv_writer:
            self.logger.warning("CSV writer not initialized. Skipping log.")
            return

        # self.logger.info('Received Flow Stats Reply from DPID: {:016x}'.format(dpid))
        # self.logger.info('---------------- ----------------- ----------------- -------- -------- --------')

        for stat in sorted([flow for flow in body if flow.priority == 1], # Only log flows installed by controller
                           key=lambda flow: (flow.match['in_port'], flow.match.get('eth_dst'))):

            # Extract basic flow information
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

            # Calculate duration (approximation)
            flow_key = (dpid, src_ip, dst_ip, src_port, dst_port, proto)
            start_time = self.flow_start_times.get(flow_key, timestamp - stat.duration) # Estimate if not found
            duration = timestamp - start_time

            # Create log entry dictionary
            log_entry = {
                'ts': timestamp,
                'datapath_id': f"{dpid:016x}", # Format DPID as hex string
                'src_ip': src_ip,
                'src_port': src_port,
                'dst_ip': dst_ip,
                'dst_port': dst_port,
                'proto': proto,
                'duration': duration, # Use calculated duration
                'src_bytes': stat.byte_count, # Assuming byte_count is mostly src->dst for this flow
                'dst_bytes': 0, # Ryu stats don't easily separate src/dst bytes for a single flow rule
                'src_pkts': stat.packet_count, # Assuming packet_count is mostly src->dst
                'dst_pkts': 0, # Ryu stats don't easily separate src/dst packets
                # --- Default values for fields not directly available ---
                'service': '-',
                'conn_state': 'SF', # Assuming established if stats exist, needs refinement
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
            for col in self.column_names:
                if col not in log_entry:
                    log_entry[col] = 0 if col in ['missed_bytes', 'dst_bytes', 'dst_pkts', 'dst_ip_bytes', 'http_request_body_len', 'http_response_body_len'] else '-'


            # Write to CSV
            try:
                # Create a row respecting the order in self.column_names
                ordered_row = {col: log_entry.get(col) for col in self.column_names}
                self.csv_writer.writerow(ordered_row)
            except Exception as e:
                 self.logger.error(f"Error writing log entry: {e} - Data: {log_entry}")


        # Flush the buffer to ensure data is written to disk
        if self.log_file:
            self.log_file.flush()

    # Optional: Handler for port stats if needed later
    # @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    # def _port_stats_reply_handler(self, ev):
    #     body = ev.msg.body
    #     self.logger.info('datapath         port     '
    #                      'rx-pkts  rx-bytes rx-error '
    #                      'tx-pkts  tx-bytes tx-error')
    #     self.logger.info('---------------- -------- '
    #                      '-------- -------- -------- '
    #                      '-------- -------- --------')
    #     for stat in sorted(body, key=attrgetter('port_no')):
    #         self.logger.info('%016x %8x %8d %8d %8d %8d %8d %8d',
    #                          ev.msg.datapath.id, stat.port_no,
    #                          stat.rx_packets, stat.rx_bytes, stat.rx_errors,
    #                          stat.tx_packets, stat.tx_bytes, stat.tx_errors)

