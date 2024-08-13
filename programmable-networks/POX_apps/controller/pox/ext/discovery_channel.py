import pox.openflow.libopenflow_01 as of
from pox.core import core
from pox.lib.recoco import Timer
from pox.lib.revent.revent import EventMixin
from pox.lib.revent.revent import Event
from pox.lib.addresses import EthAddr
from pox.lib.packet.ethernet import ethernet
from pox.lib.packet.arp import arp
from pox.lib.packet.lldp import lldp
from pox.lib.util import dpidToStr
#import numpy as np


class Link():
	#Memorize all useful info in link object
	def __init__(self, sid1, sid2, dpid1, port1, dpid2, port2, mac1, mac2):
		self.name = "s" + str(sid1) + "_" + "s" + str(sid2)
		self.sid1 = sid1
		self.sid2 = sid2
		self.dpid1 = dpidToStr(dpid1)
		self.dpid2 = dpidToStr(dpid2)
		self.port1 = int(port1)
		self.port2 = int(port2)
		self.mac1 = mac1
		self.mac2 = mac2
		self.name1 = "s" + str(sid1)
		self.name2 = "s" + str(sid2)
		self.raw_dpid1 = dpid1
		self.raw_dpid2 = dpid2

class linkDiscovery():

	def __init__(self):
		self.switches = dict() # <key: dpid; value: list of switch's ports> to handle send probes and extract SIDs
		self.links = {} # list of link objects the final objective of the module is to populate this
		self.switch_id = {} # <key: a progressive ID; value: the dpid of the switch> to send probes
		self.switch_MACs = {} #<dpid : MAC> to get MACs to insert into the links more easily
		core.openflow.addListeners(self)
		Timer(15, self.sendProbes, recurring=False)
		

	def _handle_ConnectionUp(self, event):
		id_ = None
		#Find id of the device connected:
		for port in event.ofp.ports:
			if port.port_no == 65534:
				name = str(port.name)
				id_ = int(name[1]) #We extract id from the device information in the event
		
		self.switch_id[id_] = event.dpid
		self.switches[event.dpid] = event.ofp.ports
		
		#Memorize dpid as key and list of (port, MAC) tuple of the switch
		self.switch_MACs[dpidToStr(event.dpid)] = []
		for i in event.ofp.ports:
			self.switch_MACs[dpidToStr(event.dpid)].append((i.port_no ,i.hw_addr))

	def _handle_PacketIn(self, event):
		eth_frame = event.parsed
		if eth_frame.src == EthAddr("00:11:22:33:44:55"):
			eth_dst = eth_frame.dst.toStr().split(':') #We have on the last number of MAC <SID|PORT of mac>
			sid1 = int(eth_dst[5][1])
			dpid1 = self.switch_id[sid1]
			port1 = int(eth_dst[5][0])
			dpid2 = event.dpid
			sid2 = ""
			
			#Extract sid2	
			for port in self.switches[dpid2]:
				if port.port_no == 65534:
					sid2 = str(port.name[-1])
					
			port2 = event.ofp.in_port
			mac1 = None
			mac2 = None
			for k in self.switch_MACs.keys():
				if k == dpidToStr(dpid1):
					#When we find dpid1 get MAC 1
					for j in self.switch_MACs[k]:
						if j[0] == port1:
							mac1 = j[1]
				elif k == dpidToStr(dpid2):
					#Same for MAC 2
					for j in self.switch_MACs[k]:
						if j[0] == port2:
							mac2 = j[1]
			#Extract MACs
			link = Link(sid1, sid2, dpid1, port1, dpid2, port2, mac1, mac2)
			dpid1 = dpidToStr(dpid1)
			dpid2 = dpidToStr(dpid2)
			print(f"Added link s{sid1}-s{sid2} with ports {port1}-{port2} and macs {mac1}---{mac2} and dpids {dpid1}  {dpid2}\n")
			if link.name not in self.links:
				self.links[link.name] = link
				
				
	
	def sendProbes(self):
		#Send the probkes to discover channels
		for sid in self.switch_id:
			dpid = self.switch_id[sid]
			name = ""
			for port in self.switches[dpid]:
				if port.port_no == 65534:
					name = str(port.name[-1])

			for port in self.switches[dpid]:
				# the 65534 port connects the dataplane with the control plane
				if port.port_no != 65534:
					mac_src = EthAddr("00:11:22:33:44:55")
					mac_dst = EthAddr("00:00:00:00:00:" + "" + str(port.port_no) + "" + name)  
					ether = ethernet()
					ether.type = ethernet.ARP_TYPE
					ether.src = mac_src
					ether.dst = mac_dst
					ether.payload = arp()
					msg = of.ofp_packet_out()
					msg.data = ether.pack()
					msg.actions.append(of.ofp_action_output(port = port.port_no))
					core.openflow.sendToDPID(dpid, msg)
	def getLinks(self):
		return (self.switches, self.links, self.switch_MACs, self.switch_id)

def launch():
	core.registerNew(linkDiscovery)
