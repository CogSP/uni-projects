import pox.openflow.libopenflow_01 as of
from pox.core import core
from pox.lib.recoco import Timer
from pox.lib.revent.revent import EventMixin
from pox.lib.revent.revent import Event
from pox.lib.addresses import EthAddr
from pox.lib.packet.ethernet import ethernet
import pox.lib.packet as pkt
from pox.lib.packet.lldp import lldp
from pox.lib.util import dpidToStr

#Auxiliary Graph class to abstract the graph topology and compute shortest path
class Graph:

	def __init__(self):
		self.edges = [] 
	
	#Construct graph using edges list
	def add_edges_from(self, l):
		self.edges = l
		
	#Returns all nodes of graph
	def nodes(self):
		n = []
		for e in self.edges:
			if e[0] not in n:
				n.append(e[0])
			elif e[1] not in n:
				n.append(e[1])
		return n
	
	#Returns all edges of node <node>
	def get_edges(self, node): 
		edges = []
		for n in self.nodes():
			if (node, n) in self.edges and (node, n) not in edges:
				edges.append((node, n))
			if (n, node) in self.edges and (n, node) not in edges:
				edges.append((n, node))
		return edges
		
	#Does Dijkstra
	def shortest_path(self, source, target):
		print("SRC IS: ", source)
		print("Target is: ", target)
		distances = {}
		
		#Define initial distance costs
		for node in self.nodes():
			if node == source:
				distances[node] = 0
			else:
				distances[node] = float('inf')
		predecessors = {}

		
		visited = set()
		while visited != set(self.nodes()):
		   
			min_node = None
			min_distance = float('inf')
			
			#Get min distance node if there is one and mark it as visited, when there isn't one it means the algorithm has finished
			for node in self.nodes():
				if node not in visited and distances[node] < min_distance:
					min_node = node
					min_distance = distances[node]

			if min_node is None:
				break

			visited.add(min_node) #at the start visit node n0, then others that have minimal distance
			edges = self.get_edges(min_node) #get node edges
			
			#Get neighbors of current minimal distance node and update neighbours distances and predecessors of neighbours will be the current min distance node (we use that to retrieve the path at the end)
			for edge in edges:
				neighbor = edge[1] if edge[0] == min_node else edge[0]
				new_distance = distances[min_node] + 1  
				if new_distance < distances[neighbor]:
					distances[neighbor] = new_distance
					predecessors[neighbor] = min_node
		#Finally construct Shortest Path
		shortest_path = []
		current_node = target 
		#Reconstruct path from target to source
		while current_node != source:
			if current_node not in predecessors:
				return None #No path could be found (never happens in our case)
			shortest_path.insert(0, current_node)
			current_node = predecessors[current_node] #new current node will be predecessor of current node

		shortest_path.insert(0, source) 

		return shortest_path

class UserMobility:
	
	#Start reading from handle connection up to better understand the overall flow!
	def __init__(self):
		self.marked_nodes = []
		core.openflow.addListeners(self)
		self.switches = None #keys are all switches with a "Ports" list as value the main attributes of these "ports" are the name (s1,s2,...)
		self.links = [] #list of all link objects found by the other module
		self.connections = {} #key is dpid connection object is value
		self.graph = None #abstracted graph
		self.gw_dpid = None 
		self.current_start_node = None
		self.fake_host_mac = EthAddr("11:22:33:44:55:66") #This fake addr is the "host" mac matched by all flow rules in the path in the internal SDN network
		self.current_shortest_path = []
	
	#Installs a flow rule that redirects packets from dl_addr source to out_port
	def install_flow_rule(self, out_port, dl_addr, conn):
		msg = of.ofp_flow_mod()
		msg.priority = 50000
		match = of.ofp_match(dl_src = dl_addr, dl_type = ethernet.IP_TYPE) #must be IP and with correct source MAC of the path
		msg.match = match
		action = of.ofp_action_output(port=out_port) #redirect packet on out_port
		msg.actions.append(action)
		conn.send(msg)
		
	#function to install flow rules in the correct order		
	def sort_and_install(self, sp):
		#OVERALL IDEA: ANALYZE NODES OF THE SHORTEST PATH ONE AT A TIME AND INSTALL THE FLOW RULES BY GETTING THE CORRECT PORTS
		#MACS ARE FAKE AND FIXED FOR 
		for i in range(1, len(sp)-1): #exclude Access point and gateway that need to be configured separately

			#We analyze prev<->curr<->next nodes, where if prev is the host, it's not considered
			prev = None
			if (i - 1 >= 0):
				prev = sp[i-1]
			curr = sp[i]
			next = sp[i+1]
			
			curr_next = None
			#Get link representing curr<->next
			for link in self.links:
				if link.name1 == curr and link.name2 == next:
					curr_next = link
					break
			
			prev_curr = None
			#Get link representing prev<->curr
			for link in self.links:
				if link.name1 == prev and link.name2 == curr:
					prev_curr = link
					break #if prev is host this remains None
			
			if curr_next == None:
				print("Error links not found")
				exit(1) #should not happen

			#We get the <-curr-> ports directions in the shortest path and declare MACs to match for the rules, especially MAC of host
			h_mac = self.fake_host_mac #The networks sees the MAC of the host as a fake address
			if prev_curr == None:
				#This means we have the host as previous node in the shortest path
				port_out_to_prev = event.port
				print(f"Dealing with host with mac {h_mac} and port to it {port_out_to_prev}")
			else:
				port_out_to_prev = prev_curr.port2

			port_out = curr_next.port1
			dst_mac = curr_next.mac2
			
			#get curr connection (flow rules will be installed on current inspected node)
			conn = None
			dpid_curr = curr_next.raw_dpid1
			for k in self.connections.keys():
				if k == dpid_curr:
					conn = self.connections[dpid_curr]
					break	
		
			#Should never happen				
			if port_out == None or port_out_to_prev == None or conn == None:
				print("Error information not found!")
				exit(1)
			
			dst_mac = EthAddr("00:00:00:00:00:11") #Fake address to match outer web
				
			#Install flow rules for curr -> next and next -> curr packet exchange		 
			self.install_flow_rule(port_out, h_mac, conn) #From curr to next we match the fake address (won't be changed on update)
			self.install_flow_rule(port_out_to_prev, dst_mac, conn) #From next back to curr we match outer web address (again, won't be changed either on path change, as these MACs are the fixed Matched macs)
			print(f"Installed flow rule on {curr} to redirect packet on output port {port_out} when receiving it from MAC address {h_mac}")
			print(f"Installed flow rule on {curr} to redirect packet on output port {port_out_to_prev} when receiving it from MAC address {dst_mac}\n")
	
	#Handle gateway separately: it's flow rules are always the same, we configure it once
	def gateway_rules_install(self, sp):
		gw = sp[-1]
		l = None
		for link in self.links:
			if link.name2 == gw: #link (s2, s5)
				l = link
				break
		conn = None
		port = l.port2
		dpid_gw = l.raw_dpid2
		for k in self.connections.keys():
			if k == dpid_gw:
				conn = self.connections[dpid_gw]
				break
			
		#Install flow rule gw->internet
		msg = of.ofp_flow_mod()
		msg.priority = 50000
		match = of.ofp_match(in_port=port) 
		msg.match = match
		change_mac = of.ofp_action_dl_addr.set_dst(EthAddr("00:00:00:00:00:11")) #Change mac to Internet mac so that internet node will accept the packet
		
		action = of.ofp_action_output(port=2) #Redirect packet on out_port
		msg.actions.append(change_mac)
		msg.actions.append(action)
		conn.send(msg)
		
		#Install flow rule Internet->gw->s2
		msg2 = of.ofp_flow_mod()
		msg2.priority = 50000
		match2 = of.ofp_match(in_port=2) 
		msg2.match = match2
		change_mac2 = of.ofp_action_dl_addr.set_src(self.fake_host_mac)  #Change mac to host mac to have the flow rules work
		action2 = of.ofp_action_output(port=port)
		msg2.actions.append(action2)
		conn.send(msg2)
	
	#Access point flow rules are different, they need to account for the host	
	def AP_rules_install(self, event, sp):
		ap = sp[0] #access point is always first node of sp
		next = sp[1]
		l = None #get link ap->next so that we get output redirect port
		for link in self.links:
			if link.name1 == ap and link.name2 == next:
				l = link
				break
		conn = None
		out_port = l.port1
		dpid_ap = l.raw_dpid1
		for k in self.connections.keys():
			if k == dpid_ap:
				conn = self.connections[dpid_ap]
				break
		
		
		#Install flow rule AP->rest of network
		msg = of.ofp_flow_mod()
		msg.priority = 50000
		match = of.ofp_match(dl_src = event.parsed.src, dl_type = ethernet.IP_TYPE) #We match packet coming source MAC address (the real one)
		msg.match = match
		change_mac = of.ofp_action_dl_addr.set_src(self.fake_host_mac) #Change mac so that SDN network rules match the fake address of host
		action = of.ofp_action_output(port=out_port) #Port is port towards the next node after the AP
		msg.actions.append(change_mac) #Change mac
		msg.actions.append(action) #And redirect packet to correct port
		conn.send(msg)
		
		#Install flow rule AP->host
		msg2 = of.ofp_flow_mod()
		msg2.priority = 50000
		match2 = of.ofp_match(dl_src = EthAddr("00:00:00:00:00:11"),  dl_type = ethernet.IP_TYPE) 
		msg2.match = match2
		change_mac2 = of.ofp_action_dl_addr.set_dst(event.parsed.src) #Change mac dst so that Host will accept it
		action2 = of.ofp_action_output(port=event.ofp.in_port) #Port is port of event (AP port matching the host)
		msg2.actions.append(change_mac2)
		msg2.actions.append(action2)
		conn.send(msg2)
		print(f"Installed flow rules on AP {ap} and port to host is {event.ofp.in_port}\n")
	
	def AP_wipe(self): #We need to delete flow rules on AP every time user changes path, otherwise the old rules might prevent the new ones from being installed
		#Find old AP connection and delete all flow rules from AP
		ap = self.marked_nodes[0]
		print(f"Node {ap} was wiped of its flow rules")
		next = self.marked_nodes[1]

		dpid_ap = None
		for link in self.links:
			if link.name1 == ap and link.name2 == next:
				dpid_ap = link.raw_dpid1
				break
		
		conn = None
		for k in self.connections.keys():
			if k == dpid_ap:
				conn = self.connections[dpid_ap]
				break
		
		msg = of.ofp_flow_mod()
		msg.command = of.OFPFC_DELETE #of.ofp_flow_mod_command.OFPFC_DELETE
		conn.send(msg)
	
	#Start of the code workflow		 		
	def _handle_ConnectionUp(self, event):
		dpid = event.dpid
		self.connections[dpid] = event.connection
		
	def _handle_PacketIn(self, event):
		packet = event.parsed
		src_addr = packet.src
		linksList = core.linkDiscovery.links
		addresses = ["00:11:22:33:44:55"]
		
		#Get the links addresses of the switches!
		for l in linksList:
			sid = linksList[l].sid1
			interface = linksList[l].port1
			addresses.append("00:00:00:00:00:" + str(interface) +""+ str(sid))
		
		#Only the host packets will trigger a change, not the switches ones!
		if packet.src in addresses:
			return 

		if packet.find('ipv4') == None:
			return #fake gateway handles ARP

		print("Event raised by, ",dpidToStr(event.connection.dpid))
		
		#CASE 1: FIRST TIME ROUTING INITIALIZATION
		#GET GRAPH AND ALL THE RELEVANT STRUCTURES IF IT'S THE FIRST ROUTING CONFIGURATION
		if len(self.marked_nodes) == 0:
			tup = core.linkDiscovery.getLinks()
			self.switches = tup[0] #switches
			for i in tup[1]:
				self.links.append(tup[1][i]) #save all links
			
			#Create graph
			links_ = []
			for link in self.links:
				links_.append(("s"+str(link.sid1),"s"+str(link.sid2)))
			self.graph = Graph()
			self.graph.add_edges_from(links_)
			
			#Get gateway dpid
			for i in self.links:
				if i.name1 == 's5':
					self.gw_dpid = i.raw_dpid1
					break
			
			#Get source node as the first switch the packet traverses
			#The algorithm should assign flow rules for everyone with the first AP traversal
			#We use names as s1, s2, etc... as identifiers of nodes
			src = ''
			for i in self.switches[event.dpid]:
				if i.port_no == 65534:
					src = i.name
		
			#Base case: no configured path exists, the internet is the goal, calculate shortest path to gateway
			sp = self.graph.shortest_path(source=src, target='s5') #get shortest path
			self.current_start_node = event.connection.dpid #start is AP

			print("Shortest path is: ", sp)
			self.current_shortest_path = sp #memorize shortest path for when we need to update it later
				
			#Install flow rules on all path nodes
			self.sort_and_install(sp)
			self.gateway_rules_install(sp)
			self.AP_rules_install(event, sp)
			for i in sp:
				#mark nodes as configured
				self.marked_nodes.append(i)
					
		#CASE 2: USER CHANGED INTERFACE BUT NOT AP
		elif event.connection.dpid == self.current_start_node:
			print("User changed interface but not AP, reconfiguring only AP")
			#We just reconfigure AP to handle the other interface instead
			self.AP_wipe()
			self.AP_rules_install(event, self.current_shortest_path)
		
		#CASE 3: USER CHANGED BOTH INTERFACE AND AP		
		#If the user changed AP node we apply algorithm to find the less costly path in terms of rule reconfiguration	
		#Cost modeling: the already configured path has cost 1, since it is already configured and we just need to reconfigure the initial node of the old path to accept from the new port the  packets, whilst every new node we have to configure on a path adds a cost of 1.
		#The idea is to calculate the shortest path, and, if the sp uses the old path, then we readjust the new path so that it will use as much of the old path as possible, since it's already configured
		else:
			print("Recalculating path...")
			#Get source node as the first switch the packet traverses
			src = ''
			for i in self.switches[event.dpid]:
				if i.port_no == 65534:
					src = i.name
			graph = self.graph
			new_sp = graph.shortest_path(source=src, target='s5')
			
			old_sp = []
			for i in new_sp:
				if i in self.marked_nodes:
					old_sp.append(i)
			print("New sp is ", new_sp)
			
			#Old path was used
			if len(old_sp) > 0:
				self.AP_wipe() #Takes current AP on marked nodes and removes all rules, we remove rules on old AP
				
				#The old path was used so we adjust the new shortest path to reuse as much as possible of the old already configured path
				path = []
				for new in new_sp:
					if new not in self.marked_nodes:
						path.append(new)
				
				linking_point = None #earliest link between new and old path, self.marked_nodes and new_sp are ordered already

				for n in self.marked_nodes:
					for new in new_sp:
						#We link the first new path node we can find to the old path
						if (new, n) in graph.get_edges(n) and new not in self.marked_nodes and n in self.marked_nodes:
							linking_point = (new, n)
							break
					if linking_point:
						break
				
				#Handle case in which new AP is too part of the old path (it was a switch in the old path): we just reuse AP 
				if linking_point == None:
					linking_point = (None, None) #In this case the sp calculated is the actual adjusted sp, we just need to reconfigure the AP
				else:
					print("Linking point is ", linking_point)
				
				#We find first instance of a link between old path and a node of the new path
				path = []
				for n in new_sp:
					if n == linking_point[0]:
						path.append(n)
						break
					else:
						path.append(n)
				
					
				print("Old path was(marked nodes): ", self.marked_nodes)	
				#And get the real path as the shortest path untile that link and the rest of the old path
				for i in range(len(self.marked_nodes)):
					if self.marked_nodes[i] == linking_point[1]:
						path += [self.marked_nodes[i]] #We add the second part of the linking point
						path += [self.marked_nodes[i+1]] #And the node next to that to configure the linking point correctly
						print("Path before sort is ", path)
						self.sort_and_install(path) 
						path += self.marked_nodes[i+2:] #We add rest of old path
						break 
						#All of this is ignored if new AP is part of old path
						
				print("Final path is ", path)
				
				#Update current start AP and sp
				self.current_start_node = event.connection.dpid
				self.current_shortest_path = path
				
				self.marked_nodes = path #Update marked nodes
				
				#Now we just configure new AP
				self.AP_wipe()
				self.AP_rules_install(event, path)
				
					
			else:
				print("Shortest path doesn't use nodes of old path")
				#Doesn't happen in our topology but this case is simpler anyway, we just configure the new path as a first path 
				self.sort_and_install(new_sp)
				self.AP_rules_install(event, new_sp)
				self.gateway_rules_install(new_sp)
				
			
	
def launch():
	core.registerNew(UserMobility)
