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

class HostTracked (Event):
    def __init__ (self, packet):
        Event.__init__(self)
        self.packet = packet
        
#KEEPS TRACK OF THE ACCESS POINT WHERE THE HOST IS CONNECTED
class HostTracking (EventMixin):

    _eventMixin_events = set([HostTracked])

    def __init__(self):
        self.position_tracking = None
        core.openflow.addListeners(self)

    
    def _handle_PacketIn(self, event):
        packet = event.parsed
        linksList = core.linkDiscovery.links #Get links from discovery module
        addresses = ["00:11:22:33:44:55"]
        
	#Recreate list of all MAC addresses 
        for l in linksList:
            sid = linksList[l].sid1
            interface = linksList[l].port1
            res = "00:00:00:00:00:" + str(interface) + "" + str(sid)
            if res not in addresses:
            	addresses.append(res)

	#The host will be accessing the network from the AP MAC not present in the links
        if packet.src not in addresses:
            print(packet.src)
            self.position = (packet.src.toStr(), packet.src.toStr().split(':')[5][1], packet.src.toStr().split(':')[5][0])
            print(f"Mobile host is connected to S{packet.src.toStr().split(':')[5][1]}, on the interface {packet.src.toStr().split(':')[5][0]} ")
            self.raiseEvent(HostTracked(packet))
          
        return        
            

def launch():
    ht = HostTracking()
    core.register("host_tracking", ht)
