#include "ns3/applications-module.h"
#include "ns3/basic-energy-source.h"
#include "ns3/core-module.h"
#include "ns3/csma-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/netanim-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/simple-device-energy-model.h"
#include "ns3/ssid.h"
#include "ns3/wifi-radio-energy-model.h"
#include "ns3/yans-wifi-helper.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("HW2_Task1_Team_ 2");

int main(int argc, char* argv[]){
   
    uint nWifi=5;
    bool useRtsCts=false;
    bool verbose=false;
    bool useNetAnim=false; 
   
    CommandLine cmd(__FILE__);
    cmd.AddValue("useRtsCts", "se settato a true viene abilitato l'utilizzo di handshake rts/cts",useRtsCts);
    cmd.AddValue("verbose", "se settato a true abilita l'uso di logs per server e client udp echo",verbose);
    cmd.AddValue("useNetAnim", "se settato a true vengono generati i file relativi a NetAnim",useNetAnim);
    
    cmd.Parse(argc,argv);
 
    UintegerValue threshold=useRtsCts? UintegerValue(100): UintegerValue(2346); //due valori minori o maggiori di 512 bytes
    Config::SetDefault("ns3::WifiRemoteStationManager::RtsCtsThreshold",threshold);
    

    if(verbose){
    LogComponentEnable ("UdpEchoClientApplication", LOG_LEVEL_INFO);
	LogComponentEnable ("UdpEchoServerApplication", LOG_LEVEL_INFO);
    }
    NodeContainer wifiNodes;

    wifiNodes.Create(nWifi); //container dei nodi della rete presa in considerazione

    YansWifiChannelHelper channel=YansWifiChannelHelper:: Default(); //utilizzo questa istanza di ChannelHelper per creare il canale che verrà usato da un PhyHelper (strato fisico)
    YansWifiPhyHelper phy; //permette di settare i parametri relativi allo strato fisico
    phy.SetChannel(channel.Create());

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211g); //specifico lo standard wifi
    wifi.SetRemoteStationManager("ns3::AarfWifiManager"); //specifico algoritmo AARF
    WifiMacHelper mac; //helper per i settaggi dello strato 2
    mac.SetType("ns3::AdhocWifiMac","QosSupported",BooleanValue(false)); //specifico che la quality of service non è supportata
    
    NetDeviceContainer devices; 
    devices=wifi.Install(phy,mac, wifiNodes); //installo su tutti i nodi della rete lo strato fisico specificato e lo strato di collegamento

    InternetStackHelper stack;
    stack.Install(wifiNodes);

    Ipv4AddressHelper address;
    address.SetBase("192.168.1.0","255.255.255.0");
    Ipv4InterfaceContainer wifiInterface;
    wifiInterface=address.Assign(devices);

    uint16_t port0=20;
    UdpEchoServerHelper echoServer(port0); //n0 sulla porta 20
    ApplicationContainer appEchoServer = echoServer.Install(wifiNodes.Get(0));

    UdpEchoClientHelper echoClient4(wifiInterface.GetAddress(0),port0);
    echoClient4.SetAttribute("MaxPackets",UintegerValue(2));
    echoClient4.SetAttribute("PacketSize",UintegerValue(512));
    echoClient4.SetAttribute("Interval", TimeValue(Seconds(1.0)));
    UdpEchoClientHelper echoClient3(wifiInterface.GetAddress(0),port0);
    echoClient3.SetAttribute("PacketSize",UintegerValue(512));
    echoClient3.SetAttribute("MaxPackets",UintegerValue(2));
    echoClient3.SetAttribute("Interval", TimeValue(Seconds(2.0)));

    ApplicationContainer appEchoClient4=echoClient4.Install(wifiNodes.Get(4));

    ApplicationContainer appEchoClient3=echoClient3.Install(wifiNodes.Get(3));

    appEchoClient4.Start(Seconds(1.0));
    
    appEchoClient3.Start(Seconds(2.0));

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();
    
    Simulator::Stop(Seconds(7.0)); //settare timer simulazione, provato a togliere la simulazione non termina, 3 secondi dall'ultimo invio
    MobilityHelper mobility;
    mobility.SetPositionAllocator("ns3::GridPositionAllocator",
                                  "MinX",
                                  DoubleValue(0.0),
                                  "MinY",
                                  DoubleValue(0.0),
                                  "DeltaX",
                                  DoubleValue(5.0),
                                  "DeltaY",
                                  DoubleValue(10.0),
                                  "GridWidth",
                                  UintegerValue(3),
                                  "LayoutType",
                                  StringValue("RowFirst"));
                                  
    mobility.SetMobilityModel("ns3::RandomWalk2dMobilityModel",
                              "Bounds",
                              RectangleValue(Rectangle(-90, 90, -90, 90)));
    mobility.Install(wifiNodes);
    std::string onOff=useRtsCts? "on":"off"; 
    
    
    if(!useNetAnim){
    phy.EnablePcap("task1-"+onOff+"-2.pcap",devices.Get(2),true,true); //pcap nodo 2, vanno generati gli altri
	Simulator::Run();
	Simulator::Destroy();
	return 0;
    }
    //else do: per ora da problemi scrivere un if useNetAnim
    AnimationInterface anim("wireless-task1-rts-"+onOff+".xml");
    for(uint32_t i=0;i<wifiNodes.GetN();i++){
        if(i==0){
            anim.UpdateNodeDescription(wifiNodes.Get(i),"SRV-0");
            anim.UpdateNodeColor(wifiNodes.Get(i),255,0,0);
        }
        else if(i==4){
            anim.UpdateNodeDescription(wifiNodes.Get(i),"CLI-4");
            anim.UpdateNodeColor(wifiNodes.Get(i),0,255,0);
        }
        else if(i==3){
            anim.UpdateNodeDescription(wifiNodes.Get(i),"CLI-3");
            anim.UpdateNodeColor(wifiNodes.Get(i),0,255,0);
        }
        else{
            anim.UpdateNodeDescription(wifiNodes.Get(i),"HOC-"+std::to_string(i));
            anim.UpdateNodeColor(wifiNodes.Get(i),0,0,255);
        }
    }
    
    anim.EnablePacketMetadata();
    anim.EnableWifiMacCounters(Seconds(0), Seconds(10));
    anim.EnableWifiPhyCounters(Seconds(0), Seconds(10)); //forse non necessari
    
    phy.EnablePcap("task1-" + onOff +"-2.pcap",devices.Get(2),true,true); //pcap nodo 2, vanno generati gli altri
    /* per abilitare i pcap degli altri nodi decommentare
    phy.EnablePcap("task1-" + onOff + "-0.pcap",devices.Get(0),true,true);
    phy.EnablePcap("task1-" + onOff + "-1.pcap",devices.Get(1),true,true);
    phy.EnablePcap("task1-" + onOff +"-3.pcap",devices.Get(3),true,true);
    phy.EnablePcap("task1-" + onOff +"-4.pcap",devices.Get(4),true,true);
    */
    
    
    Simulator::Run();
    Simulator::Destroy();
    return 0;
}
