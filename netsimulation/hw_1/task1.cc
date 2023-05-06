#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/csma-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/ipv4-global-routing-helper.h"
#include <iostream>
#include <string>

//HOMEWORK 3, TEAM 2 (Leader: Camilla Santoro, mat. 1933643)

using namespace ns3;
NS_LOG_COMPONENT_DEFINE ("Task_1_Team_2"); //Definiamo la stringa con cui far partire le info di logging

int main (int argc, char *argv[]){
int configuration = 0; //il parametro da cui dipenderà l'esecuzione di una delle 3 configurazioni

CommandLine cmd (__FILE__); 
cmd.AddValue("configuration", "Scegliere configurazione tra 0, 1, 2",configuration); //Runnare il programma usando il seguente comando: ./ns3 run "task1 --configuration=num_config" con il numero desiderato al posto di num_config
cmd.Parse (argc, argv); 

LogComponentEnable("Task_1_Team_2", LOG_LEVEL_INFO); //Abilitiamo le stringhe di logging che andremo a specificare con NS_LOG_INFO
NS_LOG_INFO("Costruendo la topologia..");

uint32_t nCSMA1 = 3; //numero di nodi nella CSMA 1
uint32_t nCSMA2 = 2; //numero di nodi extra nella CSMA 2, oltre ad n6 (quindi ovviamente in totale sono sempre 3)

/*NOTA: Per fare in modo che l'ordine degli id dei nodi venga rispettato, nella seconda CSMA faremo
  prima la add di n6 e poi la Create(nCSMA2), questo è il motivo per cui tale parametro è a 2 anzichè 3.
*/

//CSMA 1 con n3 -> n0 n1 n2, n3 in p2p n1
NodeContainer csmaNodes1;
csmaNodes1.Create (nCSMA1); //crea tutti e tre i nuovi nodi n0,n1,n2 e li aggiunge alla csma1
Ptr<Node> n0 = csmaNodes1.Get(0); //Puntatori utili per utilizzare i nodi nel resto del programma
Ptr<Node> n1 = csmaNodes1.Get(1);
Ptr<Node> n2 = csmaNodes1.Get(2);

//interfaccia p2p I0: n1--n3
NodeContainer p2pNodes0; //l'ordine dei nodi è dato dalle interfacce I0, I1... scritte sul pdf
p2pNodes0.Create (1); //crea il nodo n3 e lo aggiunge alla interfaccia 0
Ptr<Node> n3 = p2pNodes0.Get(0);
p2pNodes0.Add(n1); //aggiunge il nodo n1 (centro della CSMA 1) alla interfaccia 0

CsmaHelper csmaHelper1; //definisco e setto gli attributi dell'helper della csma1
csmaHelper1.SetChannelAttribute ("DataRate", StringValue ("25Mbps"));
csmaHelper1.SetChannelAttribute ("Delay", TimeValue (MicroSeconds (10)));
PointToPointHelper pointToPoint0; //helper della interfaccia 0
pointToPoint0.SetDeviceAttribute ("DataRate", StringValue ("80Mbps")); //setto gli attributi della interfaccia 0
pointToPoint0.SetChannelAttribute ("Delay", TimeValue(MicroSeconds(5)));

//interfaccia I2: n4--n5
NodeContainer p2pNodes2;
p2pNodes2.Create(2); //Creaiamo la p2p tra n4 ed n5 (interfaccia I2)
Ptr<Node> n4 = p2pNodes2.Get(0);
Ptr<Node> n5 = p2pNodes2.Get(1);

PointToPointHelper pointToPoint2; //definisco e setto gli attributi dell'helper della interfaccia 2
pointToPoint2.SetDeviceAttribute("DataRate", StringValue ("80Mbps"));
pointToPoint2.SetChannelAttribute ("Delay", TimeValue (MicroSeconds (5)));

//interfaccia I1: n3--n6
NodeContainer p2pNodes1;
p2pNodes1.Create(1); //creiamo n6 e lo aggiungiamo alla interfaccia 1
Ptr<Node> n6 = p2pNodes1.Get(0);
p2pNodes1.Add(n3); //aggiungiamo n3 all'interfaccia 1

//CSMA 2 con n3 -> n6 n7 n8, n3 in p2p con n6
NodeContainer csmaNodes2; 
csmaNodes2.Add(n6); //aggiunge il nodo n6 al container della csma2
csmaNodes2.Create(nCSMA2); //"aggiunge" i nodi extra n7 ed n8 e crea la lan
Ptr<Node> n7 = csmaNodes2.Get(1);
Ptr<Node> n8 = csmaNodes2.Get(2);


CsmaHelper csmaHelper2; //Si settano gli attributi della csma2 
csmaHelper2.SetChannelAttribute ("DataRate", StringValue ("30Mbps"));
csmaHelper2.SetChannelAttribute ("Delay", TimeValue (MicroSeconds (20)));


PointToPointHelper pointToPoint1; //si settano gli attributi sulla interfaccia 1
pointToPoint1.SetDeviceAttribute("DataRate", StringValue ("80Mbps"));
pointToPoint1.SetChannelAttribute ("Delay", TimeValue (MicroSeconds (5)));

//interfaccia I3: n5--n6
NodeContainer p2pNodes3;
p2pNodes3.Add(n5); //abbiamo aggiunto n5 all'interfaccia I3
p2pNodes3.Add(n6); //abbiamo aggiunto n6 all'interfaccia I3

PointToPointHelper pointToPoint3; //si settano gli attributi sulla interfaccia 3
pointToPoint3.SetDeviceAttribute("DataRate", StringValue ("80Mbps"));
pointToPoint3.SetChannelAttribute ("Delay", TimeValue (MicroSeconds (5)));

/*
Abbiamo ora inizializzato la rete con i pezzi 'fisici' ed i vari 
attributi delle interfacce. Abbiamo inoltre i puntatori ad ogni nodo. Rimane da:
-Inserire le schede di rete (NetDevices)
-Installare le stack protocollari
-Montare le applicazioni 
*/

//Installiamo ora, per ogni interfaccia e csma nella rete, le schede di rete

NS_LOG_INFO("Installando le schede di rete sui nodi...");
NetDeviceContainer p2pDevices0;
p2pDevices0 = pointToPoint0.Install (p2pNodes0);
NetDeviceContainer p2pDevices1;
p2pDevices1 = pointToPoint1.Install (p2pNodes1);
NetDeviceContainer p2pDevices2;
p2pDevices2 = pointToPoint2.Install (p2pNodes2);
NetDeviceContainer p2pDevices3;
p2pDevices3 = pointToPoint3.Install (p2pNodes3);
NetDeviceContainer csmaDevices1;
csmaDevices1 = csmaHelper1.Install (csmaNodes1);
NetDeviceContainer csmaDevices2;
csmaDevices2 = csmaHelper2.Install (csmaNodes2);

//Abbiamo installato le schede di rete su tutti i device
//Installiamo ora la stack protocollare su tutta la rete, stando attenti ad evitare che per un determinato nodo (condiviso tra due interfacce ad esempio) venga installata due volte.

NS_LOG_INFO("Installando la stack protocollare sui nodi della rete...");
InternetStackHelper stack;
stack.Install(csmaNodes1);
stack.Install(csmaNodes2);
stack.Install(n3);
stack.Install(p2pNodes2);

//Assegnamo gli indirizzi IP (IPV4) ai nodi della rete

NS_LOG_INFO("Definendo gli indirizzi IP dei nodi della rete...");
Ipv4AddressHelper address0;
Ipv4AddressHelper address1;
Ipv4AddressHelper address2;
Ipv4AddressHelper address3;
address0.SetBase("10.0.1.0","255.255.255.252");
address1.SetBase("10.0.2.0","255.255.255.252");
address2.SetBase("10.0.3.0","255.255.255.252");
address3.SetBase("10.0.4.0","255.255.255.252");
Ipv4InterfaceContainer p2pInterfaces0;
Ipv4InterfaceContainer p2pInterfaces1;
Ipv4InterfaceContainer p2pInterfaces2;
Ipv4InterfaceContainer p2pInterfaces3;
p2pInterfaces0 = address0.Assign (p2pDevices0);
p2pInterfaces1 = address1.Assign (p2pDevices1);
p2pInterfaces2 = address2.Assign (p2pDevices2);
p2pInterfaces3 = address3.Assign (p2pDevices3);


Ipv4AddressHelper addresslan1;
Ipv4AddressHelper addresslan2;
addresslan1.SetBase("192.138.1.0", "255.255.255.0");
addresslan2.SetBase("192.138.2.0", "255.255.255.0"); 
Ipv4InterfaceContainer csmaInterfaces1;
Ipv4InterfaceContainer csmaInterfaces2;
csmaInterfaces1 = addresslan1.Assign (csmaDevices1);
csmaInterfaces2 = addresslan2.Assign (csmaDevices2);

//la definizione della topologia è dunque terminata, ci rimane da generare il traffico sulla rete in funzione della configurazione selezionata

NS_LOG_INFO("PRONTO A ESEGUIRE LA CONFIGURAZIONE");

if(configuration == 0){

//CONFIGURAZIONE 0 -- Dobbiamo configurare un OnOff TCP su n4 ed un TCP sink su n2 

LogComponentEnable("PacketSink", LOG_LEVEL_INFO);
LogComponentEnable("OnOffApplication", LOG_LEVEL_INFO);

NS_LOG_INFO("Avvio configurazione 0");
uint16_t port = 2400;
ApplicationContainer SinkApp; //Il server sink su n2
Address SinkAddress(InetSocketAddress(csmaInterfaces1.GetAddress(2),port));//definiamo l'address a cui il client dovrà connettersi, con la relativa porta e l'indirizzo IP del nodo n2
Address MaskAddress(InetSocketAddress(Ipv4Address::GetAny(),port)); //al server ci basta specificare la porta su cui stare in ascolto, dato che sa già il suo stesso indirizzo 
PacketSinkHelper packetSinkHelper ("ns3::TcpSocketFactory", MaskAddress); 
SinkApp = packetSinkHelper.Install(csmaNodes1.Get(2)); //installiamo il server Sink sul nodo n2
SinkApp.Start(Seconds(0)); //avviamo il sink a 0 sec
SinkApp.Stop(Seconds(20.0)); //il sink termina a 20 secondi dall'inizio della simulazione

ApplicationContainer onOffApp; //il client onOff su n4
OnOffHelper onOffHelper ("ns3::TcpSocketFactory", SinkAddress); //comunichiamo al client l'indirizzo IP a cui questo andrà a connettersi, secondo il protocollo TCP
onOffHelper.SetAttribute ("OnTime", StringValue ("ns3::ConstantRandomVariable[Constant=1]")); 
onOffHelper.SetAttribute ("OffTime", StringValue ("ns3::ConstantRandomVariable[Constant=0]"));
onOffHelper.SetAttribute("PacketSize",UintegerValue(1500) ); //settiamo la dimensione dei pacchetti a 1500 byte come da specifica
onOffApp.Add(onOffHelper.Install(p2pNodes2.Get(0))); //installiamo il client OnOff sul nodo n4

onOffApp.Start(Seconds(3.0)); //avviamo il client OnOff al secondo 3
onOffApp.Stop(Seconds(15.0)); //il client termina di trasmettere a 15 secondi dall'inizio della simulazione

Ipv4GlobalRoutingHelper::PopulateRoutingTables (); //facciamo sì che la rete sia dinamica e i pacchetti vengano correttamente indirizzati al destinatario


pointToPoint0.EnablePcap("task1-0-n3.pcap", p2pDevices0.Get(0),false,1); //genero pcap n3
pointToPoint2.EnablePcap("task1-0-n5.pcap", p2pDevices2.Get(1),false,1); //genero pcap n5
pointToPoint1.EnablePcap("task1-0-n6.pcap", p2pDevices1.Get(0),false,1); //genero pcap n6

csmaHelper1.EnableAscii("task1-0-n2.tr", csmaDevices1.Get(2),true); //genero ascii-tracing per n2 (sink)
pointToPoint2.EnableAscii("task1-0-n4.tr", p2pDevices2.Get(0),true); //genero ascii-tracing per n4 (client onOff)
//segue la generazione di tutti i pcap con tutte le interfacce utilizzati nella parte di analisi
/*
pointToPoint0.EnablePcap("task1-0-i0-n1.pcap",p2pDevices0.Get(1),0,1);
pointToPoint0.EnablePcap("task1-0-i0-n3.pcap",p2pDevices0.Get(0),0,1);
pointToPoint1.EnablePcap("task1-0-i1-n3.pcap",p2pDevices1.Get(1),0,1);
pointToPoint1.EnablePcap("task1-0-i1-n6.pcap",p2pDevices1.Get(0),0,1);
pointToPoint2.EnablePcap("task1-0-i2-n4.pcap",p2pDevices2.Get(0),0,1);
pointToPoint2.EnablePcap("task1-0-i2-n5.pcap",p2pDevices2.Get(1),0,1);
pointToPoint3.EnablePcap("task1-0-i3-n5.pcap",p2pDevices3.Get(0),0,1); 
pointToPoint3.EnablePcap("task1-0-i3-n6.pcap",p2pDevices3.Get(1),0,1);
csmaHelper2.EnablePcap("task1-0-csma2-n6.pcap",csmaDevices2.Get(0),0,1);
csmaHelper2.EnablePcap("task1-0-csma2-n7.pcap",csmaDevices2.Get(1),0,1);
csmaHelper2.EnablePcap("task1-0-csma2-n8.pcap",csmaDevices2.Get(2),0,1);
csmaHelper1.EnablePcap("task1-0-csma1-n0.pcap",csmaDevices1.Get(0),0,1);
csmaHelper1.EnablePcap("task1-0-csma1-n1.pcap",csmaDevices1.Get(1),0,1);
csmaHelper1.EnablePcap("task1-0-csma1-n2.pcap",csmaDevices1.Get(2),0,1);*/

//FINE CONFIGURAZIONE 0

}

else if(configuration ==1){

//CONFIGURAZIONE 1 -- 2 sink, ovvero n2 ed n0, e 2 client (n4 che invia ad n0, n8 che invia ad n2) 

LogComponentEnable("PacketSink", LOG_LEVEL_INFO);
LogComponentEnable("OnOffApplication", LOG_LEVEL_INFO);

NS_LOG_INFO("Avvio configurazione 1");
uint16_t port2 = 2400;
ApplicationContainer SinkApp2; //sink su n2
Address SinkAddress2(InetSocketAddress(csmaInterfaces1.GetAddress(2), port2));
Address MaskAddress2(InetSocketAddress(Ipv4Address::GetAny(),port2));
PacketSinkHelper packetSinkHelper2("ns3::TcpSocketFactory", MaskAddress2); 
SinkApp2 = packetSinkHelper2.Install(n2);
SinkApp2.Start(Seconds(0)); //avviamo il sink relativo al nodo 2 ad un secondo dall'inizio della simulazione
SinkApp2.Stop(Seconds(20.0)); //terminiamo il sink a 20 secondi dall'inizio


uint16_t port0 = 7777;
ApplicationContainer SinkApp0; //sink su n0
Address SinkAddress0(InetSocketAddress(csmaInterfaces1.GetAddress(0), port0));
Address MaskAddress0(InetSocketAddress(Ipv4Address::GetAny(),port0));
PacketSinkHelper packetSinkHelper0("ns3::TcpSocketFactory", MaskAddress0); 
SinkApp0 = packetSinkHelper0.Install(n0);
SinkApp0.Start(Seconds(0));
SinkApp0.Stop(Seconds(20.0));

ApplicationContainer onOffApp4; //client onOff su n4

OnOffHelper onOffHelper4 ("ns3::TcpSocketFactory", SinkAddress0); //n4 -> n0
onOffHelper4.SetAttribute ("OnTime", StringValue ("ns3::ConstantRandomVariable[Constant=1]"));
onOffHelper4.SetAttribute ("OffTime", StringValue ("ns3::ConstantRandomVariable[Constant=0]"));

onOffHelper4.SetAttribute("PacketSize",UintegerValue(2500) );
onOffApp4.Add(onOffHelper4.Install(n4));

onOffApp4.Start(Seconds(5.0));
onOffApp4.Stop(Seconds(15.0));

ApplicationContainer onOffApp8; //client onOff su n8

OnOffHelper onOffHelper8 ("ns3::TcpSocketFactory", SinkAddress2); //n8 -> n2
onOffHelper8.SetAttribute ("OnTime", StringValue ("ns3::ConstantRandomVariable[Constant=1]"));
onOffHelper8.SetAttribute ("OffTime", StringValue ("ns3::ConstantRandomVariable[Constant=0]"));

onOffHelper8.SetAttribute("PacketSize",UintegerValue(4500));
onOffApp8.Add(onOffHelper8.Install(n8));

onOffApp8.Start(Seconds(2.0));
onOffApp8.Stop(Seconds(9.0));

Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

pointToPoint0.EnablePcap("task1-1-n3.pcap", p2pDevices0.Get(0),0,1); //n3
pointToPoint2.EnablePcap("task1-1-n5.pcap", p2pDevices2.Get(1),0,1); //n5
pointToPoint1.EnablePcap("task1-1-n6.pcap", p2pDevices1.Get(0),0,1); //n6

csmaHelper1.EnableAscii("task1-1-n2.tr", csmaDevices1.Get(2),true); //n2
pointToPoint2.EnableAscii("task1-1-n4.tr", p2pDevices2.Get(0),true); //n4
csmaHelper2.EnableAscii("task1-1-n8.tr", csmaDevices2.Get(2),true); //n8
csmaHelper1.EnableAscii("task1-1-n0.tr",csmaDevices1.Get(0),true); //n0
//segue la generazione di tutti i pcap utilizzati nella parte di analisi
/*
pointToPoint0.EnablePcap("task1-1-i0-n1.pcap",p2pDevices0.Get(1),0,1);
pointToPoint0.EnablePcap("task1-1-i0-n3.pcap",p2pDevices0.Get(0),0,1);
pointToPoint1.EnablePcap("task1-1-i1-n3.pcap",p2pDevices1.Get(1),0,1);
pointToPoint1.EnablePcap("task1-1-i1-n6.pcap",p2pDevices1.Get(0),0,1);
pointToPoint2.EnablePcap("task1-1-i2-n4.pcap",p2pDevices2.Get(0),0,1);
pointToPoint2.EnablePcap("task1-1-i2-n5.pcap",p2pDevices2.Get(1),0,1);
pointToPoint3.EnablePcap("task1-1-i3-n5.pcap",p2pDevices3.Get(0),0,1); 
pointToPoint3.EnablePcap("task1-1-i3-n6.pcap",p2pDevices3.Get(1),0,1);
csmaHelper2.EnablePcap("task1-1-csma2-n6.pcap",csmaDevices2.Get(0),0,1);
csmaHelper2.EnablePcap("task1-1-csma2-n7.pcap",csmaDevices2.Get(1),0,1);
csmaHelper2.EnablePcap("task1-1-csma2-n8.pcap",csmaDevices2.Get(2),0,1);
csmaHelper1.EnablePcap("task1-1-csma1-n0.pcap",csmaDevices1.Get(0),0,1);
csmaHelper1.EnablePcap("task1-1-csma1-n1.pcap",csmaDevices1.Get(1),0,1);
csmaHelper1.EnablePcap("task1-1-csma1-n2.pcap",csmaDevices1.Get(2),0,1);*/

//FINE CONFIGURAZIONE 1

}
else if(configuration ==2){

//CONFIGURAZIONE 2 -- |TCP onOff client n4 -> TCP sink n2| |UDP onOff client n7 -> UDP sink n0| |UDP echo client n8 -> UDP echo server n2|

LogComponentEnable("PacketSink", LOG_LEVEL_INFO);
LogComponentEnable("OnOffApplication", LOG_LEVEL_INFO);
LogComponentEnable ("UdpEchoClientApplication", LOG_LEVEL_INFO);
LogComponentEnable ("UdpEchoServerApplication", LOG_LEVEL_INFO);

NS_LOG_INFO("Avvio configurazione 2");
uint16_t port2 = 2600;
ApplicationContainer SinkApp2;
Address SinkAddress2(InetSocketAddress(csmaInterfaces1.GetAddress(2), port2));
Address MaskAddress2(InetSocketAddress(Ipv4Address::GetAny(),port2));
PacketSinkHelper packetSinkHelper2("ns3::TcpSocketFactory", MaskAddress2); 
SinkApp2 = packetSinkHelper2.Install(n2);
SinkApp2.Start(Seconds(0)); //avviamo il sink relativo al nodo 2 ad un secondo dall'inizio della simulazione
SinkApp2.Stop(Seconds(20.0));

ApplicationContainer onOffApp4;
OnOffHelper onOffHelper4 ("ns3::TcpSocketFactory", SinkAddress2); // n4->n2 (Comunicazione tra i due TCP onOff/sink)
onOffHelper4.SetAttribute ("OnTime", StringValue ("ns3::ConstantRandomVariable[Constant=1]"));
onOffHelper4.SetAttribute ("OffTime", StringValue ("ns3::ConstantRandomVariable[Constant=0]"));
onOffHelper4.SetAttribute("PacketSize",UintegerValue(3000) );
onOffApp4.Add(onOffHelper4.Install(n4));
onOffApp4.Start(Seconds(3.0));
onOffApp4.Stop(Seconds(9.0));

uint16_t port0 = 2500;
ApplicationContainer SinkApp0;
Address SinkAddress0(InetSocketAddress(csmaInterfaces1.GetAddress(0), port0));
Address MaskAddress0(InetSocketAddress(Ipv4Address::GetAny(),port0));
PacketSinkHelper packetSinkHelper0("ns3::UdpSocketFactory", MaskAddress0); 
SinkApp0 = packetSinkHelper0.Install(n0);
SinkApp0.Start(Seconds(0));
SinkApp0.Stop(Seconds(20.0));

ApplicationContainer onOffApp7;
OnOffHelper onOffHelper7 ("ns3::UdpSocketFactory", SinkAddress0); // n7->n0 (Comunicazione tra i due UDP onOff/sink)
onOffHelper7.SetAttribute ("OnTime", StringValue ("ns3::ConstantRandomVariable[Constant=1]"));
onOffHelper7.SetAttribute ("OffTime", StringValue ("ns3::ConstantRandomVariable[Constant=0]"));
onOffHelper7.SetAttribute("PacketSize",UintegerValue(3000) );
onOffApp7.Add(onOffHelper7.Install(n7));
onOffApp7.Start(Seconds(5.0));
onOffApp7.Stop(Seconds(15.0));

//Parte echo:

//Server echo (UDP)
uint16_t portecho2 = 63;  
UdpEchoServerHelper serverecho2 (portecho2);
ApplicationContainer appechoserver = serverecho2.Install (n2);
appechoserver.Start (Seconds (0));
appechoserver.Stop (Seconds (20.0));

//Client echo (UDP)
uint32_t packetSize = 2560;
uint32_t maxPacketCount = 5; //invio 5 pacchetti
Time interPacketInterval = Seconds (2.0); //WARNING: Da cambiare in base alla risposta al nostro quesito
UdpEchoClientHelper clientecho8 (csmaInterfaces1.GetAddress(2), portecho2 ); // n8->n2 (Comunicazione tra i due UDP echo client/server)
clientecho8.SetAttribute ("MaxPackets", UintegerValue (maxPacketCount));
clientecho8.SetAttribute ("Interval", TimeValue (interPacketInterval));
clientecho8.SetAttribute ("PacketSize", UintegerValue (packetSize));
ApplicationContainer appechoclient = clientecho8.Install(n8);
uint8_t data[]={7,7,5,3,1,9,3};
clientecho8.SetFill(appechoclient.Get(0),data, sizeof(data),packetSize); //il contenuto dei messaggi è la somma delle nostre matricole [1941484, 1933643, 1938214, 1939852]
appechoclient.Start (Seconds (3.0)); 

Ipv4GlobalRoutingHelper::PopulateRoutingTables();

pointToPoint0.EnablePcap("task1-2-n3.pcap", p2pDevices0.Get(0),0,1); //genero pcap n3
pointToPoint2.EnablePcap("task1-2-n5.pcap", p2pDevices2.Get(1),0,1); //genero pcap n5
pointToPoint1.EnablePcap("task1-2-n6.pcap", p2pDevices1.Get(0),0,1); //genero pcap n6

csmaHelper1.EnableAscii("task1-2-n2.tr", csmaDevices1.Get(2),true); //n2
csmaHelper1.EnableAscii("task1-2-n0.tr",csmaDevices1.Get(0),true); //n0
pointToPoint2.EnableAscii("task1-2-n4.tr", p2pDevices2.Get(0),true); //n4
csmaHelper2.EnableAscii("task1-2-n8.tr", csmaDevices2.Get(2),true); //n8
csmaHelper2.EnableAscii("task1-2-n7.tr", csmaDevices2.Get(1),true); //n7
//segue la generazione di tutti i pcap generati nella parte di analisi

/*
pointToPoint0.EnablePcap("task1-2-i0-n1.pcap",p2pDevices0.Get(1),0,1);
pointToPoint0.EnablePcap("task1-2-i0-n3.pcap",p2pDevices0.Get(0),0,1);
pointToPoint1.EnablePcap("task1-2-i1-n3.pcap",p2pDevices1.Get(1),0,1);
pointToPoint1.EnablePcap("task1-2-i1-n6.pcap",p2pDevices1.Get(0),0,1);
pointToPoint2.EnablePcap("task1-2-i2-n4.pcap",p2pDevices2.Get(0),0,1);
pointToPoint2.EnablePcap("task1-2-i2-n5.pcap",p2pDevices2.Get(1),0,1);
pointToPoint3.EnablePcap("task1-2-i3-n5.pcap",p2pDevices3.Get(0),0,1); 
pointToPoint3.EnablePcap("task1-2-i3-n6.pcap",p2pDevices3.Get(1),0,1);
csmaHelper2.EnablePcap("task1-2-csma2-n6.pcap",csmaDevices2.Get(0),0,1);
csmaHelper2.EnablePcap("task1-2-csma2-n7.pcap",csmaDevices2.Get(1),0,1);
csmaHelper2.EnablePcap("task1-2-csma2-n8.pcap",csmaDevices2.Get(2),0,1);
csmaHelper1.EnablePcap("task1-2-csma1-n0.pcap",csmaDevices1.Get(0),0,1);
csmaHelper1.EnablePcap("task1-2-csma1-n1.pcap",csmaDevices1.Get(1),0,1);
csmaHelper1.EnablePcap("task1-2-csma1-n2.pcap",csmaDevices1.Get(2),0,1);*/



//FINE CONFIGURAZIONE 2

}
else{ //Se il parametro passato non è coerente con l'homework (0, 1, 2 sono gli unici valori accettabili) allora esci e segnala un errore

exit(EXIT_FAILURE);

}

Simulator::Stop(Seconds(20)); //Setta il tempo della simulazione 
Simulator::Run(); //Parte la simulazione
Simulator::Destroy(); //Si chiude la simulazione
NS_LOG_INFO("Termino configurazione");
return 0;




}
