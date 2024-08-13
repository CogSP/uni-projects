/* -*- P4_16 -*- */
#include <core.p4>
#include <v1model.p4>

// ethertype per ipv4 e ipv6 rispettivamente
const bit<16> TYPE_IPV4 = 0x0800;
const bit<16> TYPE_IPV6 = 0x86DD;

// https://en.wikipedia.org/wiki/List_of_IP_protocol_numbers
// 253-254 sono protocol number utilizzabili per sperimentazione e testing
const bit<8> TYPE_CONSENSUS = 253;

// protocol number di ip per udp e tcp rispettivamente
const bit<8> TYPE_UDP = 17;
const bit<8> TYPE_TCP = 6;

/*************************************************************************
*********************** H E A D E R S  ***********************************
*************************************************************************/

typedef bit<9>  egressSpec_t;

header ethernet_t {
    bit<48> dstAddr;
    bit<48> srcAddr;
    bit<16> etherType;
}

header ipv4_t {
    bit<4> version;
    bit<4> ihl;
    bit<8> diffserv;
    bit<16> totalLen;
    bit<16> identification;
    bit<3> flags;
    bit<13> fragOffset;
    bit<8> ttl;
    bit<8> protocol;
    bit<16> hdrChecksum;
    bit<32> srcAddr;
    bit<32> dstAddr;
}

header ipv6_t {
    bit<4> version;
    bit<8> traffClass;
    bit<20> flowLabel;
    bit<16> payloadLen;
    bit<8> nextHeader;
    bit<8> hoplim;
    bit<128> srcAddr;
    bit<128> dstAddr;
}

header consensus_t {
    bit<8> accepted_number;
    bit<8> denied_number;
    bit<8> proto_id;
}

header tcp_t {
    bit<16> srcPort;
    bit<16> dstPort;
    bit<32> sequenceNum;
    bit<32> ackNum;
    bit<4> dataOffset;
    bit<3> reserved;
    bit<9> flags;
    bit<16> winSize;
    bit<16> checksum;
    bit<16> urgentPointer;
}

header udp_t {
    bit<16> srcPort;
    bit<16> dstPort;
    bit<16> len;
    bit<16> checksum;
}

struct metadata {
    bool consensus;
}

struct headers {
    ethernet_t ethernet;
    ipv4_t ipv4;
    ipv6_t ipv6;
    consensus_t consensus;
    udp_t udp;
    tcp_t tcp;
}

/*************************************************************************/
/**************************  P A R S E R  ********************************/
/*************************************************************************/

parser MyParser(packet_in packet,
                out headers hdr,
                inout metadata meta,
                inout standard_metadata_t standard_metadata) {

    state start {
        transition parse_ethernet;
    }

    state parse_ethernet {
        packet.extract(hdr.ethernet);
        transition select(hdr.ethernet.etherType){
            TYPE_IPV4: parse_ipv4;
            TYPE_IPV6: parse_ipv6;
            default: accept;
        }
    }

    state parse_ipv4{
        packet.extract(hdr.ipv4);
        transition select(hdr.ipv4.protocol){
            TYPE_CONSENSUS: parse_consensus;
            TYPE_TCP: parse_tcp;
            TYPE_UDP: parse_udp;
            default: accept;
        }
    }

    state parse_ipv6{
        packet.extract(hdr.ipv6);
        transition select(hdr.ipv6.nextHeader){
            TYPE_CONSENSUS: parse_consensus;
            TYPE_TCP: parse_tcp;
            TYPE_UDP: parse_udp;
            default: accept;
        }
    }

    state parse_consensus {
        packet.extract(hdr.consensus);
        transition select(hdr.consensus.proto_id){
            TYPE_TCP : parse_tcp;
            TYPE_UDP : parse_udp;
            default: accept;
        }
    }

    state parse_tcp {
        packet.extract(hdr.tcp);
        transition accept;
    }

    state parse_udp {
        packet.extract(hdr.udp);
        transition accept;
    }
}

/*************************************************************************/
/**************  C H E C K S U M   V E R I F I C A T I O N  **************/
/*************************************************************************/

control MyVerifyChecksum(inout headers hdr, inout metadata meta) {
    apply {  }
}


/*************************************************************************/
/*****************  I N G R E S S   P R O C E S S I N G  *****************/
/*************************************************************************/

control MyIngress(inout headers hdr, inout metadata meta, inout standard_metadata_t standard_metadata) {


    // se lo switch accetta il pacchetto, incrementa di uno il counter
    action accept(){
        hdr.consensus.accepted_number = hdr.consensus.accepted_number + 1;
    }

    // se lo switch rifiuta O SI ASTIENE dal voto, decrementa di uno il 
    // questo approccio conservativo porta la rete a rifiutare i pacchetti
    // per cui il grado di indecisione sulla malevolenza di questi è alto
    action deny(){
        //log_msg("in deny accepted_number = {}", {hdr.consensus.accepted_number});
        hdr.consensus.denied_number = hdr.consensus.denied_number + 1;
        //log_msg("in deny accepted_number = {}", {hdr.consensus.accepted_number});
    }

    action drop() {
        mark_to_drop(standard_metadata);
    }

    action ipv4_forward(bit<9> port) {
        hdr.ipv4.ttl = hdr.ipv4.ttl - 1;
        standard_metadata.egress_spec = port;
    }

    // per il pacchetto ipv4 che lascia l'host sorgente, arrivando allo switch
    // di ingresso
    action from_src_ipv4(){
        hdr.consensus.setValid();

        hdr.consensus.accepted_number = 0;
        hdr.consensus.denied_number = 0;
        // ipv4.protocol contiene il L4 header, che, considerando consensus come un
        // protocollo a metà tra IP e lo strato di trasporto
        // deve diventare il next header di consensus
        hdr.consensus.proto_id = hdr.ipv4.protocol; 
        hdr.ipv4.protocol = TYPE_CONSENSUS;
    }


    // per il pacchetto ipv4 che arriva allo switch di uscita 
    action ipv4_to_dest(bit<9> port){
        // CONTROLLO SUI VOTI: se il numero degli accettati è maggiore del numero di rifiuti
        // la rete ha votato per tenere il pacchetto, altrimenti il pacchetto viene droppato
       
        meta.consensus = hdr.consensus.accepted_number > hdr.consensus.denied_number;

        // l'header consensus viene rimosso
        hdr.ipv4.protocol = hdr.consensus.proto_id;
        hdr.consensus.setInvalid();
        hdr.ipv4.ttl = hdr.ipv4.ttl - 1;
        standard_metadata.egress_spec = port;
    }
    
    
    action ipv6_forward(bit<9> port){
        hdr.ipv6.hoplim = hdr.ipv6.hoplim - 1;
        standard_metadata.egress_spec = port;
    }

    // per il pacchetto ipv6 che lascia l'host sorgente, arrivando allo switch
    // di ingresso
    action from_src_ipv6(){
        hdr.consensus.setValid();
        hdr.consensus.proto_id = hdr.ipv6.nextHeader;
        hdr.ipv6.nextHeader = TYPE_CONSENSUS;
    }

    // per il pacchetto ipv6 che arriva allo switch di uscita 
    action ipv6_to_dest(bit<9> port){
        bit<8> accepted_number = hdr.consensus.accepted_number;
        meta.consensus = accepted_number > hdr.consensus.denied_number;
        hdr.ipv6.nextHeader = hdr.consensus.proto_id;
        hdr.consensus.setInvalid();
        hdr.ipv6.hoplim = hdr.ipv6.hoplim - 1;
        standard_metadata.egress_spec = port;
    }


    table ipv4_forwarding {
        key = {
            hdr.ipv4.dstAddr: lpm;
        }

        actions = {
            ipv4_forward;
            ipv4_to_dest;
            drop;
        }

        size = 1024;
        default_action = drop();
    }

    table ipv6_forwarding {
        key = {
            hdr.ipv6.dstAddr: lpm;
        }

        actions = {
            ipv6_forward;
            ipv6_to_dest;
            drop;
        }

        size = 1024;
        default_action = drop();
    }



// le preprocessor directives ci permettono di far ispezionare allo switch di livello i solo la 
// table di voting di livello i
#ifdef LAYER2_VOTING
    
    
    // il match viene fatto sull'indirizzo MAC di src e dst, 
    // ma si distingue anche per protocollo di L3 utilizzato (ipv4 vs ipv6)
    table l2_voting {
        key = {
            hdr.ethernet.srcAddr : exact;
            hdr.ethernet.dstAddr : exact;
            hdr.ethernet.etherType : exact;
        }

        actions = {
            accept;
            deny;
        }

        size = 1024;
        default_action = deny();
    }
#endif

#ifdef LAYER3_VOTING
  
    table l3_ipv4_voting {
        key = {
            hdr.ipv4.srcAddr: exact;
        }

        actions = {
            accept;
            deny;
        }

        size = 1024;
        default_action = deny();
    }

    table l3_ipv6_voting {
        key = {
            hdr.ipv6.srcAddr: exact;
        }

        actions = {
            accept;
            deny;
        }

        size = 1024;
        default_action = deny();
    }
#endif

#ifdef LAYER4_VOTING
    
    table l4_udp_voting {
        key = {
            hdr.udp.dstPort : exact;
        }

        actions = {
            accept;
            deny;
        }

        size = 1024;
        default_action = deny();
    }

    table l4_tcp_voting {
        key = {
            hdr.tcp.dstPort : exact;
        }

        actions = {
            accept;
            deny;
        }

        size = 1024;
        default_action = deny();
    }
#endif

    apply {
        
        // questa condizione è verificata nello switch di ingresso
        if(!hdr.consensus.isValid()){
            if(hdr.ipv4.isValid()) from_src_ipv4();
            else from_src_ipv6();
        }


// le preprocessor directives ci permettono di far ispezionare allo switch di livello i solo la 
// table di voting di livello i
#ifdef LAYER2_VOTING
        if(hdr.ethernet.isValid()) l2_voting.apply();
#endif

#ifdef LAYER3_VOTING
        if(hdr.ipv4.isValid()) l3_ipv4_voting.apply();
        else if(hdr.ipv6.isValid()) l3_ipv6_voting.apply();
#endif

#ifdef LAYER4_VOTING
        if(hdr.tcp.isValid()) l4_tcp_voting.apply();
        else if(hdr.udp.isValid()) l4_udp_voting.apply();
#endif

    
        // forwarding del pacchetto, dopo aver votato
        if(hdr.ipv4.isValid()){
            ipv4_forwarding.apply();
        } 
        else if(hdr.ipv6.isValid()) {
            ipv6_forwarding.apply();
        }
        
        if(!hdr.consensus.isValid()) {
            log_msg("meta.consensus = {}", {meta.consensus});
            if (meta.consensus == false) {
                mark_to_drop(standard_metadata);
            }
        }
    }
}

/*************************************************************************/
/****************  E G R E S S   P R O C E S S I N G   *******************/
/*************************************************************************/

control MyEgress(inout headers hdr,
                 inout metadata meta,
                 inout standard_metadata_t standard_metadata) {
    apply {

    }
}

/*************************************************************************/
/*************   C H E C K S U M    C O M P U T A T I O N   **************/
/*************************************************************************/

control MyComputeChecksum(inout headers  hdr, inout metadata meta) {
     apply {
        update_checksum(
            hdr.ipv4.isValid(),
            { hdr.ipv4.version,
              hdr.ipv4.ihl,
              hdr.ipv4.diffserv,
              hdr.ipv4.totalLen,
              hdr.ipv4.identification,
              hdr.ipv4.flags,
              hdr.ipv4.fragOffset,
              hdr.ipv4.ttl,
              hdr.ipv4.protocol,
              hdr.ipv4.srcAddr,
              hdr.ipv4.dstAddr },
            hdr.ipv4.hdrChecksum,
            HashAlgorithm.csum16);
    }
}

/*************************************************************************/
/***********************  D E P A R S E R  *******************************/
/*************************************************************************/

control MyDeparser(packet_out packet, in headers hdr) {
    apply {
        packet.emit(hdr.ethernet);
        packet.emit(hdr.ipv4);
        packet.emit(hdr.ipv6);
        packet.emit(hdr.consensus);
        packet.emit(hdr.tcp);
        packet.emit(hdr.udp);
    }
}

/*************************************************************************/
/**************************  S W I T C H  ********************************/
/*************************************************************************/


V1Switch(
    MyParser(),
    MyVerifyChecksum(),
    MyIngress(),
    MyEgress(),
    MyComputeChecksum(),
    MyDeparser()
) main;
