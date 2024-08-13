THE IMAGE OF THE TOPOLOGY IS AVAILABLE IN /shared

SWITCH S1:

Accetta:
- traffico ipv6 tra h1 e h4 (in entrambe le direzioni)
- traffico ipv6 tra h2 e h4 (in entrambe le direzioni)
- traffico ipv4 e ipv6 che va da h1 a h2 (ma non viceversa)

Vota No:
- traffico ipv4 e ipv6 tra h1 e h3 (in entrambe le direzioni)
- traffico ipv4 tra h2 e h4 (in entrambe le direzioni)


SWITCH S2:

Accetta:
- traffico ipv6 da h3
- traffico ipv6 da h4

Vota No:
- traffico ipv4 da h3
- traffico ipv4 da h4


SWITCH S3:

Accetta:
- traffico ipv4 e ipv6 da h1
- traffico ipv4 e ipv6 da h2
- traffico ipv4 e ipv6 da h4

Vota No:
- niente

SWITCH S4:

Accetta:
- traffico dalla porta tcp 1122
- traffico dalla porta udp 4455

Vota No:
- niente

SWITCH S5:

Accetta:
- traffico dalla porta tcp 120

Vota No:
- niente

SWITCH S6:

Accetta:
- traffico ipv4 e ipv6 tra h4 e h1 (in entrambe le direzioni)
- traffico ipv4 e ipv6 tra h4 e h2 (in entrambe le direzioni)
- traffico ipv6 da h4 a h3 (solo in questa direzione)

Vota No:
- niente

Esempi:

- h4 esegue: ping4 192.168.0.1
	- s6: allow
	- s4: ininfluente (L4)
	- s2: abstain -> deny
	- s1: abstain -> deny\
	Il pacchetto viene quindi scartato 

- h4 esegue: ping6 2001::1
	- s6: allow
	- s4: ininfluente (L4)
	- s2: allow
	- s1: allow\
	Il pacchetto raggiunge quindi h1, ma h1 riesce a rispondere?
	- s1: allow
	- s2: abstain -> deny
	- s4: ininfluente (L4)
	- s6: allow\
	La risposta arriva quindi a h1

- h4 esegue: nc -p <not 1122> 192.168.0.1 <not 1122>
  h1 esegue: nc -lvp <not 1122>
	- s6: allow
	- s4: deny
	- s2: deny
	- s1: deny\
	Il pacchetto viene quindi scartato 


- h4 esegue: nc -p 1122 192.168.0.1 1122
  h1 esegue: nc -lvp 1122
	- s6: allow
	- s4: allow
	- s2: deny
	- s1: deny\
	Il pacchetto viene quindi scartato 


- h4 esegue: nc -p 1122 2001::1 1122
  h1 esegue: nc -6 -l -v -p 1122
	- s6: allow
	- s4: allow
	- s2: allow
	- s1: allow\
	Risposta:
	- s1: allow
	- s2: allow
	- s4: deny
	- s6: allow\
	La comunicazione è quindi possible


- h4 esegue: ping4 192.168.0.2
	- s6: allow
	- s5: ininfluente (L4)
	- s3: allow
	- s1: deny\
	Risposta:
	- s1: deny
	- s3: allow
	- s5: ininflunte (L4)
	- s6: allow\
	La comunicazione è quindi possible


- h1 esegue: ping4 192.168.0.2	
	- s1: allow\
	Risposta:
	- s1: deny\
	Il pacchetto viene quindi scartato


# Primo Progetto
Il primo progetto di laboratorio è disponibile alla repository: https://github.com/kev187038/KatharaLab_1


