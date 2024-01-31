
%esercizio_1a

%fatti, casi base
fibonacci(0, 0). 
fibonacci(1, 1).

%regole
fibonacci(N, F) :- 
    N > 1, %necessario altrimenti la ricorsione si ferma mai
    N1 is N-1,
    N2 is N-2,
    fibonacci(N1, F1), %nel momento in cui una delle due chiamate
    fibonacci(N2, F2), %ha in input N = 1 o 0 valgono i casi base (nei fatti)
    F is F1 + F2.      


%esercizio_1b

%fatti
% 1 non è numero primo, ma 2 e 3 sì e sono i nostri casi base
prime(2).  
prime(3).  

%regole
prime(N) :-
  N > 3,   
  N mod 2 =\= 0, % controllo se è divisibile per 2, se lo è
  		 % inutile dopo controllare i numeri pari
  \+ controllo_fattori(N, 3). 
  

% True se F è un fattore di N
controllo_fattori(N, F) :-
  N mod F =:= 0.


controllo_fattori(N, F) :-
  F * F < N, % un fattore di N non può essere maggiore di sqrt(N)
  F_curr is F + 2, % non essendo divisibile per 2 salto di 2 in 2
  controllo_fattori(N, F_curr).




%esercizio_2a

% caso base: se i due vettori sono vuoti il prodotto è 0
cdot([], [], 0).


cdot([A | Resto_A], [B | Resto_B], Result) :-
    cdot(Resto_A, Resto_B, Partial), % Partial conterrà la parte di prodotto
    				     % scalare già calcolata
    Result is A * B + Partial. % aggiungo la parte relativa agli elementi A e B correnti
    
    
%esercizio_2b



steep(L) :-
    steep(L, 0, 0).

% caso base 
steep([], _, _).

    
steep([X|Xs], CurrentSum, CurrentMax) :-
     X >= CurrentSum, % l'elemento corrente deve essere maggiore della somma 
     	       % calcolata fino ad ora
     Sum_New is CurrentSum + X,
     Max is max(X, CurrentMax), % nel caso in cui X sia maggiore
     				% del massimo corrente lo diventa lui
     
     steep(Xs, Sum_New, Max).


%esercizio_2c


seg(S, L) :-
    append(_, Ultimo, L), % separo L e mi prendo la seconda parte
    in_testa(S, Ultimo). % S è alla testa della seconda parte di L?
   
   
%caso base
in_testa([], _).


% per avere S segmento di Ultimo
% X deve essere il primo elemento di entrambe le liste 
in_testa([X | XResto], [X | B]) :-
    in_testa(XResto, B). % ricorsivamente andiamo a controllare anche il resto



%esercizio_2d

%caso base: aggiungo l'elemento alla lista vuota
isort(E, [], [E]).


% Se l'elemento da aggiungere è minore o uguale al primo elemento della lista (la ipotizziamo ordinata), bisogna inserire E come nuovo primo elemento 
isort(E, [L_elem | L_Resto], [E, L_elem | L_Resto]) :-
    E =< L_elem.
    
% Altro caso: E è maggiore. vado avanti nella lista 
% per inserire E nel resto della lista ordinata L
isort(E, [L_elem | L_Resto], [L_elem | L_ordinato]) :-
    E > L_elem,
    isort(E, L_Resto, L_ordinato).


%esercizio_3

%fatti, archi del grafo
arc(a,b).
arc(a,c).
arc(b,d).
arc(b,e).
arc(c,b).
arc(e,c).


% NOTA: se si prova a inserire un nodo che non esiste 
% nel grafo, es. dfv(g, N), verrà stampato N = g
% dopodiché il programma terminerà

% wrapper per dfv/3
dfv(R, N) :-
    dfv(R, N, []). % parte la visita

% L è la lista dei nodi esplorati
dfv(R, N, L) :-
    \+ member(R, L),  % il nodo non deve essere già stato esplorato
    write('N = '), write(R), write(';\n'),  
    arc(R, Vicino), % la visita continua dal nodo successivo
    dfv(Vicino, N, [R|L]).  % [R|L] perché adesso R è un nodo esplorato
