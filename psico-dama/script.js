
var colore_bianchi  
var colore_neri

//questa richiesta xmlhttp serve per capire quale tema è stato scelto dall'utente
xmlhttp = new XMLHttpRequest();
        xmlhttp.onreadystatechange = () => {
            if(xmlhttp.readyState === 4) {
                if(xmlhttp.status === 200){

                    testo = xmlhttp.responseText.trim();
                    console.log("risposta: " + testo);
                    if(testo == "std"){
                        document.getElementById("foglioback").href = "background.css";
                        document.getElementById("fogliogame").href = "game.css";
                        colore_bianchi = "rgb(252, 249, 249)"
                        colore_neri = "black"
                    }
                    else if(testo == "dark"){
                        document.getElementById("foglioback").href = "background_dark.css";
                        document.getElementById("fogliogame").href = "game_dark.css";

                        colore_bianchi = "rgb(252, 249, 249)"
                        colore_neri = "darkgrey"
                    }
                    else if(testo == "trop"){
                        document.getElementById("foglioback").href = "background_tropical.css";
                        document.getElementById("fogliogame").href = "game_trop.css";

                        colore_bianchi = "rgb(188, 199, 32);"
                        colore_neri = "rgb(53, 179, 37);"
                    }
                    else{
                        console.log("c'è un problema");
                    }
                }
            }
           }
        xmlhttp.open("GET",`theme.php`, true);
        xmlhttp.send();


const board = [ //salviamo una rappresentazione della damiera in un array
    null, 0, null, 1, null, 2, null, 3,
    4, null, 5, null, 6, null, 7, null,
    null, 8, null, 9, null, 10, null, 11,
    null, null, null, null, null, null, null, null,
    null, null, null, null, null, null, null, null,
    12, null, 13, null, 14, null, 15, null,
    null, 16, null, 17, null, 18, null, 19,
    20, null, 21, null, 22, null, 23, null
]

let found = false; //variabile che serve a capire se ci sono pedine che potrebbero mangiare
                   //in questo modo obblighiamo l'utente a mangiare se può farlo

let findPiece = function(pieceId) { //La funzione restituisce, dato l'id html del pezzo, il corrispondente indice nella damiera di back-end
    let parsed = parseInt(pieceId);
    return board.indexOf(parsed);
}



const cells = document.querySelectorAll("td"); 
const pieces = document.querySelectorAll("p");
let whitesPieces = document.querySelectorAll(".white-piece"); 
for(let i = 0; i < whitesPieces.length;i++)
            whitesPieces[i].classList.add('turno'); //aggiungiamo la classe turno a tutte le pedine bianche in modo che si illuminino quando ci passo il mouse
let blacksPieces = document.querySelectorAll(".black-piece"); 
const whiteTurnText = document.getElementById("wtt");
const blackTurnText = document.getElementById("btt");
const divider = document.querySelector("#divider")


let turn = true;  /* true = white, false = black*/ 
let whiteScore = 12; //pezzi rimanenti al bianco
let blackScore = 12; //pezzi rimanenti al nero
let playerPieces;  //se è il turno del bianco playerpieces = whitepieces


let selectedPiece = { //classe che gestisce le informazioni della pedina selezionata
    //default (nessun pezzo selezionato)
    pieceId: -1,
    indexOfBoardPiece: -1,
    isKing: false,

    //possibili spazi in cui la pedina può muoversi
    seventhSpace: false,
    ninthSpace: false,
    fourteenthSpace: false,
    eighteenthSpace: false,
    // +7/+9 per i bianchi, -7/-9 per i neri (mosse standard in avanti)

    //mosse all'indietro (se una pedina è re)
    minusSeventhSpace: false,
    minusNinthSpace: false,
    minusFourteenthSpace: false,
    minusEighteenthSpace: false,
}


 

//assegniamo un event listener a tutti gli elementi 
function givePiecesEventListeners() {

    if (turn) {
        //console.log("white's turn");
        
        for (let i = 0; i < whitesPieces.length; i++) {
            whitesPieces[i].addEventListener("click", getPlayerPieces);
        }
    } else {
        
        //console.log("black's turn");
        for (let i = 0; i < blacksPieces.length; i++) {
            blacksPieces[i].addEventListener("click", getPlayerPieces);
        }
    }
}




//funzione che cambia i pezzi del giocatore attuale a seconda del turno

function getPlayerPieces() {
    //whites turn
    if (turn) {
        //console.log("a white piece has been clicked");
        playerPieces = whitesPieces;
    } else {
        //console.log("a black piece has been clicked");
        playerPieces = blacksPieces;
    }

    removeCellonclick();
    resetBorders();
}

//leva il click da tutte le celle, in modo da poterlo assegnare solo a quelle giuste in seguito
function removeCellonclick() {
    for (let i = 0; i < cells.length; i++) {
        cells[i].removeAttribute("onclick");
    }
}

//resetta il colore dei bordi, così da poterlo riassegnare in seguito alla pedina selezionata
function resetBorders() {
    for (let i = 0; i < playerPieces.length; i++) {
        if (screen.width >= 350 && screen.height >= 700) {

            if (turn){
                playerPieces[i].style.background = colore_bianchi;
            }

            else {
                playerPieces[i].style.background = colore_neri;
            }

            playerPieces[i].style.border = "0.1em solid #808080";
        }
        
        else {
            playerPieces[i].style.border = "0";
            if (turn) {
                playerPieces[i].style.background = colore_bianchi;
            }
            
            else {
                playerPieces[i].style.background = colore_neri
            }
        }
    } 
    resetSelectedPieceProperties();
    getSelectedPiece(); //cominciamo a operare sulla pedina selezionata
}

function resetSelectedPieceProperties() { //reset delle proprietà del pezzo selezionato
    selectedPiece.pieceId = -1;
    selectedPiece.indexOfBoardPiece = -1;
    selectedPiece.isKing = false;
    selectedPiece.seventhSpace = false;
    selectedPiece.ninthSpace = false;
    selectedPiece.fourteenthSpace = false;
    selectedPiece.eighteenthSpace = false;
    selectedPiece.minusSeventhSpace = false;
    selectedPiece.minusNinthSpace = false;
    selectedPiece.minusFourteenthSpace = false;
    selectedPiece.minusEighteenthSpace = false;
}



function getSelectedPiece() {

    selectedPiece.pieceId = parseInt(event.target.id); //prendiamo l'id del pezzo selezionato
    selectedPiece.indexOfBoardPiece = findPiece(selectedPiece.pieceId); //prendiamo l'indice del pezzo selezionato nella damiera di back-end
    
    isPieceKing(); //controlliamo se il pezzo selezionato è un re
}



function isPieceKing() {

    //console.log("sto per controllare se l'elemento è king. L'elemento ha id = " + selectedPiece.pieceId);
    console.log("il selected piece ha Id", selectedPiece.pieceId);
    console.log("l'elemento è", document.getElementById(selectedPiece.pieceId));

    if (document.getElementById(selectedPiece.pieceId).classList.contains("king")) { //controlliamo se il pezzo selezionato è un re
        selectedPiece.isKing = true;
    } else {
        selectedPiece.isKing = false;
    }

    getAvailableSpaces(); //gli spazi in cui può muoversi il pezzo variano a seconda che sia re o no
}

function getAvailableSpaces() { 

    //possiamo muoverci se la cella è vuota e se è una casella valida (quindi non bianca)
    //se il pezzo è un re, può muoversi anche all'indietro, controlleremno se è re in seguito ma per ora supponiamo che lo sia
    if (board[selectedPiece.indexOfBoardPiece + 7] === null  
        && cells[selectedPiece.indexOfBoardPiece + 7].classList.contains("white") !== true) { 
            selectedPiece.seventhSpace = true; 
    }

    if (board[selectedPiece.indexOfBoardPiece + 9] === null &&
        cells[selectedPiece.indexOfBoardPiece + 9].classList.contains("white") !== true) {
            selectedPiece.ninthSpace = true; 
    }

    if (board[selectedPiece.indexOfBoardPiece - 7] === null &&
        cells[selectedPiece.indexOfBoardPiece - 7].classList.contains("white") !== true) {
            selectedPiece.minusSeventhSpace = true; 
    }

    if (board[selectedPiece.indexOfBoardPiece - 9] === null &&
        cells[selectedPiece.indexOfBoardPiece - 9].classList.contains("white") !== true) {
            selectedPiece.minusNinthSpace = true; 
    }
    
    
    checkAvailableJumpSpaces();
}


function YouGottaEat(index) { //funzione che controlla se un certo pezzo sulla scacchiera (generico, non il selezionato) può mangiare, in quel caso deve farlo
    
    console.log("controlliamo se il pezzo", index, "è obbligato a mangiare")
    
    if (turn) {

        if (
        index + 9 < 64 && index + 18 < 64 &&
        board[index + 14] === null 
        && cells[index + 14].classList.contains("white") !== true
        && board[index + 7] !== null
        && board[index + 7] >= 12){
            console.log("c'è un nero in + 7 da mangiare per poi andare + 14")
            console.log("index = ", index.toString());
            found = true;
        }

        if (board[index + 18] === null 
        &&  index + 9 < 64 && index + 18 < 64
        && cells[index + 18].classList.contains("white") !== true
        && board[index + 9] !== null
        && board[index + 9] >= 12) {
            console.log("c'è un nero in + 9 da mangiare per poi andare + 18")
            console.log("index = ", index.toString());
            found = true;
        }
        
        
        if (
        index - 7 >= 0 && index - 14 >=0
        && board[index- 14] === null 
        && document.getElementById(board[index]).classList.contains("king")
        && cells[index - 14].classList.contains("white") !== true
        && board[index -7] !== null
        && board[index - 7] >= 12) {
            console.log("poiché questo bianco è un re, può mangiare al contrario un pezzo nero in -7 per poi andare in -14")
            console.log("index = ", index.toString());
            found = true;
        }

        if (board[index - 18] === null 
            && index - 9 >= 0 && index - 18 >=0
        && document.getElementById(board[index]).classList.contains("king")
        && cells[index - 18].classList.contains("white") !== true
        && board[index -9] !== null
        && board[index - 9] >= 12) {
            console.log("poiché questo bianco è un re, può mangiare al contrario un pezzo nero in -9 per poi andare in -18")
            console.log("index = ", index.toString());
            found = true;
        }
    } else {

        if (board[index + 14] === null
        &&  index + 7 < 64 && index + 14 < 64
        && document.getElementById(board[index]).classList.contains("king")
        && cells[index + 14].classList.contains("white") !== true
        && board[index + 7] < 12 && board[index + 7] !== null) {
            console.log("poiché questo nero è un re, può mangiare al contrario un pezzo nero in +7 per poi andare in +14")
            console.log("indexKing = ", index.toString());
            found = true;
        }

        if (board[index + 18] === null 
        &&  index + 9 < 64 && index + 18 < 64
        && document.getElementById(board[index]).classList.contains("king")
        && cells[index + 18].classList.contains("white") !== true
        && board[index + 9] < 12 && board[index + 9] !== null) {
            console.log("poiché questo nero è un re, può mangiare al contrario un pezzo nero in +9 per poi andare in +18")
            console.log("indexKing = ", index.toString());
            found = true;
        }
        if (board[index - 14] === null && cells[index - 14].classList.contains("white") !== true
        && index - 7 >= 0 && index - 14 >=0
        && board[index - 7] < 12 
        && board[index - 7] !== null) {
            console.log("c'è un bianco in  -7 da mangiare per poi andare -14")
            console.log("index = ", index.toString());
            found = true;
        }
        if (board[index - 18] === null && cells[index - 18].classList.contains("white") !== true
        && index - 9 >= 0 && index - 18 >=0
        && board[index - 9] < 12
        && board[index - 9] !== null) {
            console.log("c'è un bianco in -9 da mangiare per poi andare -18")
            console.log("index = ", index.toString());
            found = true;
        }
    }
}


function checkAvailableJumpSpaces() { //controlliamo se il pezzo può mangiare

    if (turn) {

        if (board[selectedPiece.indexOfBoardPiece + 14] === null 
        && cells[selectedPiece.indexOfBoardPiece + 14].classList.contains("white") !== true
        && board[selectedPiece.indexOfBoardPiece + 7] >= 12) {
            console.log("Non entrare qui!!!");
            selectedPiece.fourteenthSpace = true;
        } else {
            selectedPiece.fourteenthSpace = false;
        }


        if (board[selectedPiece.indexOfBoardPiece + 18] === null 
        && cells[selectedPiece.indexOfBoardPiece + 18].classList.contains("white") !== true
        && board[selectedPiece.indexOfBoardPiece + 9] >= 12) {
            selectedPiece.eighteenthSpace = true;
        } else {
            selectedPiece.eighteenthSpace = false;
        }


        if (board[selectedPiece.indexOfBoardPiece - 14] === null 
        && cells[selectedPiece.indexOfBoardPiece - 14].classList.contains("white") !== true
        && board[selectedPiece.indexOfBoardPiece - 7] >= 12) {
            selectedPiece.minusFourteenthSpace = true;
        } else {
            selectedPiece.minusFourteenthSpace = false;
        }


        if (board[selectedPiece.indexOfBoardPiece - 18] === null 
        && cells[selectedPiece.indexOfBoardPiece - 18].classList.contains("white") !== true
        && board[selectedPiece.indexOfBoardPiece - 9] >= 12) {
            selectedPiece.minusEighteenthSpace = true;
        } else {
            selectedPiece.minusEighteenthSpace = false;
        }

    } else {
        if (board[selectedPiece.indexOfBoardPiece + 14] === null 
        && cells[selectedPiece.indexOfBoardPiece + 14].classList.contains("white") !== true
        && board[selectedPiece.indexOfBoardPiece + 7] < 12 && board[selectedPiece.indexOfBoardPiece + 7] !== null) {
            selectedPiece.fourteenthSpace = true;
        } else {
            selectedPiece.fourteenthSpace = false;
        }


        if (board[selectedPiece.indexOfBoardPiece + 18] === null 
        && cells[selectedPiece.indexOfBoardPiece + 18].classList.contains("white") !== true
        && board[selectedPiece.indexOfBoardPiece + 9] < 12 && board[selectedPiece.indexOfBoardPiece + 9] !== null) {
            selectedPiece.eighteenthSpace = true;
        } else {
            selectedPiece.eighteenthSpace = false;
        }

        if (board[selectedPiece.indexOfBoardPiece - 14] === null && cells[selectedPiece.indexOfBoardPiece - 14].classList.contains("white") !== true
        && board[selectedPiece.indexOfBoardPiece - 7] < 12 
        && board[selectedPiece.indexOfBoardPiece - 7] !== null) {
            selectedPiece.minusFourteenthSpace = true;
        } else {
            selectedPiece.minusFourteenthSpace = false;
        }


        if (board[selectedPiece.indexOfBoardPiece - 18] === null && cells[selectedPiece.indexOfBoardPiece - 18].classList.contains("white") !== true
        && board[selectedPiece.indexOfBoardPiece - 9] < 12
        && board[selectedPiece.indexOfBoardPiece - 9] !== null) {
            selectedPiece.minusEighteenthSpace = true; 
        } else {
            selectedPiece.minusEighteenthSpace = false;
        }
    }
    
    for(let i = 0; i<64; i+=1){ //si controlla se qualche pezzo obbliga a mangiare, chiamando su tutti YouGottaEat()
        //Va fatta la verifica e in caso settato found a true 
        if(turn){
            if(board[i] !== null && board[i] <= 11)
                YouGottaEat(i);
        }
        else{
            if(board[i] !== null && board[i] >= 12)
                YouGottaEat(i);
        }
    }
    if(found){ //se qualche pezzo può mangiare, lui non può muoversi senza farlo, quindi gli lasciamo possibili solo le mosse in cui mangia
        selectedPiece.minusSeventhSpace = false;
        selectedPiece.minusNinthSpace = false;
        selectedPiece.ninthSpace = false;
        selectedPiece.seventhSpace = false;
    }
    checkPieceConditions();
}

//se il pezzo non è re dobbiamo impedirgli di fare i movimenti all'indietro
function checkPieceConditions() {

    if (selectedPiece.isKing) {
        givePieceBorder();
    } else { 
       
        if (turn) {
            selectedPiece.minusSeventhSpace = false;
            selectedPiece.minusNinthSpace = false;
            selectedPiece.minusFourteenthSpace = false;
            selectedPiece.minusEighteenthSpace = false;
        } else /*blacks turn: we do the inverse*/ {
            selectedPiece.seventhSpace = false;
            selectedPiece.ninthSpace = false;
            selectedPiece.fourteenthSpace = false;
            selectedPiece.eighteenthSpace = false;            
        }
        givePieceBorder();
    }

}

function givePieceBorder() {
    //se il pezzo può fare almeno un movimento gli coloriamo il bordo
    if (selectedPiece.seventhSpace || selectedPiece.ninthSpace || selectedPiece.fourteenthSpace || selectedPiece.eighteenthSpace
        || selectedPiece.minusSeventhSpace || selectedPiece.minusNinthSpace || selectedPiece.minusFourteenthSpace || selectedPiece.minusEighteenthSpace) {
            
            if (screen.width >= 350 && screen.height >= 700) {
                
                if (turn) {
                   document.getElementById(selectedPiece.pieceId).style.background = colore_bianchi;
                }

                else {
                   document.getElementById(selectedPiece.pieceId).style.background = colore_neri;
                }

                document.getElementById(selectedPiece.pieceId).style.border = "0.3em solid green"; //it's selected
            }

            else {
                document.getElementById(selectedPiece.pieceId).style.border = "0";
                document.getElementById(selectedPiece.pieceId).style.background = "rgb(39, 199, 25)";
            }


            console.log("hai selezionato il pezzo in posizione", selectedPiece.indexOfBoardPiece);
            giveCellsClick();
    } else {
        //il pezzo non può muoversi da nessuna parte
        return;
    }
}

function giveCellsClick() { //diamo la possibilità di muoversi verso le celle in cui il pezzo può andare
    if (selectedPiece.seventhSpace) {
        cells[selectedPiece.indexOfBoardPiece + 7].setAttribute("onclick", "makeMove(7)");
    }
    if (selectedPiece.ninthSpace) {
        cells[selectedPiece.indexOfBoardPiece + 9].setAttribute("onclick", "makeMove(9)");
    }
    if (selectedPiece.fourteenthSpace) {
        cells[selectedPiece.indexOfBoardPiece + 14].setAttribute("onclick", "makeMove(14)");
    }
    if (selectedPiece.eighteenthSpace) {
        cells[selectedPiece.indexOfBoardPiece + 18].setAttribute("onclick", "makeMove(18)");
    }
    if (selectedPiece.minusSeventhSpace) {
        cells[selectedPiece.indexOfBoardPiece - 7].setAttribute("onclick", "makeMove(-7)");
    }
    if (selectedPiece.minusNinthSpace) {
        cells[selectedPiece.indexOfBoardPiece - 9].setAttribute("onclick", "makeMove(-9)");
    }

    if (selectedPiece.minusFourteenthSpace) {
        cells[selectedPiece.indexOfBoardPiece - 14].setAttribute("onclick", "makeMove(-14)");
    }

    if (selectedPiece.minusEighteenthSpace) {
        cells[selectedPiece.indexOfBoardPiece - 18].setAttribute("onclick", "makeMove(-18)");
    }

}

//fine della cascata di funzioni che si attivano quando si clicca su un pezzo


function makeMove(number) {

    //rimuoviamo il pezzo dal front-end perchè lo sposteremo
    
    found = false;
    document.getElementById(selectedPiece.pieceId).remove();

    cells[selectedPiece.indexOfBoardPiece].innerHTML = "";


    //inseriamo il pezzo nella nuova posizione
    if (turn) {
        if (selectedPiece.isKing) {                                    
           
            if (screen.width >= 350 && screen.height >= 700) {
                cells[selectedPiece.indexOfBoardPiece + number].innerHTML = `<p class="white-piece king" id="${selectedPiece.pieceId}"><i class="fa-solid fa-crown fa-flip"></i></p>`;
            }

            else {
                cells[selectedPiece.indexOfBoardPiece + number].innerHTML = `<p class="white-piece king" id="${selectedPiece.pieceId}"></p>`;
            }

            whitesPieces = document.querySelectorAll(".white-piece"); 
        } else {
            cells[selectedPiece.indexOfBoardPiece + number].innerHTML = `<p class="white-piece" id="${selectedPiece.pieceId}"></p>`;
            whitesPieces = document.querySelectorAll(".white-piece");
        }
    } else {   
        if (selectedPiece.isKing) {

            if (screen.width >= 350 && screen.height >= 700) {
                cells[selectedPiece.indexOfBoardPiece + number].innerHTML = `<p class="black-piece king" id="${selectedPiece.pieceId}"><i class="fa-solid fa-crown fa-flip" style="color: white;"></i></p>`;   //WARNING: must use the "backtick" ` symbol
            }
            
            else {
                cells[selectedPiece.indexOfBoardPiece + number].innerHTML = `<p class="black-piece king" id="${selectedPiece.pieceId}"></p>`;   //WARNING: must use the "backtick" ` symbol
            }
            
            blacksPieces = document.querySelectorAll(".black-piece"); 
        } else {
            cells[selectedPiece.indexOfBoardPiece + number].innerHTML = `<p class="black-piece" id="${selectedPiece.pieceId}"></p>`;
            blacksPieces = document.querySelectorAll(".black-piece");
        }
    }

    let indexOfPiece = selectedPiece.indexOfBoardPiece

    //a seconda che il movimento sia standard o una presa, chiamiamo changeData() con 2 o tre argomenti
    if (number == 14 || number == 18 || number == -14 || number == -18) { 
        changeData(indexOfPiece, indexOfPiece + number, indexOfPiece + number / 2 );
    } else {
        changeData(indexOfPiece, indexOfPiece + number);
    }


}


//funzione che cambia i dati della board dopo un movimento
function changeData(indexOfBoardPiece, modifiedIndex, removePiece) {
    board[indexOfBoardPiece] = null;
    board[modifiedIndex] = parseInt(selectedPiece.pieceId);
    if (turn && selectedPiece.pieceId < 12 && modifiedIndex >= 56) { //il pezzo è diventato re
        document.getElementById(selectedPiece.pieceId).classList.add("king");

        if (screen.width >= 350 && screen.height >= 700) {
            /*NOTA: se il pezzo diventà re quando lo schermo non rispetta questa proprietà e poi si ridimensiona lo schermo per rispettare la condizione
            bisognerà aspettare di muovere il pezzo un'altra volta per far apparire la corona*/ 
            cells[modifiedIndex].innerHTML = `<p class="white-piece king" id="${selectedPiece.pieceId}"><i class="fa-solid fa-crown fa-flip"></i></p>`;
        }
        
        else {
            cells[modifiedIndex].innerHTML = `<p class="white-piece king" id="${selectedPiece.pieceId}"></p>`;
        }

       
        whitesPieces = document.querySelectorAll(".white-piece");
    
    }
    if (turn == false && selectedPiece.pieceId >= 12 && modifiedIndex <= 7) { //same but for black
        document.getElementById(selectedPiece.pieceId).classList.add("king");

        if (screen.width >= 350 && screen.height >= 700) {
            cells[modifiedIndex].innerHTML = `<p class="black-piece king" id="${selectedPiece.pieceId}"><i class="fa-solid fa-crown fa-flip" style="color: white;"></i></p>`; 
        }


        else {
            cells[modifiedIndex].innerHTML = `<p class="black-piece king" id="${selectedPiece.pieceId}"></p>`; 
        }
        
       
        blacksPieces = document.querySelectorAll(".black-piece");

    }
    if (removePiece)//c'è stata una presa
        {   
        board[removePiece] = null;
        if (turn && selectedPiece.pieceId < 12) {
            cells[removePiece].innerHTML = "";
            blackScore--;
        }
        if (turn == false && selectedPiece.pieceId >= 12) {
            cells[removePiece].innerHTML = "";
            whiteScore--;
        }
        
        selectedPiece.indexOfBoardPiece = findPiece(selectedPiece.pieceId);
        let keep = checkMultiple(); //controlliamo se il pezzo può fare altre prese (presa multipla), in quel caso deve farlo
        console.log("Valore di keep = " + keep);
        if(keep){
            selectedPiece.seventhSpace = false;
            selectedPiece.minusSeventhSpace = false;
            selectedPiece.ninthSpace = false;
            selectedPiece.minusNinthSpace = false;
            checkAvailableJumpSpaces(); //punto cruciale: se può continuare a mangiare DEVE farlo e quindi rimane il pezzo selezionato
                                        // e ricomincia la cascata ma dando la possibilità di mangiare e basta, quindi chiamo subito
                                        //checkAvailableJumpSpaces() e non checkAvailableSpaces()
            return;
        }
    }

    resetSelectedPieceProperties();
    removeCellonclick();  
    removeEventListeners();
}

function checkMultiple(){ //ritorna true se il pezzo può fare altre prese dalla cella in cui è finito mangiando
    if (turn) {
        if (board[selectedPiece.indexOfBoardPiece + 14] === null 
        && cells[selectedPiece.indexOfBoardPiece + 14].classList.contains("white") !== true
        && board[selectedPiece.indexOfBoardPiece + 7] >= 12){
            return true;
          
            
        }
        if (board[selectedPiece.indexOfBoardPiece + 18] === null 
        && cells[selectedPiece.indexOfBoardPiece + 18].classList.contains("white") !== true
        && board[selectedPiece.indexOfBoardPiece + 9] >= 12) {
            return true;
         
            
        }
        if (selectedPiece.isKing && board[selectedPiece.indexOfBoardPiece - 14] === null 
        && cells[selectedPiece.indexOfBoardPiece - 14].classList.contains("white") !== true
        && board[selectedPiece.indexOfBoardPiece - 7] >= 12) {
            return true;
           
            
        }
        if (selectedPiece.isKing && board[selectedPiece.indexOfBoardPiece - 18] === null 
        && cells[selectedPiece.indexOfBoardPiece - 18].classList.contains("white") !== true
        && board[selectedPiece.indexOfBoardPiece - 9] >= 12) {
            return true;
            
           
        }
    } else {
        if (selectedPiece.isKing && board[selectedPiece.indexOfBoardPiece + 14] === null 
        && cells[selectedPiece.indexOfBoardPiece + 14].classList.contains("white") !== true
        && board[selectedPiece.indexOfBoardPiece + 7] < 12 && board[selectedPiece.indexOfBoardPiece + 7] !== null) {
            return true;
            
            
        }
        if (selectedPiece.isKing && board[selectedPiece.indexOfBoardPiece + 18] === null 
        && cells[selectedPiece.indexOfBoardPiece + 18].classList.contains("white") !== true
        && board[selectedPiece.indexOfBoardPiece + 9] < 12 && board[selectedPiece.indexOfBoardPiece + 9] !== null) {
            return true;
            
           
        }
        if (board[selectedPiece.indexOfBoardPiece - 14] === null && cells[selectedPiece.indexOfBoardPiece - 14].classList.contains("white") !== true
        && board[selectedPiece.indexOfBoardPiece - 7] < 12 
        && board[selectedPiece.indexOfBoardPiece - 7] !== null) {
            return true;
            
            
        }
        if (board[selectedPiece.indexOfBoardPiece - 18] === null && cells[selectedPiece.indexOfBoardPiece - 18].classList.contains("white") !== true
        && board[selectedPiece.indexOfBoardPiece - 9] < 12
        && board[selectedPiece.indexOfBoardPiece - 9] !== null) {
            return true;
            
            
        }
        
    }

    return false;

}


function removeEventListeners() {

    if (turn) {
        for (let i = 0; i < whitesPieces.length; i++) {
            whitesPieces[i].removeEventListener("click", getPlayerPieces);
        }
    } else {
        for (let i = 0; i < blacksPieces.length; i++) {
            blacksPieces[i].removeEventListener("click", getPlayerPieces);
        }
    }
    
    checkForWin();
}


function checkForWin() {
    if (blackScore === 0 ) { //se il bianco ha vinto facciamo apparire un placeholder che indica la vittoria
        divider.style.display = "none";
        placeholder_for_win_message = document.getElementById("placeholder-for-win-message-id").innerHTML = 
            `<div class="modal-container show" id="modal-container-id">
                <div class="modal">
                    <form action="" method="" id = "form">  
                        <div class="user-details">
                            <div class="input-box">
                                <span class="details">${whiteTurnText.textContent} wins! </span>
                            </div>
                        </div>
                        <div class="bottone">
                            <button type="button" onclick = "location.href='game.php'" id = "play again">Play Again</button> 
                        </div>
                        <div class="bottone">
                            <button type="button" onclick = "location.href='index.php'" id = "home">Home</button> 
                        </div>
                    </form>
                </div>
            </div>`;

         //comunichiamo con il database per aggiornare il numero di vittorie del vincitore
        var xmlhttp = new XMLHttpRequest();
        console.log( "contenuto del testo:" + whiteTurnText.getAttribute("name"))
        xmlhttp.open("GET","winner.php?q="+whiteTurnText.getAttribute("name"), true );
        xmlhttp.send();
        console.log("richiesta fatta");


    } else if (whiteScore === 0) {
        divider.style.display = "none";

        placeholder_for_win_message = document.getElementById("placeholder-for-win-message-id").innerHTML = 
            `<div class="modal-container show" id="modal-container-id">
                <div class="modal">
                    <form action="" method="" id = "form">  
                        <div class="user-details">
                            <div class="input-box">
                                <span class="details"> ${blackTurnText.textContent} wins! </span>
                            </div>
                        </div>
                        <div class="bottone">
                            <button type="button" onclick = "location.href='game.php'" id = "play again">Play Again</button> 
                        </div>
                        <div class="bottone">
                            <button type="button" onclick = "location.href='index.php'" id = "home">Home</button> 
                        </div>
                    </form>
                </div>
            </div>`;
            
        //comunichiamo con il database per aggiornare il numero di vittorie del vincitore
        var xmlhttp = new XMLHttpRequest();
        console.log( "contenuto del testo:" + blackTurnText.getAttribute("name"));
        xmlhttp.open("GET","winner.php?q="+blackTurnText.getAttribute("name"), true );
        xmlhttp.send();
        console.log("richiesta fatta");
        
    }
    changePlayer();
}

function changePlayer() {
    if (turn) {
        turn = false;
        whiteTurnText.style.color = "lightGrey";
        blackTurnText.style.color = "black";
        for(let i = 0; i < blacksPieces.length;i++)
            blacksPieces[i].classList.add('turno');
        for(let i = 0; i < whitesPieces.length;i++)
            whitesPieces[i].classList.remove('turno');
    } else {
        turn = true;
        for(let i = 0; i < blacksPieces.length;i++)
            blacksPieces[i].classList.remove('turno');
        for(let i = 0; i < whitesPieces.length;i++)
            whitesPieces[i].classList.add('turno');
        whiteTurnText.style.color = "black";
        blackTurnText.style.color = "lightGrey";

    }
    givePiecesEventListeners();

}



//PER IL FORM DI LOGIN 
const modal_container = document.getElementById("modal-container-id");

const close = document.getElementById("form");

modal_container.classList.add('show');


   const usr1 = document.getElementById("user1name");
   const usr2 = document.getElementById("user2name");
   const pwd1 = document.getElementById("pwd");
   const pwd2 = document.getElementById("cpwd");

   //di seguito una serie di controlli lato server che verificano se
   //l'utente ha inserito username e password corretti

   pwd1.addEventListener("input", (event)=>{

   var xmlhttp = new XMLHttpRequest();
   xmlhttp.onreadystatechange = () => {
    if(xmlhttp.readyState === 4) {
        if(xmlhttp.status === 200){
            testo = xmlhttp.responseText.trim();
            console.log("Risposta:" + testo);
            if(testo == "no"){
                usr1.setCustomValidity("Inserire username/password corretti!");
                pwd1.setCustomValidity("Inserire username/password corretti!");
            }
            else{
                usr1.setCustomValidity("");
                pwd1.setCustomValidity("");
            }
        }
    }
   }

   xmlhttp.open("GET",`popup.php?usr=${usr1.value}&pwd=${pwd1.value}`, true );
   xmlhttp.send();



   })

   usr1.addEventListener("input", (event)=>{
    
    var xmlhttp = new XMLHttpRequest();
    xmlhttp.onreadystatechange = () => {
     if(xmlhttp.readyState === 4) {
         if(xmlhttp.status === 200){
             testo = xmlhttp.responseText.trim();
             console.log("Risposta:" + testo);
             if(testo == "no"){
                 usr1.setCustomValidity("Inserire username/password corretti!");
                 pwd1.setCustomValidity("Inserire username/password corretti!");
             }
             else{
                 usr1.setCustomValidity("");
                 pwd1.setCustomValidity("");
             }
         }
     }
    }
 
 
    xmlhttp.open("GET",`popup.php?usr=${usr1.value}&pwd=${pwd1.value}`, true );
    xmlhttp.send();
 
 
 
    })

   pwd2.addEventListener("input", (event)=>{
    
   var xmlhttp = new XMLHttpRequest();
   xmlhttp.onreadystatechange = () => {
    if(xmlhttp.readyState === 4) {
        if(xmlhttp.status === 200){
            testo = xmlhttp.responseText.trim();
                console.log("Risposta:" + testo);
                if(testo == "no"){
                    usr2.setCustomValidity("Inserire username/password corretti!");
                    pwd2.setCustomValidity("Inserire username/password corretti!");
                }
                else{
                    usr2.setCustomValidity("");
                    pwd2.setCustomValidity("");
                }
        }
    }
   }
   xmlhttp.open("GET",`popup.php?usr=${usr2.value}&pwd=${pwd2.value}`, true );
   xmlhttp.send();
   })

   usr2.addEventListener("input", (event)=>{
    
    var xmlhttp = new XMLHttpRequest();
    xmlhttp.onreadystatechange = () => {
     if(xmlhttp.readyState === 4) {
         if(xmlhttp.status === 200){
             testo = xmlhttp.responseText.trim();
                 console.log("Risposta:" + testo);
    
                 if(testo == "no"){
                     usr2.setCustomValidity("Inserire username/password corretti!");
                     pwd2.setCustomValidity("Inserire username/password corretti!");
                 }
                 else{
                     usr2.setCustomValidity("");
                     pwd2.setCustomValidity("");
                 }
         }
     }
    }
    xmlhttp.open("GET",`popup.php?usr=${usr2.value}&pwd=${pwd2.value}`, true );
    xmlhttp.send();
    })

    //controllo extra che verifica se stiamo inserendo lo stesso utente due volte
close.addEventListener("submit", (event)=>{
    event.preventDefault();
    testo1 = usr1.value;
    testo2 = usr2.value;
    if(testo1 == testo2){
        alert("Inserire due utenti diversi!!!");
        return;
    }
    whiteTurnText.textContent = `${testo1} (WHITE)`;
    blackTurnText.textContent = `${testo2} (BLACK)`;
    whiteTurnText.style.fontFamily= "Luckiest Guy";
    blackTurnText.style.fontFamily= "Luckiest Guy"
    whiteTurnText.setAttribute("name", testo1 );
    blackTurnText.setAttribute("name", testo2);
    modal_container.classList.remove('show');
    window.setTimeout(function () {
        // rimuovo proprio il form dalla pagina in modo che non impedisca di selezionare le pedine 
        
        var parent = document.getElementById("body");
        var child = document.getElementById("modal-container-id");
        parent.removeChild(child);
    }, 1000) //il timer serve a far vedere la transizione css

});




//parte il programma
givePiecesEventListeners();

