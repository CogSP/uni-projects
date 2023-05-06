/* GAME STATE DATA: declaration of the board, with all the pieces and their
html's ids */

var colore_bianchi  
var colore_neri

xmlhttp = new XMLHttpRequest();
        xmlhttp.onreadystatechange = () => {
            if(xmlhttp.readyState === 4) {
                if(xmlhttp.status === 200){

                    testo = xmlhttp.responseText;
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


const board = [ /*64 item array*/
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

let findPiece = function(pieceId) { //La funzione restituisce, dato l'id html del pezzo, il corrispondente indice nella damiera di back-end
    let parsed = parseInt(pieceId);
    return board.indexOf(parsed);
}


// references to the web html page (DOM references)
const cells = document.querySelectorAll("td"); /* list of all the "td"s in the  html file*/
const pieces = document.querySelectorAll("p");
let whitesPieces = document.querySelectorAll(".white-piece"); /* does it work? */
let blacksPieces = document.querySelectorAll(".black-piece"); /* does it work? */
const whiteTurnText = document.getElementById("wtt");
const blackTurnText = document.getElementById("btt");
const divider = document.querySelector("#divider")


/* players properties: game state */
let turn = true;  /* true = white, false = black*/ 
let whiteScore = 12; // how many pieces the white player currently has
let blackScore = 12;
let playerPieces;  // if its white's turn playerPieces = whitesPieces, otherwise 
                   //playerPieces = blacksPieces


let selectedPiece = {
    // default values when no piece is selected
    pieceId: -1,
    indexOfBoardPiece: -1,
    isKing: false,

    /* these are all the moves and whether they are possible or not */
    /*TODO: check if it's really that*/ 
    seventhSpace: false,
    ninthSpace: false,
    fourteenthSpace: false,
    eighteenthSpace: false,
    // +7/+9 for whites, -7/-9 for blacks
    // we didnt' create two variables because when a piece become king, regardless
    // of its color, it can move in all the four positions 
    minusSeventhSpace: false,
    minusNinthSpace: false,
    minusFourteenthSpace: false,
    minusEighteenthSpace: false,
}


    /* EVENT LISTENER */

// give to all the pieces an event listener. This event listener, when the 
//piece is clicked, will invoke the function "getPlayerPieces"
function givePiecesEventListeners() {

    if (turn) {
        //console.log("white's turn");
        
        for (let i = 0; i < whitesPieces.length; i++) {
            blacksPieces[i].classList.remove("turno");
            whitesPieces[i].classList.add("turno");
            whitesPieces[i].addEventListener("click", getPlayerPieces);
        }
    } else {
        
        //console.log("black's turn");
        for (let i = 0; i < blacksPieces.length; i++) {
            whitesPieces[i].classList.remove("turno");
            blacksPieces[i].classList.add("turno");
            blacksPieces[i].addEventListener("click", getPlayerPieces);
        }
    }
}

    /* END OF EVENT LISTENER CONFIGURATION */


/* here starts the functions chain for when we click a piece: */

function getPlayerPieces() {
    //whites turn
    if (turn) {
        //console.log("a white piece has been clicked");
        playerPieces = whitesPieces;
    } else {
        //console.log("a black piece has been clicked");
        playerPieces = blacksPieces;
    }

        //Controllo per ogni bianco che non possa mangiare; se un qualsiasi bianco può mangiare
        //il pezzo selezionato non dove poterlo fare
    

    removeCellonclick();
    resetBorders();
}

// we remove the "onclick" attribute from all the cells
// after, in the functions chain, we are going to re-calculate which cells should have this attribute
function removeCellonclick() {
    for (let i = 0; i < cells.length; i++) {
        cells[i].removeAttribute("onclick");
    }
}

// resetting the borders color to the initial value, so we can later just color the one that is selected
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
    getSelectedPiece(); //here we start operating on the current selected piece
}

function resetSelectedPieceProperties() {
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


// usint findPiece(), this will get us the board's index where the
//selected piece is located
function getSelectedPiece() {
    // we use "event.target.id" but event is now depreecated 
    // TODO: replace it

    selectedPiece.pieceId = parseInt(event.target.id);

    console.log("UDDIO QUALCUNO HA CLICCATO IL PEZZO CON ID", selectedPiece.pieceId)

    selectedPiece.indexOfBoardPiece = findPiece(selectedPiece.pieceId);
    
    isPieceKing();
}



function isPieceKing() {

    //element.classList give us the list of classes to which our element belongs
    //console.log("sto per controllare se l'elemento è king. L'elemento ha id = " + selectedPiece.pieceId);
    console.log("il selected piece ha Id", selectedPiece.pieceId);
    console.log("l'elemento è", document.getElementById(selectedPiece.pieceId));
    if (document.getElementById(selectedPiece.pieceId).classList.contains("king")) {
        selectedPiece.isKing = true;
    } else {
        selectedPiece.isKing = false;
    }

    getAvailableSpaces();
}

function getAvailableSpaces() { 
    //we are using the string equality operator

    if (board[selectedPiece.indexOfBoardPiece + 7] === null /*the cell is available (id == null -> no piece here)*/ 
        && cells[selectedPiece.indexOfBoardPiece + 7].classList.contains("white") !== true) { /*never go on white cells: this is 
        important because if the piece is on the edge, +7/+9 will get us on the opposite end of the board, so we use the fact that these cells are white to prevent jumping there*/
            selectedPiece.seventhSpace = true; // moving to upper-right is available
    }

    if (board[selectedPiece.indexOfBoardPiece + 9] === null &&
        cells[selectedPiece.indexOfBoardPiece + 9].classList.contains("white") !== true) {
            selectedPiece.ninthSpace = true; 
    }

    // same for -7 and -9
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


function YouGottaEat(index) { //funzione che controlla se un certo pezzo sulla scacchiera (generico, non il selezionato) può mangiare
    
    console.log("controlliamo se il pezzo", index, "è obbligato a mangiare")
    
    if (turn) {

        if (
        index + 9 < 64 && index + 18 < 64 &&
        board[index + 14] === null 
        && cells[index + 14].classList.contains("white") !== true
        && board[index + 7] !== null
        && board[index + 7] >= 12) /*black pieces have id >= 12*/ {
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
        
        if(index - 7 >= 0 && index - 14 >=0) {
            console.log("board[index - 14] === null:", board[index - 14] === null);
            console.log("cells[index - 14].classList.contains(white) !== true:", cells[index - 14].classList.contains("white") !== true );
            console.log("document.getElementById(board[index]).classList.contains('king'):", document.getElementById(board[index]).classList.contains("king"));
            console.log("board[index -7] !== null: ", board[index -7] !== null);
            console.log("board[index - 7] >= 12: ", board[index - 7] >= 12);
            console.log("\n");
            console.log("ora printo ClassList:", cells[index].classList)
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

        if(index - 9 >= 0 && index - 18 >=0){
        console.log("Condizioni verificate: ");
        console.log("board[index - 18] === null: ", board[index - 18] === null );
        console.log("cells[index - 18].classList.contains(white) !== true:", cells[index - 18].classList.contains("white") !== true);
        console.log("document.getElementById(board[index]).classList.contains('king')", document.getElementById(board[index]).classList.contains("king"));
        console.log("board[index -9] !== null: ", board[index -9] !== null);
        console.log("board[index - 9] >= 12: ", board[index - 9] >= 12);
        console.log("\n");
        console.log("ora printo ClassList:", cells[index].classList)
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

        if (index + 7 < 64 && index + 18 < 64) {
            console.log("Condizioni verificate:");
            console.log("board[index + 14] === null", board[index + 14] === null);
            console.log("index + 7 < 64 && index + 14 < 64", index + 7 < 64 && index + 14 < 64);
            console.log("document.getElementById(board[index]).classList.contains('king')", document.getElementById(board[index]).classList.contains("king"))
            console.log("cells[index + 14].classList.contains('white') !== true", cells[index + 14].classList.contains("white") !== true)
            console.log("board[index + 7] < 12 && board[index + 7] !== null", board[index + 7] < 12 && board[index + 7] !== null)
            console.log("ora printo ClassList:", cells[index].classList)
        }

        if (board[index + 14] === null
        &&  index + 7 < 64 && index + 14 < 64
        && document.getElementById(board[index]).classList.contains("king")
        && cells[index + 14].classList.contains("white") !== true
        && board[index + 7] < 12 && board[index + 7] !== null) {
            console.log("poiché questo nero è un re, può mangiare al contrario un pezzo nero in +7 per poi andare in +14")
            console.log("indexKing = ", index.toString());
            found = true;
        }


        if (index + 9 < 64 && index + 18 < 64) {
            console.log("board[index + 18] === null", board[index + 18] === null)
            console.log("index + 9 < 64 && index + 18 < 64", index + 9 < 64 && index + 18 < 64)
            console.log("document.getElementById(board[index]).classList.contains('king')", document.getElementById(board[index]).classList.contains("king"))
            console.log("cells[index + 18].classList.contains('white') !== true", cells[index + 18].classList.contains("white") !== true)
            console.log("board[index + 9] < 12 && board[index + 9] !== null", board[index + 9] < 12 && board[index + 9] !== null)
            console.log("ora printo ClassList:", cells[index].classList)
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

// IS THERE A WAY TO ELMINATE THIS IF-ELSE BRANCH, DOING JUST ONE CODE FOR BOTH THE TEAMS?
// THE PROBLEM IS THAT WE NEED TO DIFFERENTIATE BETWEEN ID >= 12 (BLACKS) AND < 12 (WHITES)
function checkAvailableJumpSpaces() {

    if (turn) {

        if (board[selectedPiece.indexOfBoardPiece + 14] === null 
        && cells[selectedPiece.indexOfBoardPiece + 14].classList.contains("white") !== true
        && board[selectedPiece.indexOfBoardPiece + 7] >= 12) /*black pieces have id >= 12*/ {
            console.log("Non entrare qui!!!");
            selectedPiece.fourteenthSpace = true;
            // found = true;
        } else { /*PEFFORZA ALTRIMENTI C'È QUEL PROBLEMA CHE DICEVO A ISIDORO PER CUI CON LA MOSSA SOLITA PER FARE LA DOPPIA MANGIATA E DIVENTARE CON IL BIANCO RE POTEVO ANDARE DUE VOLTE A SINISTRA MANGIANDO SOPRA UNA PEDINA*/
        /*NON HO ANCORA CAPITO DOVE PERO' FOURTEENTHSPACE VENIVA RESA TRUE ANCHE SE DOVEVA ESSERE FALSE*/ 
            selectedPiece.fourteenthSpace = false;
        }


        if (board[selectedPiece.indexOfBoardPiece + 18] === null 
        && cells[selectedPiece.indexOfBoardPiece + 18].classList.contains("white") !== true
        && board[selectedPiece.indexOfBoardPiece + 9] >= 12) {
            selectedPiece.eighteenthSpace = true;
            // found = true;
        } else {
            selectedPiece.eighteenthSpace = false;
        }


        if (board[selectedPiece.indexOfBoardPiece - 14] === null 
        && cells[selectedPiece.indexOfBoardPiece - 14].classList.contains("white") !== true
        && board[selectedPiece.indexOfBoardPiece - 7] >= 12) {
            selectedPiece.minusFourteenthSpace = true;
            // found = true;
        } else {
            selectedPiece.minusFourteenthSpace = false;
        }


        if (board[selectedPiece.indexOfBoardPiece - 18] === null 
        && cells[selectedPiece.indexOfBoardPiece - 18].classList.contains("white") !== true
        && board[selectedPiece.indexOfBoardPiece - 9] >= 12) {
            selectedPiece.minusEighteenthSpace = true;
            // found = true;
        } else {
            selectedPiece.minusEighteenthSpace = false;
        }

    } else {
        if (board[selectedPiece.indexOfBoardPiece + 14] === null 
        && cells[selectedPiece.indexOfBoardPiece + 14].classList.contains("white") !== true
        && board[selectedPiece.indexOfBoardPiece + 7] < 12 && board[selectedPiece.indexOfBoardPiece + 7] !== null) {
            selectedPiece.fourteenthSpace = true;
            // found = true;
        } else {
            selectedPiece.fourteenthSpace = false;
        }


        if (board[selectedPiece.indexOfBoardPiece + 18] === null 
        && cells[selectedPiece.indexOfBoardPiece + 18].classList.contains("white") !== true
        && board[selectedPiece.indexOfBoardPiece + 9] < 12 && board[selectedPiece.indexOfBoardPiece + 9] !== null) {
            selectedPiece.eighteenthSpace = true;
            // found = true;
        } else {
            selectedPiece.eighteenthSpace = false;
        }

        if (board[selectedPiece.indexOfBoardPiece - 14] === null && cells[selectedPiece.indexOfBoardPiece - 14].classList.contains("white") !== true
        && board[selectedPiece.indexOfBoardPiece - 7] < 12 
        && board[selectedPiece.indexOfBoardPiece - 7] !== null) {
            selectedPiece.minusFourteenthSpace = true;
            // found = true;
        } else {
            selectedPiece.minusFourteenthSpace = false;
        }


        if (board[selectedPiece.indexOfBoardPiece - 18] === null && cells[selectedPiece.indexOfBoardPiece - 18].classList.contains("white") !== true
        && board[selectedPiece.indexOfBoardPiece - 9] < 12
        && board[selectedPiece.indexOfBoardPiece - 9] !== null) {
            selectedPiece.minusEighteenthSpace = true;
            // found = true;   
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
    if(found){ //Se qualche pezzo può mangiare, lui non può muoversi senza farlo!
        selectedPiece.minusSeventhSpace = false;
        selectedPiece.minusNinthSpace = false;
        selectedPiece.ninthSpace = false;
        selectedPiece.seventhSpace = false;
    }
    checkPieceConditions();
}

// restrict the movements if the piece is not a king
function checkPieceConditions() {

    if (selectedPiece.isKing) {
        givePieceBorder();
    } else {
        //whites turn, the piece is not a king: it can't move -7/-9, so we set this variables to false 
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
        //this piece can't move
        return;
    }
}

function giveCellsClick() {
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

/* end of the functions chain for when you click a cell*/


function makeMove(number) {

    //remove the piece from the front end because we are moving elsewhere
    
    found = false;
    document.getElementById(selectedPiece.pieceId).remove();

    cells[selectedPiece.indexOfBoardPiece].innerHTML = "";


    // to save in javascript memory the new position of the piece
    if (turn) {
        if (selectedPiece.isKing) {                                     // these are two classes: white-piece and king
           
            if (screen.width >= 350 && screen.height >= 700) {
                cells[selectedPiece.indexOfBoardPiece + number].innerHTML = `<p class="white-piece king" id="${selectedPiece.pieceId}"><i class="fa-solid fa-crown fa-flip"></i></p>`;
            }

            else {
                cells[selectedPiece.indexOfBoardPiece + number].innerHTML = `<p class="white-piece king" id="${selectedPiece.pieceId}"></p>`;
            }

            console.log("MAONNA RAGAZZI IL PEZZO CON ID", selectedPiece.pieceId, "E' DIVENTATO LEZZO");
            whitesPieces = document.querySelectorAll(".white-piece"); /* why am I recalculating this? */
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
            
            console.log("MAONNA RAGAZZI IL PEZZO CON ID", selectedPiece.pieceId, "E' DIVENTATO LEZZO");
            blacksPieces = document.querySelectorAll(".black-piece"); 
        } else {
            cells[selectedPiece.indexOfBoardPiece + number].innerHTML = `<p class="black-piece" id="${selectedPiece.pieceId}"></p>`;
            blacksPieces = document.querySelectorAll(".black-piece");
        }
    }

    // we can't pass object properties directly into the arguments of the function (why?) so I need to save it
    let indexOfPiece = selectedPiece.indexOfBoardPiece
    if (number == 14 || number == 18 || number == -14 || number == -18) /* the piece is eating someone else*/ { //IN QUESTO CASO DOVREMMO CONTROLLARE EVENTUALI MANGIATE MULTIPLE
        changeData(indexOfPiece, indexOfPiece + number, indexOfPiece + number / 2 /*position of the eaten piece*/);
    } else {
        changeData(indexOfPiece, indexOfPiece + number);
    }


}


// this will change the data stored in the back end
function changeData(indexOfBoardPiece, modifiedIndex, removePiece) {
    board[indexOfBoardPiece] = null;
    board[modifiedIndex] = parseInt(selectedPiece.pieceId);
    if (turn && selectedPiece.pieceId < 12 && modifiedIndex >= 56) { //the piece reached the end: it become a king
        document.getElementById(selectedPiece.pieceId).classList.add("king");

        if (screen.width >= 350 && screen.height >= 700) {
            /*NOTA: se il pezzo diventà re quando lo schermo non rispetta questa proprietà e poi si ridimensiona lo schermo per rispettare la condizione
            bisognerà aspettare di muovere il pezzo un'altra volta per far apparire la corona*/ 
            cells[modifiedIndex].innerHTML = `<p class="white-piece king" id="${selectedPiece.pieceId}"><i class="fa-solid fa-crown fa-flip"></i></p>`;
        }
        
        else {
            cells[modifiedIndex].innerHTML = `<p class="white-piece king" id="${selectedPiece.pieceId}"></p>`;
        }

        console.log("MAONNA RAGAZZI IL PEZZO CON ID", selectedPiece.pieceId, "E' DIVENTATO LEZZO");
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
        
        console.log("MAONNA RAGAZZI IL PEZZO CON ID", selectedPiece.pieceId, "E' DIVENTATO LEZZO");
        blacksPieces = document.querySelectorAll(".black-piece");

    }
    if (removePiece) /*someone got eaten*/ {
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
        let keep = checkMultiple();
        console.log("Valore di keep = " + keep);
        if(keep){
            selectedPiece.seventhSpace = false;
            selectedPiece.minusSeventhSpace = false;
            selectedPiece.ninthSpace = false;
            selectedPiece.minusNinthSpace = false;
            checkAvailableJumpSpaces(); //punto cruciale: se può continuare a mangiare DEVE farlo e quindi rimane il pezzo selezionato
            return;
        }
    }

    resetSelectedPieceProperties();
    removeCellonclick();  //these first two are necessary for the next turn
    removeEventListeners();
}

function checkMultiple(){
    if (turn) {
        if (board[selectedPiece.indexOfBoardPiece + 14] === null 
        && cells[selectedPiece.indexOfBoardPiece + 14].classList.contains("white") !== true
        && board[selectedPiece.indexOfBoardPiece + 7] >= 12) /*black pieces have id >= 12*/ {
            return true;
            // found = true;
            
        }
        if (board[selectedPiece.indexOfBoardPiece + 18] === null 
        && cells[selectedPiece.indexOfBoardPiece + 18].classList.contains("white") !== true
        && board[selectedPiece.indexOfBoardPiece + 9] >= 12) {
            return true;
            // found = true;
            
        }
        if (selectedPiece.isKing && board[selectedPiece.indexOfBoardPiece - 14] === null 
        && cells[selectedPiece.indexOfBoardPiece - 14].classList.contains("white") !== true
        && board[selectedPiece.indexOfBoardPiece - 7] >= 12) {
            return true;
            // found = true;
            
        }
        if (selectedPiece.isKing && board[selectedPiece.indexOfBoardPiece - 18] === null 
        && cells[selectedPiece.indexOfBoardPiece - 18].classList.contains("white") !== true
        && board[selectedPiece.indexOfBoardPiece - 9] >= 12) {
            return true;
            // found = true;
           
        }
    } else {
        if (selectedPiece.isKing && board[selectedPiece.indexOfBoardPiece + 14] === null 
        && cells[selectedPiece.indexOfBoardPiece + 14].classList.contains("white") !== true
        && board[selectedPiece.indexOfBoardPiece + 7] < 12 && board[selectedPiece.indexOfBoardPiece + 7] !== null) {
            return true;
            // found = true;
            
        }
        if (selectedPiece.isKing && board[selectedPiece.indexOfBoardPiece + 18] === null 
        && cells[selectedPiece.indexOfBoardPiece + 18].classList.contains("white") !== true
        && board[selectedPiece.indexOfBoardPiece + 9] < 12 && board[selectedPiece.indexOfBoardPiece + 9] !== null) {
            return true;
            // found = true;
           
        }
        if (board[selectedPiece.indexOfBoardPiece - 14] === null && cells[selectedPiece.indexOfBoardPiece - 14].classList.contains("white") !== true
        && board[selectedPiece.indexOfBoardPiece - 7] < 12 
        && board[selectedPiece.indexOfBoardPiece - 7] !== null) {
            return true;
            // found = true;
            
        }
        if (board[selectedPiece.indexOfBoardPiece - 18] === null && cells[selectedPiece.indexOfBoardPiece - 18].classList.contains("white") !== true
        && board[selectedPiece.indexOfBoardPiece - 9] < 12
        && board[selectedPiece.indexOfBoardPiece - 9] !== null) {
            return true;
            // found = true;
            
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

// it also changes the player
function checkForWin() {
    if (blackScore === 0 ) {
        divider.style.display = "none";
        // for (let i = 0; i < whiteTurnText.length; i++) {
        //     whiteTurnText[i].style.color = "black";
        //     blackTurnText[i].style.display = "none";
        //     whiteTurnText[i].textContent = "WHITE WINS!";
        
        //     // aggiungiamo un pezzo HTML con il pop up
        //     // NOTA: Importante che modal-container abbia come ulteriore classe show altrimenti non si vede il pop up

        // }
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


    } else if (whiteScore === 0 ) {
        divider.style.display = "none";

        // for (let i = 0; i < blackTurnText.length; i++) {            
        //     blackTurnText[i].style.color = "black";
        //     whiteTurnText[i].style.display = "none";
        //     blackTurnText[i].textContent = "BLACK WINS!";


        //     // aggiungiamo un pezzo HTML con il pop up
        //     // NOTA: Importante che modal-container abbia come ulteriore classe show altrimenti non si vede il pop up
            
        // }
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
    } else {
        turn = true;
        whiteTurnText.style.color = "black";
        blackTurnText.style.color = "lightGrey";

    }
    givePiecesEventListeners();

}



// FOR THE LOGIN FORM
const modal_container = document.getElementById("modal-container-id");

const close = document.getElementById("form");

modal_container.classList.add('show');


   const usr1 = document.getElementById("user1name");
   const usr2 = document.getElementById("user2name");
   const pwd1 = document.getElementById("pwd");
   const pwd2 = document.getElementById("cpwd");

   pwd1.addEventListener("input", (event)=>{

   var xmlhttp = new XMLHttpRequest();
   xmlhttp.onreadystatechange = () => {
    if(xmlhttp.readyState === 4) {
        if(xmlhttp.status === 200){
            testo = xmlhttp.responseText;
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
             testo = xmlhttp.responseText;
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
            testo = xmlhttp.responseText;
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
             testo = xmlhttp.responseText;
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




//starting point: the cycle begins once the page has loaded
givePiecesEventListeners();

