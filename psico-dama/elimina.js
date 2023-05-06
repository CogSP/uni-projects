
let username = document.getElementById("username");
let password = document.getElementById("pwd");

$(document).ready(function(){
    $("#form").submit(function(event){
        event.preventDefault();
        xmlhttp = new XMLHttpRequest();
        xmlhttp.onreadystatechange = () => {
            if(xmlhttp.readyState === 4) {
                if(xmlhttp.status === 200){

                    testo = xmlhttp.responseText;
                    console.log("risposta: " + testo);
                    console.log("username: " + username.value);
                    console.log("password: " + password.value);
                    if(testo == 'fatto'){
                        alert("utente eliminato");
                        window.location.href = "settings.php";
                    }
                    else if(testo == 'errore'){
                        alert("Utente inesistente");
                    }
                    else{
                        console.log("C'è un problema");
                    }
                }
            }
           }
        xmlhttp.open("GET",`req_delete.php?usr=${username.value}&pwd=${password.value}`, true);
        xmlhttp.send();
    })
});