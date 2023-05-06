let username = document.getElementById("username");
let old_password = document.getElementById("pwd");
let new_password = document.getElementById("cpwd");

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
                    console.log("vecchia password: " + old_password.value);
                    console.log("nuova password: " + new_password.value);
                    if(testo == 'fatto'){
                        alert("password modificata");
                        window.location.href = "settings.php";
                    }
                    else if(testo == 'errore'){
                        alert("Utente inesistente o password errata");
                    }
                    else{
                        console.log("C'è un problema");
                    }
                }
            }
           }
        xmlhttp.open("GET",`req_change.php?usr=${username.value}&oldpwd=${old_password.value}&newpwd=${new_password.value}`, true);
        xmlhttp.send();
    })
});