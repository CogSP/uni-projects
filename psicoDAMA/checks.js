//Qui ci sono i check (per ora solo il controllo che la conferma è uguale
//alla password) del form di registrazione

const form = document.querySelector("form");
const password = document.getElementById("pwd");
const conferma = document.getElementById("cpwd");
const username = document.getElementById("username");
const fullname = document.getElementById("fullname")

conferma.addEventListener("input",(event)=>{
  if(password.value!= conferma.value){
    conferma.setCustomValidity("passwords need to match!")
  }
  else{
    conferma.setCustomValidity(""); //forse non serve
    //tutto ok, il form viene mandato
  }
});

username.addEventListener("input", (event)=>{
    var xmlhttp = new XMLHttpRequest();

    xmlhttp.onreadystatechange = () => {
        if(xmlhttp.readyState === 4) {
            if(xmlhttp.status === 200) {
                testo = xmlhttp.responseText;
                console.log("Risposta:" + testo);
                if(testo == "no"){
                    username.setCustomValidity("Username già in utilizzo!");
                }
                else{
                    username.setCustomValidity("");
                }
                
            } else {
                console.log('Error Code: ' + xmlhttp.status);
                console.log('Error Message: ' + xmlhttp.statusText);
            }
        }
    }


    xmlhttp.open("GET","checks.php?q="+username.value, true );
    xmlhttp.send();
  
})

