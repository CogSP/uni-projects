//Qui ci sono i check del form di registrazione

const form = document.querySelector("form");
const password = document.getElementById("pwd");
const conferma = document.getElementById("cpwd");
const username = document.getElementById("username");
const fullname = document.getElementById("fullname")

conferma.addEventListener("input",(event)=>{
  if(password.value!= conferma.value){
    conferma.setCustomValidity("Le password non coincidono!")
  }
  else{
    conferma.setCustomValidity(""); 
    //tutto ok, il form viene mandato
  }
});

username.addEventListener("input", (event)=>{
    var xmlhttp = new XMLHttpRequest();

    xmlhttp.onreadystatechange = () => {
        if(xmlhttp.readyState === 4) {
            if(xmlhttp.status === 200) {
                testo = xmlhttp.responseText.trim();
                console.log(testo);
                if(testo == "no"){
                    console.log("settato a no")
                    username.setCustomValidity("Username già in utilizzo!");
                }
                else{
                    console.log("settato a si")
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

