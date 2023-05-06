console.log("provaaaaa");

$(document).ready(function(){
    // paginagioco = window.open("game.php");
    
    $("#ct-std").click(function(event){
        event.preventDefault();
        document.getElementById("ct-std").classList.add("active");
        document.getElementById("ct-dark").classList.remove("active");
        document.getElementById("ct-trop").classList.remove("active");
        // paginagioco.document.getElementById("foglioback").href = "background.css";
        xmlhttp = new XMLHttpRequest();
        xmlhttp.onreadystatechange = () => {
            if(xmlhttp.readyState === 4) {
                if(xmlhttp.status === 200){

                    testo = xmlhttp.responseText;
                    console.log("risposta: " + testo);
                    
                }
            }
           }
        xmlhttp.open("GET",`change_theme.php?theme=std`, true);
        xmlhttp.send();

    });
    

    $("#ct-dark").click(function(event){
        event.preventDefault();
        document.getElementById("ct-dark").classList.add("active");
        document.getElementById("ct-trop").classList.remove("active");
        document.getElementById("ct-std").classList.remove("active");
        xmlhttp = new XMLHttpRequest();
        xmlhttp.onreadystatechange = () => {
            if(xmlhttp.readyState === 4) {
                if(xmlhttp.status === 200){

                    testo = xmlhttp.responseText;
                    console.log("risposta: " + testo);
                    
                }
            }
           }
        xmlhttp.open("GET",`change_theme.php?theme=dark`, true);
        xmlhttp.send();
        // paginagioco.document.getElementById("foglioback").href = "background_dark.css";
    });

    $("#ct-trop").click(function(event){
        event.preventDefault();
        document.getElementById("ct-trop").classList.add("active");
        document.getElementById("ct-dark").classList.remove("active");
        document.getElementById("ct-std").classList.remove("active");
        // paginagioco.document.getElementById("foglioback").href = "background_trop.css";
        xmlhttp = new XMLHttpRequest();
        xmlhttp.onreadystatechange = () => {
            if(xmlhttp.readyState === 4) {
                if(xmlhttp.status === 200){

                    testo = xmlhttp.responseText;
                    console.log("risposta: " + testo);
                    
                }
            }
           }
        xmlhttp.open("GET",`change_theme.php?theme=trop`, true);
        xmlhttp.send();
    });

    $("#gm-alt").click(function(event){
        event.preventDefault();
        document.getElementById("gm-alt").classList.add("active");
        document.getElementById("gm-std").classList.remove("active");
    });

    $("#gm-std").click(function(event){
        event.preventDefault();
        document.getElementById("gm-std").classList.add("active");
        document.getElementById("gm-alt").classList.remove("active");
    });

});