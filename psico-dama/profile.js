//Andrebbe rinominato perchè in realtà alla fine in questo file ci sono solo
//le cose per far funzionare mostra/nascondi password

function showPwd() {
  var input = document.getElementById('pwd');
  if (input.type === "password") {
    input.type = "text";
  } else {
    input.type = "password";
  }
}

function showCpwd() {
  var input = document.getElementById('cpwd');
  if (input.type === "password") {
    input.type = "text";
  } else {
    input.type = "password";
  }
}

