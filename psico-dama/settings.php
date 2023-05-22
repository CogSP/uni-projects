<?php 
session_start();
?>

<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Settings</title>
    <link rel="stylesheet" href="background.css">
    <link rel="stylesheet" href="profile.css">
    <script defer src = "settings.js"></script>
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.6.4/jquery.min.js" ></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js" ></script>
</head>
<!-- 
Settings:
 Tema sfondo/gioco (3 opzioni)
 Modalità alternativa
 Elimina un profilo
 Cambia password
 
 -->
<div class="bg"></div>
<div class="bg bg2"></div>
<div class="bg bg3"></div>
<body>
<div class="hero">
<nav> 
    <ul style="margin-top: 1.7%;">
        <li><a href="index.php">Home</a></li>
        <li><a href="game.php">Fight!</a></li>
        <li><a href="profile.php">Profile</a></li>
        <li><a href="rules.php">Rules</a></li>
        <li><a href="settings.php">Settings</a></li>
        <li><a href="ranking.php">Ranking</a></li>
    </ul>
</nav>
</div>

<div class = "the_form">


  <div class="container">
    <div class="title">Settings</div>
    <div class="content">
      <form action="" method="">
        <div class="user-details">

<div class="input-box"> 
    <div class="row">
        <div class="col">
            <div class="dropdown">
                <button class="btn btn-secondary dropdown-toggle" type="button" id="dropdownMenuButton2" data-bs-toggle="dropdown" aria-expanded="false" style = "background-color:rgb(240,240,240); color:black;">
                    Change Theme
                </button>
                <ul class="dropdown-menu dropdown-menu-dark" aria-labelledby="dropdownMenuButton2" style = "background-color: rgb(240,240,240);">
                    <li id="li1"><a id = "ct-std" class="dropdown-item active"  style = "color:black;" >Standard</a></li>
                    <li id="li2"><a id="ct-dark" class="dropdown-item"  style = "color:black;" >Dark</a></li>
                    <li id="li3"><a id= "ct-trop" class="dropdown-item"  style = "color:black;">Tropical</a></li>
                </ul>
            </div>
        </div>
    </div>
</div>

          <div class="input-box">
            <a href="cambiapwd.php">
            <input id = "cambiapwd" name = "cambiapwd" type= "button" value = "Cambia password" >
            </a>
          </div> 
          
          <div class="input-box">
            <a href = "elimina.php">
            <input id = "elimina" name = "elimina" type="button" value = "Elimina account" onclick = "" >
            </a>
          </div>
          
</div>
        
      </form>
    </div>
  </div>

</div>

    
</body>
</html>