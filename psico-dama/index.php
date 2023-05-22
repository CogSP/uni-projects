<?php #sfruttiamo una variabile di sessione per memorizzare il tema scelto dall'utente
session_start();
if(!isset($_SESSION["theme"])){
$_SESSION["theme"] = "std";
}
if(!isset($_SESSION["mode"])){
$_SESSION["mode"] = "std";
}
?>

<!DOCTYPE html>

<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0"> 
    <link rel="stylesheet" href="style.css"> <!-- this will link the css file-->
    <link rel="stylesheet" href="background.css">

    <title>PSICODAMA</title>
</head>
<div class="bg"></div>
<div class="bg bg2"></div>
<div class="bg bg3"></div>
<body>

    <div class="hero">

        <nav> 
            <ul
            >
                <li><a href="index.php">Home</a></li>
                <li><a href="game.php">Fight!</a></li>
                <li><a href="profile.php">Profile</a></li>
                <li><a href="rules.php">Rules</a></li>
                <li><a href="settings.php">Settings</a></li>
                <li><a href="ranking.php">Ranking</a></li>
            </ul>
        </nav>
        
        <div class="content">
            <h1>PSICODAMA</h1>
            <a href="game.php">Join the fight!</a>
        </div>

    </div>

</body>

</html>
