<?php 
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
<!--     
    
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
     -->
    <title>PSICODAMA</title>
</head>
<div class="bg"></div>
<div class="bg bg2"></div>
<div class="bg bg3"></div>
<body>

    <div class="hero">

        <!-- <video autoplay loop muted plays-inline class="back-video">
            <source src="video.mp4" type="video/mp4">
        </video> -->

        <nav> <!-- nav is a section that contains navigations link (to the same page or other) -->
            <ul>
                <li><a href="index.php">Home</a></li>
                <li><a href="game.php">Fight!</a></li>
                <li><a href="profile.php">Profile</a></li>
                <li><a href="rules.php">Rules</a></li>
                <!-- https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjcvIPex4v-AhXJSvEDHcrPCaUQFnoECA0QAw&url=https%3A%2F%2Fwww.usatoday.com%2Fstory%2Fgraphics%2F2023%2F01%2F23%2Fhow-to-play-checkers-rules-strategy%2F10795787002%2F&usg=AOvVaw0_Pa61Itzk_Kwhg77W4q26 -->
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
