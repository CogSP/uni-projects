<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ranking</title>
    <link rel="stylesheet" href="ranking.css">
    <link rel="stylesheet" href="background.css">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
</head>
<div class="bg"></div>
<div class="bg bg2"></div>
<div class="bg bg3"></div>
<body>

<div class="hero">
<nav> 
    <ul>
        <li><a href="index.php">Home</a></li>
        <li><a href="game.php">Fight!</a></li>
        <li><a href="profile.php">Profile</a></li>
        <li><a href="rules.php">Rules</a></li>
        <li><a href="settings.php">Settings</a></li>
        <li><a href="ranking.php">Ranking</a></li>
    </ul>
</nav>
</div>


    <div class = 'container mt-3'>
    <h1 class = "rank">RANKING</h1>
    <table class = 'table table-dark table-striped'>
        <tr class = 'table-warning'><td>Posizione</td><td>Username</td><td>Vittorie</td></tr>
        <?php
            $dbconnection = pg_connect("host = localhost dbname = dama user = postgres password = kub3tt0SQL") or die('Could not connect');
            $query = 
            "
            SELECT *
            FROM utente
            order by vittorie desc
            ";
            $result = pg_query($dbconnection, $query) or die('la query non va');
            $index = 1;
            while($row = pg_fetch_assoc($result)){
                echo 
                "
                    <tr><td>$index</td><td>{$row['username']}</td><td>{$row['vittorie']}</td></tr>
                ";
                $index = $index + 1;
            }
            pg_free_result($result);
            pg_close($dbconnection);
        ?>
    </table>
    </div>
</body>
</html>