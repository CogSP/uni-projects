<?php 
session_start();

?>

<?php
$usr = $_REQUEST['q'];
$dbconnection = pg_connect("host = localhost dbname = dama user = postgres password = kub3tt0SQL") or die('Could not connect');
$query = "
    UPDATE utente
    SET vittorie = vittorie + 1
    WHERE username = '$usr'; 
";
//username è primary key quindi questa query funziona come deve
$result = pg_query($dbconnection, $query) or die('la query non va');
pg_free_result($result);
pg_close($dbconnection);



?>