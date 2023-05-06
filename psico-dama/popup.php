<?php

$usr = $_REQUEST["usr"];
$pwd = $_REQUEST["pwd"];


$dbconnection = pg_connect("host = localhost dbname = dama user = postgres password = kub3tt0SQL") or die('Could not connect');

$query = 
" 
    SELECT username
    FROM utente
    WHERE (utente.password = '$pwd' and utente.username = '$usr')
";

$result = pg_query($dbconnection, $query) or die('la query non va');
$array = pg_fetch_all($result);

if(count($array)>=1) {echo 'si';}
else echo 'no';



pg_free_result($result);

pg_close($dbconnection);

?>