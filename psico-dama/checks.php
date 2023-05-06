<?php 
session_start();

?>

<?php
$q = $_REQUEST['q'];
$dbconnection = pg_connect("host = localhost dbname = dama user = postgres password = kub3tt0SQL") or die('Could not connect');

$query =
         "
            SELECT *
            FROM utente
            WHERE username = '$q'

         ";
         
$result = pg_query($dbconnection, $query) or die('la query non va');
$array = pg_fetch_all($result);
if(count($array) >=1){
    echo 'no';
}
else{
    echo 'si';
}

pg_free_result($result);
pg_close($dbconnection);



?>