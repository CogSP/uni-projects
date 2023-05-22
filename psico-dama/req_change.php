<?php 
session_start();

?>
<?php
$usr = $_REQUEST["usr"];
$oldpwd = $_REQUEST["oldpwd"];
$newpwd = $_REQUEST["newpwd"];

$dbconnection = pg_connect("host = localhost dbname = dama user = postgres password = kub3tt0SQL") or die('Could not connect');

$query = 
"
SELECT *
FROM utente
WHERE username = '$usr' and password = '$oldpwd'

";

$result = pg_query($dbconnection, $query) or die('la query non va');

$array = pg_fetch_all($result);

if(count($array)>= 1){
    $query2 = 
    "
    UPDATE utente
    SET password = '$newpwd'
    WHERE password = '$oldpwd'
    ";
    $result2 = pg_query($dbconnection, $query2) or die('la query non va');
    pg_free_result($result);
    pg_free_result($result2);
    pg_close($dbconnection);
    echo 'fatto';
}
else echo 'errore';

?>