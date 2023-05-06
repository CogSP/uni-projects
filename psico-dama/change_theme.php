<?php
session_start();

$theme = $_REQUEST["theme"];

$_SESSION["theme"] = $theme;

echo $_SESSION["theme"];

?>