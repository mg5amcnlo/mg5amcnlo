<html>
<head>
<title>Dark Matter Processes</title>
<link rel="stylesheet" type="text/css" href="pythia.css"/>
<link rel="shortcut icon" href="pythia32.gif"/>
</head>
<body>

<script language=javascript type=text/javascript>
function stopRKey(evt) {
var evt = (evt) ? evt : ((event) ? event : null);
var node = (evt.target) ? evt.target :((evt.srcElement) ? evt.srcElement : null);
if ((evt.keyCode == 13) && (node.type=="text"))
{return false;}
}

document.onkeypress = stopRKey;
</script>
<?php
if($_POST['saved'] == 1) {
if($_POST['filepath'] != "files/") {
echo "<font color='red'>SETTINGS SAVED TO FILE</font><br/><br/>"; }
else {
echo "<font color='red'>NO FILE SELECTED YET.. PLEASE DO SO </font><a href='SaveSettings.php'>HERE</a><br/><br/>"; }
}
?>

<form method='post' action='DarkMatterProcesses.php'>
 
<h2>Dark Matter Processes</h2> 
 
This page contains the production of Dark Matter via new <i>s</i>-channel 
mediators.  Currently only vector-like mediator i.e. <i>Z'^0</i> for 
Dirac DM is implemented. Note that the mediator in this case carries 
the PDG id 55. 
 
<h3><i>Z'^0</i></h3> 
 
This group currently contains only one subprocess. 
 
<br/><br/><strong>DM:ffbar2Zp2XX</strong>  <input type="radio" name="1" value="on"><strong>On</strong>
<input type="radio" name="1" value="off" checked="checked"><strong>Off</strong>
 &nbsp;&nbsp;(<code>default = <strong>off</strong></code>)<br/>
Scattering <i>f fbar &rarr;Z'^0 &rarr; X Xbar</i>. 
Code 6001. 
   
 
<p/> 
The couplings of the <i>Z'^0</i> to quarks and leptons are be 
assumed universal, i.e. generation-independent. Currently only vector 
couplings are implemented.  The choice of fixed axial and vector 
couplings implies a resonance width that increases linearly with the 
<i>Z'^0</i> mass. Note that cross sections strongly depend on the 
choice of mediator and Dark Matter masses. 
 
<p/> 
Here are the couplings 
 
<br/><br/><table><tr><td><strong>Zp:vq </td><td></td><td> <input type="text" name="2" value="0.1" size="20"/>  &nbsp;&nbsp;(<code>default = <strong>0.1</strong></code>)</td></tr></table>
vector coupling of quarks. 
   
 
<br/><br/><table><tr><td><strong>Zp:vX </td><td></td><td> <input type="text" name="3" value="0.1" size="20"/>  &nbsp;&nbsp;(<code>default = <strong>0.1</strong></code>)</td></tr></table>
vector coupling of the Dark Matter <i>X</i> particles. 
   
 
<input type="hidden" name="saved" value="1"/>

<?php
echo "<input type='hidden' name='filepath' value='".$_GET["filepath"]."'/>"?>

<table width="100%"><tr><td align="right"><input type="submit" value="Save Settings" /></td></tr></table>
</form>

<?php

if($_POST["saved"] == 1)
{
$filepath = $_POST["filepath"];
$handle = fopen($filepath, 'a');

if($_POST["1"] != "off")
{
$data = "DM:ffbar2Zp2XX = ".$_POST["1"]."\n";
fwrite($handle,$data);
}
if($_POST["2"] != "0.1")
{
$data = "Zp:vq = ".$_POST["2"]."\n";
fwrite($handle,$data);
}
if($_POST["3"] != "0.1")
{
$data = "Zp:vX = ".$_POST["3"]."\n";
fwrite($handle,$data);
}
fclose($handle);
}

?>
</body>
</html>
 
<!-- Copyright (C) 2017 Torbjorn Sjostrand --> 
