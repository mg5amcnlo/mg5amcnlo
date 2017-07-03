<html>
<head>
<title>Hadron Scattering</title>
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

<form method='post' action='HadronScattering.php'>
 
<h2>Hadron Scattering</h2> 
 
This page describes a few simple hadron (re)scattering models. 
They are intended to take into account 
that the overlap of multiple strings at low transverse dimensions 
is likely to lead to some collective effects, not unlike those 
observed in heavy-ion collisions, even if not quite as pronounced. 
Specifically, it is assumed that the hadrons produced can scatter 
against each other on the way out, before the fragmenting system 
has had time to expand enough that the hadrons get free. Thereby 
heavier particles are shifted to higher transverse momenta, at the 
expense of the lighter ones. 
 
<p/> 
The main switch on/off switch for rescattering is 
<code>HadronLevel:HadronScatter</code>, which by the default is off, 
since all models are rather simplistic and have to be used 
with some caution. Currently there are three different options available: 
 
<br/><br/><table><tr><td><strong>HadronScatter:mode  </td><td>  &nbsp;&nbsp;(<code>default = <strong>0</strong></code>; <code>minimum = 0</code>; <code>maximum = 2</code>)</td></tr></table>
The first two options are variations of the same model, whereas 
option 2 respresents a different model. 
<br/>
<input type="radio" name="1" value="0" checked="checked"><strong>0 </strong>:  The new model, based on separation in rapidity  as described in <ref>Fis16</ref>.  Further options are found <a href="#HadScatNew1">here</a>.  <br/>
<input type="radio" name="1" value="1"><strong>1 </strong>:  The new model, based on separation in rapidity  and azimuthal angle as described in <ref>Fis16</ref>. Further options  are found  <a href="#HadScatNew2">here</a>.  <br/>
<input type="radio" name="1" value="2"><strong>2 </strong>:  The old model. Further options are found  <a href="#HadScatOld">here</a>.  <note>Warning:</note> Option 2 is still at an experimental level,  and should not be used unless you know what you are doing.  <br/>
 
<h3>The New Model for Hadron Scattering</h3> 
 
Within the new modle, there are two options available for how hadron 
pairs are found: 
 
<a name="HadScatNew1"></a> 
<h4>Rapidity based</h4> 
This corresponds to <code>HadronScatter:mode = 0</code>. 
<p/> 
Probe all hadron pairs with an invariant mass <i> m<sub>inv</sub> &lt; 
(m<sup>2</sup><sub>1</sub>+p<sup>2</sup><sub>Max</sub>)<sup>1/2</sup> + 
(m<sup>2</sup><sub>2</sub>+p<sup>2</sup><sub>Max</sub>)<sup>1/2</sup></i> 
with the parameter <i>p<sub>Max</sub></i> 
<br/><br/><table><tr><td><strong>HadronScatter:pMax </td><td></td><td> <input type="text" name="2" value="0.5" size="20"/>  &nbsp;&nbsp;(<code>default = <strong>0.5</strong></code>; <code>minimum = 0.1</code>; <code>maximum = 1000000.0</code>)</td></tr></table>
   
<br/> 
If a hadron pair passes this cut, the scattering probability for hadrons of 
different strings is <i>P<sub>DS</sub>(&#x394y) = 
P<sup>max</sup><sub>DS</sub>(1 - &#x394y/&#x394y<sup>max</sup>)</i> 
with rapidity difference <i>&#x394y</i> of the hadron pair and the 
parameters <i>&#x394y<sup>max</sup></i> 
<br/><br/><table><tr><td><strong>HadronScatter:yDiffMax </td><td></td><td> <input type="text" name="3" value="1.0" size="20"/>  &nbsp;&nbsp;(<code>default = <strong>1.0</strong></code>; <code>minimum = 0.1</code>; <code>maximum = 10.0</code>)</td></tr></table>
   
<br/> 
and <i>P<sup>max</sup><sub>DS</sub></i>, see 
<a href="#HadScatNewCom">below</a>. If the hadrons are produced 
within the same string the probability is <i>P<sub>DS</sub>(&#x394y) 
P<sup>max</sup><sub>SS</sub></i> if the hadrons are further apart from each 
other as <code>HadronScatter:neighbourFar</code>, <i>0</i> if they are 
closer together as <code>HadronScatter:neighbourNear</code>, and linear 
between the maximum <code>HadronScatter:maxProbSS</code> and minimum 
probability <code>HadronScatter:minProbSS</code> inbetween. 
 
<a name="HadScatNew2"></a> 
<h4>Rapidity and Azimuth based</h4> 
This corresponds to <code>HadronScatter:mode = 1</code>. 
<p/> 
All hadron pairs are considered. The scattering probability for hadrons 
of different strings is <i>P<sub>DS</sub>(&#x394y,&#x394&#x3C6) = 
P<sup>max</sup><sub>DS</sub>(1 - ((&#x394y)<sup>2</sup> 
+(&#x394&#x3C6)<sup>2</sup>)<sup>1/2</sup>/R<sup>max</sup>)</i> 
with rapidity difference <i>&#x394y</i> and difference in azimuth 
<i>&#x394&#x3C6</i> of the hadron pair and the 
parameters <i>R<sup>max</sup></i> 
<br/><br/><table><tr><td><strong>HadronScatter:Rmax </td><td></td><td> <input type="text" name="4" value="1.0" size="20"/>  &nbsp;&nbsp;(<code>default = <strong>1.0</strong></code>; <code>minimum = 0.1</code>; <code>maximum = 10.0</code>)</td></tr></table>
   
<br/> 
and <i>P<sup>max</sup><sub>DS</sub></i>, see 
<a href="#HadScatNewCom">below</a>. 
The probability for hadron pairs from the same string is similar 
to the one before. 
 
<a name="HadScatNewCom"></a> 
<h4>Common Parameters</h4> 
 
The following paramters are used for both the above cases: 
<br/><br/><strong>HadronScatter:scatterSameString</strong>  <input type="radio" name="5" value="on" checked="checked"><strong>On</strong>
<input type="radio" name="5" value="off"><strong>Off</strong>
 &nbsp;&nbsp;(<code>default = <strong>on</strong></code>)<br/>
  If switched on, strings within the same string are allowed to 
  scatter off each other. Otherwise only hadron pairs that are 
  not produced on the same string are taken into account. 
   
 
<br/><br/><strong>HadronScatter:scatterMultipleTimes</strong>  <input type="radio" name="6" value="on" checked="checked"><strong>On</strong>
<input type="radio" name="6" value="off"><strong>Off</strong>
 &nbsp;&nbsp;(<code>default = <strong>on</strong></code>)<br/>
  If switched off, each hadron is only allowed to scatter at most once. 
  By the way that possible scattering pairs are considered in order of 
  increasing rapidity separation, this introduces a bias towards pairs 
  with small <i>y</i> separation. 
   
 
<br/><br/><table><tr><td><strong>HadronScatter:maxProbDS </td><td></td><td> <input type="text" name="7" value="0.5" size="20"/>  &nbsp;&nbsp;(<code>default = <strong>0.5</strong></code>; <code>minimum = 0.0</code>; <code>maximum = 1.0</code>)</td></tr></table>
  The maximum probability <i>P<sup>max</sup><sub>DS</sub></i> for the 
  scattering of two hadrons that are not part of the same string. 
   
 
<p/><code>mode&nbsp; </code><strong> HadronScatter:neighbourNear &nbsp;</strong> 
 (<code>default = <strong>2</strong></code>; <code>minimum = 1</code>; <code>maximum = 10</code>)<br/>
  If scattering of hadrons within the same string is allowed this 
  parameter gives the closest neighbour that is allowed. The value 1 
  corresponds to the direct neighbour. The probability associated 
  with this potential scattering partner is <code>minProbSS</code>. 
   
 
<p/><code>mode&nbsp; </code><strong> HadronScatter:neighbourFar &nbsp;</strong> 
 (<code>default = <strong>4</strong></code>; <code>minimum = 2</code>; <code>maximum = 15</code>)<br/>
  If scattering of hadrons within the same string is allowed this 
  parameter gives the neighbour starting from which the maximum 
  probability <code>maxProbSS</code> is applied. 
   
 
<br/><br/><table><tr><td><strong>HadronScatter:minProbSS </td><td></td><td> <input type="text" name="8" value="0.5" size="20"/>  &nbsp;&nbsp;(<code>default = <strong>0.5</strong></code>; <code>minimum = 0.0</code>; <code>maximum = 1.0</code>)</td></tr></table>
  The minimum probability <i>P<sup>min</sup><sub>SS</sub></i> for the 
  scattering of two hadrons within the same string. (Relative to that for 
  different strings, i.e. for the total probability the baseline 
  <code>maxProbDS</code> factor also enters.) 
   
 
<br/><br/><table><tr><td><strong>HadronScatter:maxProbSS </td><td></td><td> <input type="text" name="9" value="1.0" size="20"/>  &nbsp;&nbsp;(<code>default = <strong>1.0</strong></code>; <code>minimum = 0.0</code>; <code>maximum = 1.0</code>)</td></tr></table>
  The maximum probability <i>P<sup>max</sup><sub>SS</sub></i> for the 
  scattering of two hadrons within the same string. (Relative to that for 
  different strings, i.e. for the total probability the baseline 
  <code>maxProbDS</code> factor also enters.) 
   
 
<a name="HadScatOld"></a> 
<h3>The Old Model for Hadron Scattering</h3> 
 
<br/><b>Warning:</b> This is still at an experimental level, 
and should not be used unless you know what you are doing. 
 
<br/><br/><strong>HadronScatter:afterDecay</strong>  <input type="radio" name="10" value="on"><strong>On</strong>
<input type="radio" name="10" value="off" checked="checked"><strong>Off</strong>
 &nbsp;&nbsp;(<code>default = <strong>off</strong></code>)<br/>
Perform hadron scattering before or after first round of decays, 
involving very short-lived particles like the <i>rho</i>. 
The default is to perform scattering directly after the 
string fragmentation, before any decays. 
   
 
<br/><br/><strong>HadronScatter:allowDecayProd</strong>  <input type="radio" name="11" value="on"><strong>On</strong>
<input type="radio" name="11" value="off" checked="checked"><strong>Off</strong>
 &nbsp;&nbsp;(<code>default = <strong>off</strong></code>)<br/>
Allow two hadrons with same parent hadron to scatter. 
   
 
<br/><br/><strong>HadronScatter:scatterRepeat</strong>  <input type="radio" name="12" value="on"><strong>On</strong>
<input type="radio" name="12" value="off" checked="checked"><strong>Off</strong>
 &nbsp;&nbsp;(<code>default = <strong>off</strong></code>)<br/>
Allow hadrons which have already scattered to scatter again. 
Even if switched on, the same pair can not scatter off each 
other twice. 
   
 
<h4>Hadron selection</h4> 
 
<br/><br/><table><tr><td><strong>HadronScatter:hadronSelect  </td><td>  &nbsp;&nbsp;(<code>default = <strong>0</strong></code>; <code>minimum = 0</code>; <code>maximum = 0</code>)</td></tr></table>
Probability that a hadron is soft enough to scatter. 
(A high-<ei>pT</ei> hadron presumably being part of a jet, 
and thus produced away from the high-particle-density region 
at small transverse dimensions.) 
<br/>
<input type="radio" name="13" value="0" checked="checked"><strong>0 </strong>:   <ei>P = N exp(-pT^2 / 2 / sigma^2) /    ( (1 - k) exp(-pT^2 / 2 / sigma^2) + k pT0^p / (pT0^2 + pT^2)^(p/2), </ei>  with <ei>sigma = 2 StringPT:sigma</ei> and <ei>pT0</ei> the same as that  used in <ei>MultipartonInteractions</ei>.  <br/>
 
<br/><br/><table><tr><td><strong>HadronScatter:N </td><td></td><td> <input type="text" name="14" value="1.0" size="20"/>  &nbsp;&nbsp;(<code>default = <strong>1.0</strong></code>; <code>minimum = 0.01</code>; <code>maximum = 1.0</code>)</td></tr></table>
<i>N</i> parameter as above. 
   
<br/><br/><table><tr><td><strong>HadronScatter:k </td><td></td><td> <input type="text" name="15" value="1.0" size="20"/>  &nbsp;&nbsp;(<code>default = <strong>1.0</strong></code>; <code>minimum = 0.01</code>; <code>maximum = 1.0</code>)</td></tr></table>
<i>k</i> parameter as above. 
   
<br/><br/><table><tr><td><strong>HadronScatter:p </td><td></td><td> <input type="text" name="16" value="6" size="20"/>  &nbsp;&nbsp;(<code>default = <strong>6</strong></code>; <code>minimum = 2</code>; <code>maximum = 30</code>)</td></tr></table>
<i>p</i> parameter as above. 
   
 
<h4>Scattering probability</h4> 
 
<br/><br/><table><tr><td><strong>HadronScatter:scatterProb  </td><td>  &nbsp;&nbsp;(<code>default = <strong>0</strong></code>; <code>minimum = 0</code>; <code>maximum = 2</code>)</td></tr></table>
Probability for a pair of hadrons to scatter. 
<br/>
<input type="radio" name="17" value="0" checked="checked"><strong>0 </strong>: All hadrons scatter with probability  <ei>j max(0, 1 - dR^2 / rMax^2)</ei>. Angular distribution  is picked flat in <ei>cos(theta).</ei><br/>
<input type="radio" name="17" value="1"><strong>1 </strong>: As option 0, above, but only <ei>pi-pi</ei>,  <ei>pi-K</ei> and <ei>pi-p</ei> scatterings are considered.  <br/>
<input type="radio" name="17" value="2"><strong>2 </strong>: Only <ei>pi-pi</ei>, <ei>pi-K</ei> and  <ei>pi-p</ei> scatterings are considered, with probability  given by <ei>(1 - exp(-j sigEl)) max(0, 1 - dR^2 / rMax^2)</ei>.  The elastic cross sections and angular distributions are taken  from the partial-wave distributions.  <br/>
 
<br/><br/><table><tr><td><strong>HadronScatter:j </td><td></td><td> <input type="text" name="18" value="0.5" size="20"/>  &nbsp;&nbsp;(<code>default = <strong>0.5</strong></code>; <code>minimum = 0.0</code>; <code>maximum = 10.0</code>)</td></tr></table>
<i>j</i> parameter as above. 
   
<br/><br/><table><tr><td><strong>HadronScatter:rMax </td><td></td><td> <input type="text" name="19" value="1.0" size="20"/>  &nbsp;&nbsp;(<code>default = <strong>1.0</strong></code>; <code>minimum = 0.1</code>; <code>maximum = 2.0</code>)</td></tr></table>
<i>rMax</i> parameter as above. 
   
 
<br/><br/><strong>HadronScatter:tile</strong>  <input type="radio" name="20" value="on"><strong>On</strong>
<input type="radio" name="20" value="off" checked="checked"><strong>Off</strong>
 &nbsp;&nbsp;(<code>default = <strong>off</strong></code>)<br/>
Use tiling in <i>(eta, phi)</i> to reduce number of pairwise tests. 
   
 
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

if($_POST["1"] != "0")
{
$data = "HadronScatter:mode = ".$_POST["1"]."\n";
fwrite($handle,$data);
}
if($_POST["2"] != "0.5")
{
$data = "HadronScatter:pMax = ".$_POST["2"]."\n";
fwrite($handle,$data);
}
if($_POST["3"] != "1.0")
{
$data = "HadronScatter:yDiffMax = ".$_POST["3"]."\n";
fwrite($handle,$data);
}
if($_POST["4"] != "1.0")
{
$data = "HadronScatter:Rmax = ".$_POST["4"]."\n";
fwrite($handle,$data);
}
if($_POST["5"] != "on")
{
$data = "HadronScatter:scatterSameString = ".$_POST["5"]."\n";
fwrite($handle,$data);
}
if($_POST["6"] != "on")
{
$data = "HadronScatter:scatterMultipleTimes = ".$_POST["6"]."\n";
fwrite($handle,$data);
}
if($_POST["7"] != "0.5")
{
$data = "HadronScatter:maxProbDS = ".$_POST["7"]."\n";
fwrite($handle,$data);
}
if($_POST["8"] != "0.5")
{
$data = "HadronScatter:minProbSS = ".$_POST["8"]."\n";
fwrite($handle,$data);
}
if($_POST["9"] != "1.0")
{
$data = "HadronScatter:maxProbSS = ".$_POST["9"]."\n";
fwrite($handle,$data);
}
if($_POST["10"] != "off")
{
$data = "HadronScatter:afterDecay = ".$_POST["10"]."\n";
fwrite($handle,$data);
}
if($_POST["11"] != "off")
{
$data = "HadronScatter:allowDecayProd = ".$_POST["11"]."\n";
fwrite($handle,$data);
}
if($_POST["12"] != "off")
{
$data = "HadronScatter:scatterRepeat = ".$_POST["12"]."\n";
fwrite($handle,$data);
}
if($_POST["13"] != "0")
{
$data = "HadronScatter:hadronSelect = ".$_POST["13"]."\n";
fwrite($handle,$data);
}
if($_POST["14"] != "1.0")
{
$data = "HadronScatter:N = ".$_POST["14"]."\n";
fwrite($handle,$data);
}
if($_POST["15"] != "1.0")
{
$data = "HadronScatter:k = ".$_POST["15"]."\n";
fwrite($handle,$data);
}
if($_POST["16"] != "6")
{
$data = "HadronScatter:p = ".$_POST["16"]."\n";
fwrite($handle,$data);
}
if($_POST["17"] != "0")
{
$data = "HadronScatter:scatterProb = ".$_POST["17"]."\n";
fwrite($handle,$data);
}
if($_POST["18"] != "0.5")
{
$data = "HadronScatter:j = ".$_POST["18"]."\n";
fwrite($handle,$data);
}
if($_POST["19"] != "1.0")
{
$data = "HadronScatter:rMax = ".$_POST["19"]."\n";
fwrite($handle,$data);
}
if($_POST["20"] != "off")
{
$data = "HadronScatter:tile = ".$_POST["20"]."\n";
fwrite($handle,$data);
}
fclose($handle);
}

?>
</body>
</html>
 
<!-- Copyright (C) 2017 Torbjorn Sjostrand --> 
