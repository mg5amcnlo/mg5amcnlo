<html>
<head>
<title>Automated Variations of Shower Parameters</title>
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

<form method='post' action='Variations.php'>
 
<h2>Automated Variations of Shower Parameters for Uncertainty Bands</h2> 
 
While a number of different central "tunes" of the Pythia parameters 
are provided, it is often desired  to study how event properties change when 
some of the parameters (such as those describing the parton showers) are 
varied.   Pythia8 now has the ability to provide a series of weights 
to reflect the change in probability for a particular final state to occur 
when a subset of parton-shower parameters are varied.  Details on the 
implementation and interpretation of these weights can be found in 
[<a href="Bibliography.php" target="page">Mre16</a>]. 
Currently, the list of available automated variations 
(see <a href="#keywords">full list below</a>) includes: 
<ul> 
<li> The renormalization scale for QCD emissions in FSR; </li> 
<li> The renormalization scale for QCD emissions in ISR; </li> 
<li> The inclusion of non-singular terms in QCD emissions in FSR; </li> 
<li> The inclusion of non-singular terms in QCD emissions in ISR. </li> 
</ul> 
Similar variations would be possible for QED emissions, but these have not 
yet been implemented. 
 
<p/> 
Since the computation of the uncertainty variations takes additional 
CPU time (albeit much less than would be required for independent runs 
with the equivalent variations), the automated uncertainty variations 
are switched off by default. 
<br/><br/><strong>UncertaintyBands:doVariations</strong>  <input type="radio" name="1" value="on"><strong>On</strong>
<input type="radio" name="1" value="off" checked="checked"><strong>Off</strong>
 &nbsp;&nbsp;(<code>default = <strong>off</strong></code>)<br/>
Master switch to perform variations. 
   
 
<p/> 
The main intended purpose of these variations is to estimate 
perturbative uncertainties associated with the parton showers. Due to 
the pole at LambdaQCD, however, branchings near the perturbative 
cutoff can nominally result in very large reweighting factors, which 
is unwanted for typical applications. We therefore enable to limit the 
absolute (plus/minus) magnitude by which alphaS is allowed to vary by 
<br/><br/><table><tr><td><strong>UncertaintyBands:deltaAlphaSmax </td><td></td><td> <input type="text" name="2" value="0.2" size="20"/>  &nbsp;&nbsp;(<code>default = <strong>0.2</strong></code>; <code>minimum = 0.0</code>; <code>maximum = 1.0</code>)</td></tr></table>
 The allowed range of variation of alphaS, interpreted as abs(alphaSprime 
 - alphaS) &lt; deltaAlphaSmax. 
   
 
<p/> 
Likewise, non-singular-term variations are mainly intended to 
capture uncertainties related to missing higher-order tree-level 
matrix elements and are hence normally uninteresting for very soft 
branchings. The following parameter allows to switch off the 
variations of non-singular terms below a fixed perturbative threshold: 
<br/><br/><table><tr><td><strong>UncertaintyBands:cNSpTmin </td><td></td><td> <input type="text" name="3" value="5.0" size="20"/>  &nbsp;&nbsp;(<code>default = <strong>5.0</strong></code>; <code>minimum = 0.0</code>; <code>maximum = 20.0</code>)</td></tr></table>
Variations of non-singular terms will not be performed for branchings 
occurring below this threshold. 
   
 
<p/> 
By default, the automated shower uncertainty variations are enabled 
for the showers off the hardest interaction (and associated 
resonance decays), but not for the showers off MPI systems 
which would be more properly labeled as underlying-event uncertainties. 
If desired, the variations can be applied also to showers off MPI systems 
via the following switch: 
<br/><br/><strong>UncertaintyBands:MPIshowers</strong>  <input type="radio" name="4" value="on"><strong>On</strong>
<input type="radio" name="4" value="off" checked="checked"><strong>Off</strong>
 &nbsp;&nbsp;(<code>default = <strong>off</strong></code>)<br/>
Flag specifying whether the automated shower variations include 
showers off MPI systems or not. Note that substantially larger 
weight fluctuations must be expected when including shower 
variations for MPI, due to the (many) more systems which then 
enter in the reweightings. 
   
 
<p/> 
<b>UserHooks Warning:</b> the calculation of uncertainty variations 
will only be consistent in the absence of any external modifications 
to the shower branching probabilities via the 
<?php $filepath = $_GET["filepath"];
echo "<a href='UserHooks.php?filepath=".$filepath."' target='page'>";?>UserHooks</a> framework. It is therefore 
strongly advised to avoid combining the automated uncertainty 
calculations with any such UserHooks modifications. 
 
<p/> 
<b>Merging Warning:</b> in multi-jet merging approaches, trial showers 
are used to generate missing Sudakov factor corrections to the hard 
matrix elements. Currently that framework is not consistently combined 
with the variations introduced here, so the two should not be used 
simultaneously. This restriction will be lifted in a future release. 
 
<h3>Specifying the Variations</h3> 
 
When <code>UncertaintyBands:doVariations</code> is switched on, the user 
can define an arbitrary number of (combinations of) uncertainty variations 
to perform. Each variation is defined by a string with the following 
generic format: 
<pre> 
    Label keyword1=value keyword2=value ... 
</pre> 
where the user has complete freedom to specify the label, and each 
keyword must be selected from the 
<a href="#keywords">list of currently recognised keywords</a> below. 
Instead of an equal sign it is also possible to leave a blank between 
a keyword and its value. 
 
<p/> 
To exemplify, an uncertainty variation corresponding to simultaneously 
increasing both the ISR and FSR renormalisation scales by a factor of 
two would be defined as follows 
<pre> 
    myVariation1 fsr:muRfac=2.0 isr:muRfac=2.0 
</pre> 
 
<p/> 
Staying within the context of this example, the user might also want to 
check what a variation of the two scales independently of each other would 
produce. This can be achieved within the same run by adding two further 
variations, as follows: 
<pre> 
    myVariation2 fsr:muRfac=2.0 
    myVariation3 isr:muRfac=2.0 
</pre> 
Different histograms can then be filled with each set of weights as 
desired (see <a href="#access">accessing the uncertainty weights</a> below). 
Variations by smaller or larger factors can obviously also be added in the 
same way, again within one and the same run. 
 
<p/> 
Once a list of variations defined as above has been decided on, 
the whole list should be passed to Pythia in the form of a single 
<?php $filepath = $_GET["filepath"];
echo "<a href='SettingsScheme.php?filepath=".$filepath."' target='page'>";?>"vector of strings"</a>, defined as 
follows: 
<p/><code>wvec&nbsp; </code><strong> UncertaintyBands:List &nbsp;</strong> 
 (<code>default = <strong>{alphaShi fsr:muRfac=0.5 isr:muRfac=0.5, alphaSlo fsr:muRfac=2.0 isr:muRfac=2.0, hardHi fsr:cNS=2.0 isr:cNS=2.0, hardLo fsr:cNS=-2.0 isr:cNS=-2.0}</strong></code>)<br/>
Vector of uncertainty-variation strings defining which variations will be 
calculated by Pythia when<code>UncertaintyBands:doVariations</code> 
is switched on. 
   
 
<p/> 
For completeness, we note that a command-file specification 
equivalent to the above default variations could look as follows: 
<pre> 
    UncertaintyBands:List = { 
        alphaShi fsr:muRfac=0.5 isr:muRfac=0.5, 
        alphaSlo fsr:muRfac=2.0 isr:muRfac=2.0, 
        hardHi fsr:cNS=2.0 isr:cNS=2.0, 
        hardLo fsr:cNS=-2.0 isr:cNS=-2.0 
    } 
</pre> 
Note that each of the individual uncertainty-variation definitions 
(the elements of the vector) are separated by commas and that 
keywords separated only by spaces are interpreted as belonging to a 
single combined variation. Note also that the beginning and end of the 
vector is marked by curly braces. 
 
<a name="access"></a> 
<h3>Accessing the Uncertainty Weights</h3> 
 
During the event generation, uncertainty weights will be calculated 
for each variation defined above, via the method described in 
[<a href="Bibliography.php" target="page">Mre16</a>]. The resulting alternative weights for the event are 
accessible through the <code>Pythia::info.weight(int iWeight=0)</code> 
method. 
 
<p/> 
The baseline weight for each event (normally unity for an 
ordinary unweighted event sample) is not modified and 
corresponds to <code>iWeight = 0</code>. The uncertainty-variation 
weights are thus enumerated starting from <code>iWeight = 1</code> for 
the first variation up to <code>N</code> for the last variation, in 
the order they were specified in <code>UncertaintyBands:List</code>. 
 
<p/> 
The total number of variations that have been defined, <code>N</code>, 
can be queried using <code>Pythia::info.nWeights()</code>. 
 
<h3>NLO Compensation Term for Renormalisation-Scale Variations</h3> 
 
Additionally, there is a run-time parameter: 
<br/><br/><strong>UncertaintyBands:muSoftCorr</strong>  <input type="radio" name="5" value="on" checked="checked"><strong>On</strong>
<input type="radio" name="5" value="off"><strong>Off</strong>
 &nbsp;&nbsp;(<code>default = <strong>on</strong></code>)<br/>
This flags tells the shower to apply an O(&alpha;S<sup>2</sup>) 
compensation term to the renormalization-scale variations, which 
reduces their magnitude for soft emissions, as described in 
[<a href="Bibliography.php" target="page">Mre16</a>]. 
   
 
<a name="keywords"></a> 
<h3>List of Recognised Keywords for Uncertainty Variations</h3> 
 
The following keywords adjust the renormalisation scales and 
non-singular terms for all FSR and ISR branchings, respectively: 
<ul> 
<li><code>fsr:muRfac</code> : multiplicative factor applied to the 
renormalization scale for FSR branchings.</li> 
<li><code>isr:muRfac</code> : multiplicative factor applied to the 
renormalization scale for ISR branchings.</li> 
<li><code>fsr:cNS</code> : additive non-singular ("finite") 
term in the FSR splitting functions.</li> 
<li><code>isr:cNS</code> : additive non-singular ("finite") 
term in the ISR splitting functions.</li> 
</ul> 
Note that the <code>muRfac</code> parameters are applied linearly to 
the renormalisation scale, hence &mu;<sup>2</sup> &rarr; 
(<code>muRfac</code>)<sup>2</sup>*&mu;<sup>2</sup>. 
 
<p/> 
Optionally, a further level of detail can be accessed by specifying 
variations for specific types of branchings, with the global keywords 
above corresponding to setting the same value for all 
branchings. Using the <code>fsr:muRfac</code> parameter for 
illustration, the individual branching types that can be specified 
are: 
<ul> 
<li><code>fsr:G2GG:muRfac</code> : variation for g&rarr;gg branchings.</li> 
<li><code>fsr:Q2QG:muRfac</code> : variation for q&rarr;qg branchings.</li> 
<li><code>fsr:G2QQ:muRfac</code> : variation for g&rarr;qqbar branchings.</li> 
<li><code>fsr:X2XG:muRfac</code> : variation for gluon bremsstrahlung off 
other types of particles (such as coloured new-physics particles). </li> 
</ul> 
For the distinction between <code>Q2QG</code> and <code>X2XG</code>, 
the following switch can be used to control whether <i>b</i> and 
<i>t</i> quarks are considered to be <code>Q</code> or <code>X</code> 
particles (e.g. providing a simple way to control top-quark or bottom-quark 
radiation independently of the rest of the shower uncertainties): 
<p/><code>mode&nbsp; </code><strong> UncertaintyBands:nFlavQ &nbsp;</strong> 
 (<code>default = <strong>6</strong></code>; <code>minimum = 2</code>; <code>maximum = 6</code>)<br/>
Number of quark flavours controlled via <code>Q2QG</code> keywords, with 
higher ID codes controlled by <code>X2XG</code> keywords. Thus a change to 
5 would mean that top-quark variations would use <code>X2XG</code> keyword 
values instead of the corresponding <code>Q2QG</code> ones. 
   
 
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
$data = "UncertaintyBands:doVariations = ".$_POST["1"]."\n";
fwrite($handle,$data);
}
if($_POST["2"] != "0.2")
{
$data = "UncertaintyBands:deltaAlphaSmax = ".$_POST["2"]."\n";
fwrite($handle,$data);
}
if($_POST["3"] != "5.0")
{
$data = "UncertaintyBands:cNSpTmin = ".$_POST["3"]."\n";
fwrite($handle,$data);
}
if($_POST["4"] != "off")
{
$data = "UncertaintyBands:MPIshowers = ".$_POST["4"]."\n";
fwrite($handle,$data);
}
if($_POST["5"] != "on")
{
$data = "UncertaintyBands:muSoftCorr = ".$_POST["5"]."\n";
fwrite($handle,$data);
}
fclose($handle);
}

?>
</body>
</html>
 
<!-- Copyright (C) 2017 Torbjorn Sjostrand --> 
