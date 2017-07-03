<html>
<head>
<title>Photon-photon Interactions</title>
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

<form method='post' action='PhotonPhoton.php'>
 
<h2>Photon-photon Interactions</h2> 
 
<p> 
Interactions of two photons, either in photon-photon collision or between 
photons emitted from lepton beams. Includes both direct and resolved 
contributions and also soft QCD and MPIs for events with two resolved photons. 
Only (quasi-)real photons are considered so virtuality of the photons is 
restricted. The PDF set for resolved photons is selected in the 
<?php $filepath = $_GET["filepath"];
echo "<a href='PDFSelection.php?filepath=".$filepath."' target='page'>";?>PDF selection</a>. 
This page describes some of the special features related to these collisions 
and introduces the relevant parameters. 
</p> 
 
<h3>Types of photon-photon interactions</h3> 
 
<p> 
Since photons can be either resolved or act as point-like particles (direct), 
there are four different contributions, resolved-resolved, resolved-direct, 
direct-resolved and direct-direct. Currently these are not automatically 
mixed, but the user has to generate the relevant processes separately and 
combine them for the final result. This is illustrated in sample main program 
<code>main69.cc</code>. 
</p> 
 
<br/><br/><table><tr><td><strong>Photon:ProcessType  </td><td>  &nbsp;&nbsp;(<code>default = <strong>1</strong></code>; <code>minimum = 1</code>; <code>maximum = 4</code>)</td></tr></table>
Sets desired contribution for photon-photon interactions. 
<br/>
<input type="radio" name="1" value="1" checked="checked"><strong>1 </strong>:  Resolved-Resolved: Both colliding photons are  resolved and the partonic content is given by the PDFs. Hard processes  and non-diffractive events can be generated. <br/>
<input type="radio" name="1" value="2"><strong>2 </strong>:  Resolved-Direct: Photon A is resolved and photon B  unresolved, i.e. act as an initiator for the hard process. Hard processes  with a parton and a photon in the initial state can be generated.<br/>
<input type="radio" name="1" value="3"><strong>3 </strong>:  Direct-Resolved: As above but now photon A is unresolved  and photon B resolved. <br/>
<input type="radio" name="1" value="4"><strong>4 </strong>:  Direct-Direct: Both photons are unresolved. Hard  processes with two photon initiators can be generated.<br/>
 
<h3>Resolved photon</h3> 
 
<p> 
Photons can either interact directly as an unresolved particle or as a 
hadronic state ("Vector Meson Dominance"). In the latter case the hard 
process can be simulated using PDFs to describe the partonic structure 
of the resolved photon. The evolution equations for photons include an 
additional term that corresponds to <i>gamma &rarr; q qbar</i> splittings. 
Due to this, the PDFs are somewhat different for photons than for hadrons 
and some parts of event generation need special attention. 
</p> 
 
<h4>Process-level generation</h4> 
 
<p> 
Due to the additional term in the evolution equations the quarks in a 
resolved photon may carry a very large fraction <i>(x~1)</i> of the photon 
momentum. In these cases it may happen that, after the hard process, there is 
no energy left to construct the beam remnants. This is true especially if 
a heavy quark is taken out from the beam and a corresponding massive 
antiquark needs to be added to the remnant system to conserve flavour. Even 
though these events are allowed based on the PDFs alone, they are not physical 
and should be rejected. Therefore some amount of errors can be expected when 
generating events close to the edge of phase space, e.g. when collision 
energy is low. 
</p> 
 
<h4>Spacelike showers</h4> 
 
<p> 
The parton showers are generated according to the DGLAP evolution equations. 
Due to the <i>gamma &rarr; q qbar</i> splitting in the photon evolution, 
a few modifications are needed for the ISR algorithm. 
<ul> 
<li> 
The additional term corresponds to a possibility to find the original beam 
photon during the backwards evolution, which is added to the QED part of the 
spacelike shower evolution. If this splitting happens there is no need to 
construct the beam remnants for the given beam. 
</li> 
<li> 
The heavy quark production threshold with photon beams is handled in a 
similar manner as for hadrons, but now the splitting that is forced 
to happen is <i>gamma &rarr; Q Qbar</i>. 
</li> 
<li> 
As the splittings in backwards evolution increases the <i>x</i> of the 
parton taken from the beam, the splittings can lead to a situation where 
there is no room left for massive beam remnants. To make sure that the 
required  remnants can be constructed, splittings that would not leave 
room for the beam remnants are not allowed. 
</li> 
</ul> 
</p> 
 
<h4>MPIs in photon-photon</h4> 
 
<p> 
Multiparton interactions with resolved photon beams are generated as with 
hadron beams. The only difference follows again from the additional 
<i>gamma &rarr; q qbar</i> splittings where the beam photon becomes 
unresolved. If this splitting happens during the interleaved evolution 
for either of the photon beams no further MPIs below the branching scale 
<i>pT</i> are allowed since the photon is not resolved anymore. 
</p> 
 
<p> 
If there have been multiple interactions and a <i>gamma &rarr; q qbar 
</i> splitting occur, the kinematics of this branching are not constructed 
in the spacelike shower. Instead the <i>pT</i> scale of the branching is 
stored and the relevant momenta are then fixed in the beam remnant handling. 
Therefore the status codes for the partons related to this splitting 
actually refer to beam remnants. 
</p> 
 
<p> 
If there are no MPIs before the <i>gamma &rarr; q qbar</i> splitting, 
this splitting is constructed in the spacelike shower in the usual way, 
but the mother beam photon is not added to the event record, since a copy 
of it already exists at the top of the event record. This is unlike the 
documentation of other ISR splittings, where the mother of the branching 
is shown, but consistent with the photon not being added (a second time) 
for events that contain several MPIs. Optionally the photon can be shown, 
using the following flag. 
 
<br/><br/><strong>Photon:showUnres</strong>  <input type="radio" name="2" value="on"><strong>On</strong>
<input type="radio" name="2" value="off" checked="checked"><strong>Off</strong>
 &nbsp;&nbsp;(<code>default = <strong>off</strong></code>)<br/>
Show the evolution steps of the beam photon in the event record, if on. 
   
</p> 
 
<p> 
Currently the default values for the parameters related to multiparton 
interactions are the same as in hadronic collision so no tuning for the 
MPIs in photon-photon has been done. This holds also for the parameters 
related to the impact-parameter dependence. The total cross section for 
photon-photon collisions is paramerized as in [<a href="Bibliography.php" target="page">Sch97</a>]. Since 
the total cross section includes contribution also from elastic and 
diffractive events, a multiplicative factor is introduced to control 
the non-diffractive component. 
</p> 
 
<br/><br/><table><tr><td><strong>Photon:sigmaNDfrac </td><td></td><td> <input type="text" name="3" value="0.7" size="20"/>  &nbsp;&nbsp;(<code>default = <strong>0.7</strong></code>; <code>minimum = 0.5</code>; <code>maximum = 1.0</code>)</td></tr></table>
Fraction of non-diffractive cross section of the total cross section. 
Default value is motivated by earlier Pythia 6 studies. 
   
 
<h4>Beam Remnants</h4> 
 
<p> 
To construct the beam remnants, one should know whether the parton 
taken from the beam is a valence parton or not. The valence partons of a 
photon includes the partons that originate from <i>gamma &rarr; q qbar</i> 
splittings of the original beam photon and the valence partons from the 
hadron-like part of the PDF. In either case, the flavour of the valence 
quarks can fluctuate. Unfortunately the decomposition to the different 
components are typically not provided in the PDF sets and some further 
assumptions are needed to decide the valence content. 
</p> 
 
<p> 
When ISR is applied for photon beams it is possible to end up to the original 
beam photon during the evolution. Therefore there are three possibilities for 
the remnants: 
<ul> 
<li> 
Remnants need to be constructed for both beams. 
</li> 
<li> 
Remnants are constructed only for one side. 
</li> 
<li> 
No need for remnants on either side. 
</li> 
</ul> 
The last case is the simplest as all the partons in the event are already 
generated by the parton showers. In the first case the remnants and 
primordial <i>kT</i> are constructed similarly as for normal hadronic 
interactions [<a href="Bibliography.php" target="page">Sjo04</a>]. For the second case the momenta of the 
remnant partons can not be balanced between the two beams as the kinematics 
of the other side are already fixed. In these cases the momenta are balanced 
between the scattered system and the remnants. 
</p> 
 
<p> 
Since the primordial <i>kT</i> increases the invariant mass of the remnants 
and the scattered system, it may again happen that there is no room for the 
remnant partons after <i>kT</i> is added, so the kinematics can not be 
constructed. In this case new values for <i>kT</i> are sampled. If this 
does not work, a new shower is generated and in some rare cases the 
parton-level generation fails and the hard process is rejected. The inclusion 
of additional MPIs increases the invariant mass of the remnants and takes 
more momentum from the beam particles. Even though the MPIs that would 
not leave enough room for the remnants are rejected, these can still lead 
to a situation where the kinematics cannot be constructed due to the added 
primordial <i>kT</i>. This may cause some amount of errors especially when 
the invariant mass of <i>gamma-gamma</i> system is small. 
</p> 
 
<h3>Photon-photon in lepton-lepton</h3> 
 
<p> 
Photon-photon interactions can happen also in lepton-lepton collisions. 
How to set up these collisions is described in 
<?php $filepath = $_GET["filepath"];
echo "<a href='PDFSelection.php?filepath=".$filepath."' target='page'>";?>PDF selection</a>. Since the current 
framework can handle only (quasi-)real photons, a upper limit for the 
photon virtuality needs to be set. This can be done with the parameter 
<code>Photon:Q2max</code>. The upper limit for virtuality will set also 
the upper limit for the <i>k_T</i> of the photon, which in turn will 
be the same as the <i>k_T</i> of the scattered lepton. Also some other 
cuts can be imposed. 
 
<br/><br/><table><tr><td><strong>Photon:Q2max </td><td></td><td> <input type="text" name="4" value="1.0" size="20"/>  &nbsp;&nbsp;(<code>default = <strong>1.0</strong></code>; <code>minimum = 0.01</code>; <code>maximum = 2.0</code>)</td></tr></table>
Upper limit for (quasi-)real photon virtuality in <i>GeV^2</i>. 
   
 
<br/><br/><table><tr><td><strong>Photon:Wmin </td><td></td><td> <input type="text" name="5" value="10.0" size="20"/>  &nbsp;&nbsp;(<code>default = <strong>10.0</strong></code>; <code>minimum = 5.0</code>)</td></tr></table>
Lower limit for invariant mass of <i>gamma-gamma</i> system in <i>GeV</i>. 
   
 
<br/><br/><table><tr><td><strong>Photon:Wmax </td><td></td><td> <input type="text" name="6" value="-1.0" size="20"/>  &nbsp;&nbsp;(<code>default = <strong>-1.0</strong></code>)</td></tr></table>
Upper limit for invariant mass of <i>gamma-gamma</i> system in <i>GeV</i>. 
A value below <code>Photon:Wmin</code> means that the invariant mass of 
the original <i>l+l-</i> pair is used as an upper limit. 
   
 
<br/><br/><table><tr><td><strong>Photon:thetaAMax </td><td></td><td> <input type="text" name="7" value="-1.0" size="20"/>  &nbsp;&nbsp;(<code>default = <strong>-1.0</strong></code>; <code>maximum = 3.141593</code>)</td></tr></table>
Upper limit for scattering angle of lepton A in <i>rad</i>. A negative 
value means that no cut is applied. Since <i>k_T</i> depends on virtuality 
of the emitted photon, the <code>Photon:Q2max</code> cut is usually more 
restrictive unless a very small angle is used. 
   
 
<br/><br/><table><tr><td><strong>Photon:thetaBMax </td><td></td><td> <input type="text" name="8" value="-1.0" size="20"/>  &nbsp;&nbsp;(<code>default = <strong>-1.0</strong></code>; <code>maximum = 3.141593</code>)</td></tr></table>
As above but for lepton B. 
   
 
</p> 
 
<h4>MPIs in lepton-lepton</h4> 
 
<p> 
The invariant mass of <i>gamma-gamma</i> system from lepton beams will vary. 
Therefore, to generate MPIs and non-diffractive events in <i>gamma-gamma</i> 
collisions from lepton beams, the MPI framework is initialized with five 
values of <i>W</i> from <code>Photon:Wmin</code> to 
<code>Photon:Wmax</code>. The parameter values are then interpolated 
for the sampled <i>W</i>. 
</p> 
 
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

if($_POST["1"] != "1")
{
$data = "Photon:ProcessType = ".$_POST["1"]."\n";
fwrite($handle,$data);
}
if($_POST["2"] != "off")
{
$data = "Photon:showUnres = ".$_POST["2"]."\n";
fwrite($handle,$data);
}
if($_POST["3"] != "0.7")
{
$data = "Photon:sigmaNDfrac = ".$_POST["3"]."\n";
fwrite($handle,$data);
}
if($_POST["4"] != "1.0")
{
$data = "Photon:Q2max = ".$_POST["4"]."\n";
fwrite($handle,$data);
}
if($_POST["5"] != "10.0")
{
$data = "Photon:Wmin = ".$_POST["5"]."\n";
fwrite($handle,$data);
}
if($_POST["6"] != "-1.0")
{
$data = "Photon:Wmax = ".$_POST["6"]."\n";
fwrite($handle,$data);
}
if($_POST["7"] != "-1.0")
{
$data = "Photon:thetaAMax = ".$_POST["7"]."\n";
fwrite($handle,$data);
}
if($_POST["8"] != "-1.0")
{
$data = "Photon:thetaBMax = ".$_POST["8"]."\n";
fwrite($handle,$data);
}
fclose($handle);
}

?>
</body>
</html>
 
<!-- Copyright (C) 2017 Torbjorn Sjostrand --> 
