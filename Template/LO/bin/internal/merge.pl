#!/usr/bin/perl -w

################################################################################
# merge.pl
#  Richard Corke (richard.corke@thep.lu.se)
#
# Changed structure to not read entire files into memory
# 
# Based on merge.pl v1.2 by Michel Herquet (UCL-CP3)
# DESCRIPTION : script to merge to LHE events files
# USAGE :
# ./merge.pl eventfile1.lhe.gz eventfile2.lhe.gz ... result.lhe.gz banner.txt
# Note that result can be eventfile1, eventfile2, ...
# OUTPUT : merged file, banner
################################################################################

use Compress::Zlib;
use List::Util qw[min max];


###########################################################################
# Initialisation
###########################################################################
# Set tag names for later
my $begin_header = '<header>';
my $end_header   = '</header>';
my $begin_event  = '<event>';
my $end_event    = '</event>';
my $begin_init   = '<init>';
my $end_init     = '</init>';

# Parse the command line arguments
if ( $#ARGV < 2 ) {
  die "This script must be called with at least three filenames as arguments!\n";
}
my $bannerfile = pop(@ARGV);
my $outfile = pop(@ARGV);


###########################################################################
# First pass - extract number of events and cross-sections
###########################################################################
foreach $infile (@ARGV) {
  $gzin = gzopen($infile, "r") || die ("Couldn't open file $infile\n");

  # No. events and cross-section from current file
  $noevents = $xsec = -1;

  # Storage for current file's init block
  @gzinit = ();

  # Keep track if we are in the init block or not
  $initblock = 0;

  while (1) {
    $gzbytes = $gzin->gzreadline($gzline);
    if ($gzbytes == -1) { die("Error reading from file $infile\n"); }
    if ($gzbytes == 0)  { last; }

    # Extract <MGGenerationInfo> information
    if ($initblock == 0) {
      if ($gzline =~ s/#  Number of Events\s*:\s*(.*)\n/$1/) { $noevents = $gzline; }
      if ($gzline =~ s/#  Integrated weight \(pb\)\s*:\s*(.*)\n/$1/) { $xsec = $gzline; }

      # Check if we enter <init> block
      if ($gzline =~ m/$begin_init/) { $initblock++; next; }

    # Extract <init> information
    } else {
 
      # Check for end of <init> block
      if ($gzline =~ m/$end_init/) { last; }

      # Remove leading whitespace and split
      $gzline  =~ s/^\s+//;
      @gzparam = split(/\s+/, $gzline);

      # Check <init> block line has right no. of entries
      if ($initblock == 1) {
        if ($#gzparam != 9) { die "Not right number of params in init"; }
      } else {
        if ($#gzparam != 3) { die "Not right number of params in init"; }
      }

      push(@gzinit, [ @gzparam ]);
      $initblock++;

    }
  }
  $gzin->gzclose();

  # Check the file contained all the information we need
  if ($noevents == -1 || $xsec == -1) {
    die("Couldn't extract No. Events / Cross-section from $infile")
  }

  # Store information for later
  push(@infiles, [ $infile, $noevents, $xsec, [ @gzinit ] ]);
}


###########################################################################
# Check/compute cross-sections, totals and unit weight
###########################################################################
$oldxsec        = $infiles[0][2];
@oldinit        = @{$infiles[0][3]};
$totevents = 0;  $totxsec = 0.0;
foreach $infile (@infiles) {
  print "Input file: $infile->[0]\n";
  print " No. Events = $infile->[1], Cross-section = $infile->[2]\n";

  # Check that cross sections do not differ too much
  $newxsec = $infile->[2];
  if (abs(($newxsec - $oldxsec) / $newxsec) > 0.05 ) {
    print " WARNING Cross sections do not agree with a 5\% precision!\n";
  }
  $oldxsec = $newxsec;

  @currinit = @{$infile->[3]};
  # Same number of lines in <init> block?
  if ($#oldinit != $#currinit)             { die("Init blocks do not match"); }

  # Same number of entries on first line of <init> block?
  if ($#{$oldinit[0]} != $#{$currinit[0]}) { die("Init blocks do not match"); }

  # All entries the same on first line of <init> block?
  for ($i = 0; $i <= $#{$oldinit[0]}; $i++) {
    if ($oldinit[0][$i] != $currinit[0][$i])
      { die("Init blocks do not match"); }
  }

  # Create new init block (overwrite first file's init block data)
  for ($i = 1; $i <= $#oldinit; $i++) {
    if ($oldinit[$i][3] != $currinit[$i][3]) { die("Init blocks do not match"); }

    print " xsecup = $currinit[$i][0], xerrup = $currinit[$i][1]\n";
    print " xmaxup = $currinit[$i][2], lprup = $currinit[$i][3]\n";

    # XSECUP = sum(xsecup * no.events) / tot.events
    # XERRUP = sqrt( sum(sigma^2 * no.events^2) ) / tot.events

    # Here we temporarily store:
    #  sum(xsecup * no.events)
    #  sum(sigma^2 * no.events^2)
    if (\$oldinit == \$currinit) {
      $oldinit[$i][0] *= $infile->[1];
      $oldinit[$i][1] *= $oldinit[$i][1] * $infile->[1]**2;

    } else {
      $oldinit[$i][0] += ($currinit[$i][0] * $infile->[1]);
      $oldinit[$i][1] += $currinit[$i][1]**2 * $infile->[1]**2;

    }

    # XMAXUP = max(xmaxup)
    $oldinit[$i][2] = max($oldinit[$i][2], $currinit[$i][2]);
  }

  # Total number of events and total cross-section
  $totevents += $infile->[1];
  $totxsec += ($infile->[1] * $infile->[2]);

  print "\n";
}
print "\n";

# Finish calculation of XSECUP and XERRUP
for ($i = 1; $i <= $#oldinit; $i++) {
  $oldinit[$i][0] /= $totevents;
  $oldinit[$i][0] = sprintf('%0.5E', $oldinit[$i][0]);

  $oldinit[$i][1] = sqrt($oldinit[$i][1]);
  $oldinit[$i][1] /= $totevents;
  $oldinit[$i][1] = sprintf('%0.5E', $oldinit[$i][1]);
}

# Finish calculation of total xsec and new unit weight
$totxsec /= $totevents;
$dispxsec = sprintf('%0.5E', $totxsec);
$uwgt = sprintf('%0.5E', $totxsec / $totevents);

# Display new information
print "Banner file: $bannerfile\n";
print "Output file: $outfile\n";
print " No. Events = $totevents, Cross-section = $dispxsec\n";
for ($i = 1; $i <= $#oldinit; $i++) {
  print " xsecup = $oldinit[$i][0], xerrup = $oldinit[$i][1]\n";
  print " xmaxup = $oldinit[$i][2], lprup = $oldinit[$i][3]\n";
}
print "\n";
print " Unit weight = $uwgt\n";


###########################################################################
# Second pass - output file with new information
###########################################################################
# The first file is written out with changes to the header, init block
# and unit weight of each event.
# All events from other files are then written out with their unit weight
# changed.
###########################################################################

$gzout    = gzopen($outfile, "w") || die ("Couldn't open file $outfile\n");
open(BANNER, ">$bannerfile")      || die ("Couldn't open file $outfile\n");

$stage = 0;
$eventcount = 0;
foreach $infile (@infiles) {
  $gzin = gzopen($infile->[0], "r") || die ("Couldn't open file $infile\n");

  while (1) {
    $gzbytes = $gzin->gzreadline($gzline);
    if ($gzbytes == -1) { die("Error reading from file $infile\n"); }
    if ($gzbytes == 0)  { last; }

    # Pre-header
    if ($stage == 0) {
      if ($gzline =~ m/$begin_header/) { $stage++; }

    # Header (and output to banner)
    } elsif ($stage == 1) {
      $gzline =~ s/#  Integrated weight \(pb\)\s*:(.*)\n/#  Integrated weight (pb)  :  $dispxsec\n/;
      $gzline =~ s/#  Number of Events\s*:(.*)\n/#  Number of Events        :  $totevents\n/;
      $gzline =~ s/#  Unit wgt\s*:(.*)\n/#  Unit wgt                :  $uwgt\n/;

      if ($gzline =~ m/$end_header/)   { $stage++; }
      else { print BANNER $gzline; }

    # Init block
    } elsif ($stage == 2) {
      if ($gzline =~ m/$end_init/) {
        $gzline = "<init>\n";

        for ($i = 0; $i <= $#oldinit; $i++) {
          $gzline .= "  ";
          for ($j = 0; $j <= $#{$oldinit[$i]}; $j++) {
            $gzline .= "$oldinit[$i][$j] ";
          }
          $gzline .= "\n";
        }

        $gzline .= "</init>\n";
        
        $stage++;
      } else { next; }

    # Pre-event
    } elsif ($stage == 3) {
      if ($gzline =~ m/$begin_event/)  { $stage++; } else { next; }

    # Event information
    } elsif ($stage == 4) {
      $gzline  =~ s/^\s+//;
      @gzparam = split(/\s+/, $gzline);
      # Keep weight sign from original LHE file
      $signed_uwgt = $uwgt;
      if ($gzparam[2] < 0) {
	  $signed_uwgt = -1 * $signed_uwgt;
      }
      $gzline = " $gzparam[0] $gzparam[1] $signed_uwgt $gzparam[3] $gzparam[4] $gzparam[5]\n";
      if ($#gzparam != 5) { die "Not right number of param in first line of event"; }
      $gzline = " $gzparam[0] $gzparam[1] $uwgt $gzparam[3] $gzparam[4] $gzparam[5]\n";

      $stage++;

    # Event particles
    } elsif ($stage == 5) {
      if ($gzline =~ m/$end_event/)    { $stage = 3; $eventcount++ }

    }

    # Write out the line
    $gzout->gzwrite($gzline);
  }
}

# Write out closing tag and close
$gzout->gzwrite("</LesHouchesEvents>\n");
$gzout->gzclose();

print "Wrote $eventcount events\n";
exit(0);

