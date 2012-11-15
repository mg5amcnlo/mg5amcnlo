#!/usr/bin/perl -w

################################################################################
# replace.pl
#  Johan Alwall (alwall@slac.stanford.edu)
#
# DESCRIPTION : script to replace particle codes in MG/ME event files
# USAGE :
# replace.pl infile.lhe outfile.lhe
# Asks for list of particles to replace. To replace electrons and neutrinos
# with electrons, muons and taus, give
# 11: 11 13 15
# -12: -12 -14 -16
# done
# To replace electrons and positrons in W-W+ pair production please run
# script twice, ones for the e-/ve~, once for e+/ve.
# Automatically increases cross section with correct factor.
################################################################################

use Compress::Zlib;
use List::Util qw[min max];

print "Running script replace.pl to replace particle codes in event file.\n";

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
if ( $#ARGV != 1 ) {
  die "Usage: replace.pl infile.lhe outfile.lhe\n";
}
my $outfile = pop(@ARGV);
my $infile = pop(@ARGV);

open INFILE, "<$infile" or die ("Error: Couldn't open file $infile\n");

###########################################################################
# Read which particles to replace
###########################################################################

print "Enter particles to be replaced in syntax PID : PID1 PID2 PID3 ...\n";
print "(note that they will be replaced together, so must be same number of\n";
print " particles in each line)\n";
print "Enter \"done\" or <return> when ready\n";
$line=<STDIN>;
my %particles;
my $nparts=0;
my $i=0;

while ($line !~ m/^\s*done\s*$/i && $line !~ m/^\s*$/){
    # Remove leading whitespace
    $line  =~ s/^\s+//;
    if($line =~ m/^(-?\d+)\s*\:\s*((-?\d+\s+)+)/){
	@tmp = split (/\s+/,$2);
	if($i == 0) { $nparts = $#tmp+1; }
	elsif($#tmp+1 != $nparts){
	    die "Error: Not same number of particles in each replace: ",$#tmp+1," $nparts\n";
	}
	$i = $1;
	$particles{$i}=[ @tmp ];
	print "Replacing $i with @{$particles{$i}}\n";
    }
    elsif($line !~ m/^#/){
	print "Please use syntax PID1 : PID2 PID3 ... or \"done\" or <return> when ready\n";
    }
    $line=<STDIN>;
}
@keys = keys %particles;
if($#keys < 0) {die "Error: No particles to replace\n";}
print "Multiplying cross section and weight by $nparts\n";

###########################################################################
# Go through file and perform replacements
###########################################################################

open OUTFILE, ">$outfile" or die ("Error: Couldn't open file $outfile for writing\n");

# No. events and cross-section numbers file
$nevents = $xsec = $trunc = $unitwgt = -1;

# Keep track in which block we are
$initblock = 0;
$headerblock = 0;
$eventblock = 0;
$eventcount = 0;
$rdnseed = 0;

while (my $line = <INFILE>) {

    # Extract <MGGenerationInfo> information
    if ($line =~ m/$end_header/) { 
	if($nevents == -1 || $xsec == -1 || $trunc == -1 || $unitwgt == -1) {
	    die "Error: Couldn't find number of events and cross section in $infile.\n";
	}
	$headerblock = 0; 
	print OUTFILE "<ReplaceParticleInfo>\n";
	printf OUTFILE "#  Number of Events        : %11i\n",$nevents;
	printf OUTFILE "#  Integrated weight (pb)  : %11.4E\n",$xsec;
	printf OUTFILE "#  Truncated wgt (pb)      : %11.4E\n",$trunc;
	printf OUTFILE "#  Unit wgt                : %11.4E\n",$unitwgt;
	if($rdnseed > 0){
	    print OUTFILE " ", $rdnseed+1, "  = gseed ! Random seed for next iteration of replace.pl\n";}
	print OUTFILE "</ReplaceParticleInfo>\n";
	if($rdnseed > 0) {print "Initialize random seed with $rdnseed\n";srand($rdnseed);}
	else {print "Warning: Random seed 0, use default random seed (unreproducible)\n";}
    }
    if ($line =~ m/$end_init/) { $initblock=0; }
    if ($line =~ m/$end_event/) { 
	$eventcount++;
	$eventblock=0; 
    }
    
    if ($headerblock == 1) {
	# In header
	if ($line =~ m/^#\s+Number of Events\s*:\s*(.*)\n/) { $nevents = $1; }
	if ($line =~ m/^#\s+Integrated weight \(pb\)\s*:\s*(.*)\n/) { $xsec = $1*$nparts; }
	if ($line =~ m/^#\s+Truncated wgt \(pb\)\s*:\s*(.*)\n/) { $trunc = $1*$nparts; }
	if ($line =~ m/^#\s+Unit wgt\s*:\s*(.*)\n/) { $unitwgt = $1*$nparts; }
	if ($line =~ m/^\s*(\d+)\s*=\s*gseed/) { $rdnseed = $1; }
    } elsif ($initblock > 0) {
	# In <init> block

	if($initblock > 1) {
	# Remove leading whitespace and split
	    $line  =~ s/^\s+//;
	    @param = split(/\s+/, $line);
	    if ($#param != 3) { die "Error: Wrong number of params in init ($#param)"; }
	    $param[0]*=$nparts;
	    $param[1]*=$nparts;
	    $param[2]*=$nparts;
	    $line = sprintf " %18.11E %18.11E %18.11E %3i\n",$param[0],$param[1],$param[2],$param[3];
	}
	$initblock++;

    } elsif ($eventblock > 0) {
	# In <event> block
	# Remove leading whitespace and split
	$line  =~ m/^\s*(.*)/;
	if($line=~/^\#/){
	#    print OUTFILE $line;
	}
	else{
	    @param = split(/\s+/, $1);
	    if($eventblock == 1){
		if ($#param != 5) { die "Error: Wrong number of params in event $eventcount \($#param / 5\)"; }
		$param[2]*=$nparts;
		$line = sprintf "%2i %3i %13.6E %13.6E %13.6E %13.6E\n",$param[0],$param[1],$param[2],$param[3],$param[4],$param[5];
		
		# Randomly choose a particle from the lists
		$rnumber = int(rand($nparts));
	    }
	    else {
#	    @param = split(/\s+/, $1);
		if ($#param != 12) { die "Error: Wrong number of params in event $eventcount \($#param / 12\)"; }
		if($particles{$param[0]}){
		    $line = sprintf "%9i %4i %4i %4i %4i %4i %18.10E %18.10E %18.10E %18.10E %18.10E %1.0f. %2.0f.\n",
		    $particles{$param[0]}[$rnumber],$param[1],$param[2],$param[3],$param[4],$param[5],$param[6],$param[7],$param[8],$param[9],$param[10],$param[11],$param[12];
		}
	    }
	}
	$eventblock++;
    }
	    
    if ($line =~ m/$begin_header/) { $headerblock = 1; }
    if ($line =~ m/$begin_init/) { $initblock=1; }
    if ($line =~ m/$begin_event/) { $eventblock=1; }

    print OUTFILE "$line";
}

close OUTFILE;
close INFILE;

print "Wrote $eventcount events\n";
if( $eventcount < $nevents ) { print "Warning: $infile ended early\n"; }

exit(0);


