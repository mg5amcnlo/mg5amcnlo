#!/usr/bin/perl -w

################################################################################
#                                                                             ##
#                    MadGraph/MadEvent                                        ##
#                                                                             ##
# FILE : qnumbers.pl                                                          ##
# VERSION : 1.0                                                               ##
# DATE : 27 August 2007                                                       ##
# AUTHOR : Michel Herquet (UCL-CP3)                                           ##
#                                                                             ##
# DESCRIPTION : script add qnumbers info from particles.dat to the param_card ##
# USAGE : ./qnumbers.pl particles.dat param_card.dat                          ##
# OUTPUT : modified param_card.dat                                            ##
################################################################################


# Parse the command line arguments

if ( $#ARGV < 1 ) {
     die "This script must be called with two filenames as arguments!\n";
}

open(PARTICLES,"<$ARGV[0]") || die "Cannot open input file called $ARGV[0], stopping\n";
open(CARD,"<$ARGV[1]") || die "Cannot open input file called $ARGV[1], stopping\n";

# Save card content
@param_card=<CARD>;
close(CARD);

# Test param_card for already existing QNUMBERS and exit if it's the case
foreach $line (@param_card) {
    next if( ($line =~/^#/));
    if(lc($line) =~/^block\s+qnumbers/) {
       die "Already existing QNUMBERS block in $ARGV[0], exiting\n";
    }  
}

print "Updating $ARGV[1] with QNUMBERS...\n";

# Open param_card in write mode
open(CARD,">$ARGV[1]") || die "Cannot open input file called $ARGV[1]in write mode, stopping\n";

# Parse param card and write QNUMBERS just before mass
foreach $line_pc (@param_card) {

if(lc($line_pc)=~ m/block\s+mass/) {

# Write header
print CARD "#===========================================================\n";
print CARD "# QUANTUM NUMBERS OF NEW STATE(S) (NON SM PDG CODE) IF ANY\n"; 
print CARD "# (see below for masses and decay tables)\n";
print CARD "# These blocks are automatically created by the MadGraph\n"; 
print CARD "# qnumbers.pl script from the particles.dat model file\n";
print CARD "#===========================================================\n";

# scan particles.dat and print QNUMBERS info
foreach $line (<PARTICLES>) {
    
    # forget comments and lines after MULTIPARTICLES tag
    
    last if($line =~ m/MULTIPARTICLES/);
    next if( ($line =~ m/#/) || ($line =~/^\s+/));
    
    # split line
    @elem = split(/\s+/, $line);
    
    # Check number of elements
    if ($#elem != 8) {
        chomp($line);
        print "Line $line does not appear to be a valid entry\n";
        next;
    }
    
    # Check if PDG code < 26 (SM particles) MSSM particles
    $pdg=abs($elem[8]);
    $pdgmil=int($pdg/1000000);
    $pdg=$pdg % 1000000;
    if ($pdgmil<=2 && ($pdg<=6 || ($pdg>=11 && $pdg<=16) || ($pdg>=21 && $pdg<=25) || ($pdg>=35 && $pdg<=37))) {next;}

    # write down QNUMBERS output
    print CARD "BLOCK QNUMBERS ",abs($elem[8])," # $elem[0]\n";
    
    # write down electric charge
    # in this version, always 0
    print CARD "         1  0 # 3 times electric charge\n";
    
    #write down spin information (default scalar)
    print CARD "         2  ";
    if ($elem[2] eq 'S') {print CARD "1";}
    elsif ($elem[2] eq 'F') {print CARD "2";}
    elsif ($elem[2] eq 'V') {print CARD "3";}
    elsif ($elem[2] eq 'T') {print CARD "5";}
    else {print CARD "0";}
    print CARD " # number of spin states (2S+1)\n";
    
    #write down color information (default singlet)
    print CARD "         3  ";
    if ($elem[6] eq 'S') {print CARD "1";}
    elsif ($elem[6] eq 'T') {print CARD "3";}
    elsif ($elem[6] eq 'O') {print CARD "8";}
    else {print CARD "1";}
    print CARD " # colour rep (1: singlet, 3: triplet, 8: octet)\n";
    
    #write down particle/anti-particle info (default no)
    print CARD "         4  ";
    if ($elem[0] eq $elem[1]) {print CARD "0";}
    else {print CARD "1";}
    print CARD " # Particle/Antiparticle distinction (0=own anti)\n";
    
}
print CARD "# END of QNUMBERS blocks\n"; 
print CARD "#===========================================================\n";

}

print CARD $line_pc;
}



close(PARTICLES);
close(CARD);
