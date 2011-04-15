#!/usr/bin/perl
use strict;
use POSIX;
use File::Copy;
#use Carp ();
#$SIG{__WARN__} = \&Carp::cluck;
#$SIG{__DIE__}  = \&Carp::confess;




#########################################################################################
#                                                                                       #
#       Conversion Script: Merging tool for 	interaction.dat				#
#						VariableName.dat			#
#						particles.dat				#
#											#
# Produce necessary files for running in a new "user" model context:			#
#	--->coupl.inc									#
#	--->couplings.f									#
#	--->input.inc									#
#	--->printout.f									#
#	--->lha_reading.f								#
#	--->param_card.dat								#
#											#
#		Headers, intermediates and Footer files have to be placed in		#
#		Current_Directory/Header_and_Footer					#
#                                                                                       #
#       S. de Visscher, last modification: May 2006					#
#	Universite Catholique de Louvain             Fynu/Cp3				#
#                                                                                       #
#                                                                                       #
#########################################################################################

print "Need to keep old couplings.f and param_card.dat? yes or no: ";
my $response=<STDIN>;
while(!($response=~/yes/) && !($response=~/no/)){
	print "Please answer by yes or no: ";
	$response=<STDIN>;print "$response";
}
if($response=~/yes/){
	system("mkdir -p OldFiles");
	my $id_coupl=0;
	my $id_param=0;
	if(-f "couplings.f"){
		while (-e "OldFiles/couplings_$id_coupl.f"){
			$id_coupl++;
		}
		my $filename="couplings_$id_coupl.f";
		system("mv couplings.f OldFiles/$filename");
		print "Old couplings file saved under $filename name\n";
	}
	else {print "NO SAVING: couplings.f file is absent\n";}
	if(-f "param_card.dat"){
		while (-e "OldFiles/param_card_$id_param.dat"){
			$id_param++;
		}
		my $filename="param_card_$id_param.dat";
		system("mv param_card.dat OldFiles/$filename");
		print "Old param_card file saved under $filename name\n";
	}
	else {print "NO SAVING: param_card.dat file is absent\n";}
}
my $massfile=1;
open(INFILE1,"interactions.dat") or die "No Interaction File->Abort";
open(INFILE2,"particles.dat") or die "No Interaction File->Abort";
open(INFILE3,"VariableName.dat") or die "No VariableName file->Abort";
open(INFILE4,"Header_and_footer/couplings_header.dat") or die "No couplings_header file File->Abort";
open(INFILE5,"Header_and_footer/couplings_footer.dat") or die "No couplings_footer file File->Abort";
open(INFILE6,"Header_and_footer/coupl_header.dat") or die "No coupl_header File->Abort";
open(INFILE8a,"Header_and_footer/param_I.dat") or die "No param1 file->Abort";
open(INFILE8b,"Header_and_footer/param_II.dat") or die "No param2 file->Abort";
open(INFILE8c,"Header_and_footer/param_III.dat") or die "No param3 file->Abort";
open(INFILE9,"Header_and_footer/input_header.dat") or die "No input_header file->Abort";
open(INFILE10a,"Header_and_footer/print_I.dat") or die "No print1 file->Abort";
open(INFILE10b,"Header_and_footer/print_II.dat") or die "No print2 file->Abort";
open(INFILE10c,"Header_and_footer/print_III.dat") or die "No print3 file->Abort";
open(INFILE10d,"Header_and_footer/print_IV.dat") or die "No print4 file->Abort";
open(INFILE11,"Header_and_footer/rw_I.dat") or die "No R1 file->Abort";
open(INFILE13,"Header_and_footer/rw_II.dat") or die "No R2 file->Abort";
open(INFILE14,"Header_and_footer/rw_III.dat") or die "No R3 file->Abort";
open(INFILE15,"Header_and_footer/rw_IV.dat") or die "No R4 file->Abort";
open(INFILE16a,"Header_and_footer/printcoup_I.dat") or die "No print1 file->Abort";
open(INFILE16b,"Header_and_footer/printcoup_II.dat") or die "No print1 file->Abort";
open(INFILE17,"masses.dat") or $massfile=0;

open(OUTFILE1,">couplings.f") or die "No output writing possibilities->Abort";
open(OUTFILE2,">coupl.inc") or die "No coupl.inc writing possibilities->Abort";
open(OUTFILE4,">param_card.dat") or die "No param_card writing possibilities->Abort";
open(OUTFILE5,">input.inc") or die "No input.inc writing possibilities->Abort";
open(OUTFILE6,">printout.f") or die "No printout writing possibilities->Abort";
open(OUTFILE7,">lha_reading.f") or die "No lha_reading writing possibilities->Abort";
open(OUTFILE8,">couplings_test.f") or die "No couplings_test writing possibilities->Abort";


######################Writing files headers...###################################################################

while(<INFILE4>){print OUTFILE1;} # couplings.f header written
while(<INFILE6>){print OUTFILE2;} #coupl.inc header written
while(<INFILE9>){print OUTFILE5;}
while(<INFILE8a>){print OUTFILE4;}
while(<INFILE11>){print OUTFILE7;}
while(<INFILE10a>){print OUTFILE6;}
while(<INFILE16a>){print OUTFILE8;}


######################Writing couplings.f and coupl.inc and MASS and DECAY terms in ident_card#################################


my $new_vertex_flag=0;
my $vertex_type=-1;
my $Type=-1;
my @VT;
my @prefix;
$prefix[0]="      double precision ";
$prefix[1]="      double complex   ";
my $tag0=0;
my $tag1=0;
my @couplings_list;
my $ind=0;
my $limit=0;
my $limitparticle=0;
#start to read interaction.dat
while(my $current_line=<INFILE1>){
	if($current_line=~/USR/){$limit=1}
#if line is not empty or not a comment
	unless($current_line=~/^\s/ || $current_line=~/#/ || $limit==0){	
		chomp($current_line);
		#print "->$current_line\t";
		my @line=split/\s+/,$current_line;
		my $vertex_pos;
		#identify where to catch the name of vertex
		if(@line==5){$vertex_pos=3;}
		if(@line==6){$vertex_pos=3;}
		if(@line==8){$vertex_pos=4;}
		my $ident=$line[$vertex_pos];
		#catch the name of vertex
		$couplings_list[$ind]=$ident;
		$ind++;
		my @Shape;
		my $indShape=0;
		#run over terms before vertex name
		for(my $i=0;$i<$vertex_pos;$i++){
                        my @ordpattern;
			#run over particle.dat file to catch particle type
                        while(my $lignepart=<INFILE2>){
				if($lignepart=~/END/){$limitparticle=1;}
				if(!($lignepart=~/#/) && $limitparticle!=1){
					chomp($lignepart);
	                                my @cut=split /\s+/,$lignepart;
	                                if($line[$i] eq $cut[0] || $line[$i] eq $cut[1]){
					
	                                        $Shape[$indShape]=$cut[2];
	                                        $indShape++;
	                                }
				}
                        }
			#reset reading of particle.dat
			seek(INFILE2, my $curpos, 0);$limitparticle=0;#Reset of Infile2
			
		}
		#sort letters by order
                my @outsort= sort @Shape;
		
                my $pattern;
                for(my $k=0;$k<=$vertex_pos;$k++){$pattern=$pattern.$outsort[$k];}
		#give type number##############
                if($vertex_pos==3){
                	if($pattern eq ("FFV")){$Type=0;}
                        if($pattern eq ("VVV")){$Type=1;}
                        if($pattern eq ("FFS")){$Type=2;}
                        if($pattern eq ("SVV")){$Type=3;}
                        if($pattern eq ("SSV")){$Type=4;}
                        if($pattern eq ("SSS")){$Type=5;}
		
                        
		}
                if($vertex_pos==4){
                	if($pattern eq ("SSVV")){$Type=6;}
                        if($pattern eq ("SSSS")){$Type=7;}
                }
		if($vertex_pos==3 && $line[5] eq 'H'){
			if($pattern eq ("SVV")){$Type=8;}	
		}
		#######################################
		my $led=0;
		if(@couplings_list==1){
			if(lc($ident) eq  "gal" || lc($ident) eq  "gad" || lc($ident) eq  "gau" || lc($ident) eq  "gwf" || lc($ident) eq  "gzn" || lc($ident) eq  "gzl" || lc($ident) eq  "gzd" || lc($ident) eq  "gzu" || lc($ident) eq  " gw" || lc($ident) eq  "gwwa" || lc($ident) eq  "gwwz"|| lc($ident) eq  "gwfc" || lc($ident) eq  "gwfs" || lc($ident) eq  "gwfm"|| lc($ident) eq  "gwwh" || lc($ident) eq  "gzzh" || lc($ident) eq  "gwwhh" || lc($ident) eq  "gzzhh" || lc($ident) eq  "ghhh" || lc($ident) eq  "ghhhh" || lc($ident) eq  "ghtop" || lc($ident) eq  "ghbot" || lc($ident) eq  "ghtau" || lc($ident) eq  "ghcha" || lc($ident) eq  "gg"){$led=1;}
		}
		for(my $u=0;$u<@couplings_list-1;$u++){
			if((lc($ident) eq lc($couplings_list[$u])) || lc($ident) eq  "gal" || lc($ident) eq  "gad" || lc($ident) eq  "gau" || lc($ident) eq  "gwf" || lc($ident) eq  "gzn" || lc($ident) eq  "gzl" || lc($ident) eq  "gzd" || lc($ident) eq  "gzu" || lc($ident) eq  " gw" || lc($ident) eq  "gwwa" || lc($ident) eq  "gwwz"|| lc($ident) eq  "gwfc" || lc($ident) eq  "gwfs" || lc($ident) eq  "gwfm"|| lc($ident) eq  "gwwh" || lc($ident) eq  "gzzh" || lc($ident) eq  "gwwhh" || lc($ident) eq  "gzzhh" || lc($ident) eq  "ghhh" || lc($ident) eq  "ghhhh" || lc($ident) eq  "ghtop" || lc($ident) eq  "ghbot" || lc($ident) eq  "ghtau" || lc($ident) eq  "ghcha" || lc($ident) eq  "gg"){$led=1;}#led=1 means that the vertex is already declared
		}
		if($led==0){
			if($Type==1){
				if($tag0!=0){$VT[0]=$VT[0].",";}
				$VT[0]=$VT[0]."$ident";
				if($new_vertex_flag==1){
					my $temp;
				        print OUTFILE6 "      write(6,20) '$ident  =  ', $ident\n";
					print OUTFILE8 "      write(1,20) '$ident ', $ident\n";
					print OUTFILE1 "      $ident=1d0";
					print OUTFILE1 "\n";
				}
				$tag0=1;
			}
			else{
				if($Type!=2 && $Type!=0 && $Type!=8){
					if($tag1!=0){$VT[1]=$VT[1].",";}
					$VT[1]=$VT[1]."$ident";
					#print "type: $Type $tag1 $VT[1]\t";
					if($new_vertex_flag==1){
						print OUTFILE6 "      write(6,20) '$ident  =  ', $ident\n";
						print OUTFILE8 "      write(1,20) '$ident ', $ident\n";
			                        print OUTFILE1 "      $ident=dcmplx(1d0,Zero)";
			                        print OUTFILE1 "\n";
					}
					$tag1=1;
	
				}
				else{
					#print "-->$VT[1]\t";
					if($tag1!=0){$VT[1]=$VT[1].",";}
					#print "-->$VT[1]\t";
					$VT[1]=$VT[1]."$ident(2)";
					#print "type: $Type $tag1 $VT[1]\t";
					if($new_vertex_flag==1){
		                                print OUTFILE1 "      $ident(1)=dcmplx(1d0,Zero)";
		                                print OUTFILE1 "\n";
						print OUTFILE6 "      write(6,11) '$ident(L)  =  ', $ident(1),'$ident(R)  =  ', $ident(2)\n";
						print OUTFILE8 "      write(1,11) '$ident ', $ident(1),' ', $ident(2)\n";
						print OUTFILE1 "      $ident(2)=dcmplx(1d0,Zero)";
		                                print OUTFILE1 "\n";
					}
					$tag1=1;
	
				}
			}	
		}
	#print "\n";
	}
	        if($current_line=~/USRVertex/){$new_vertex_flag=1;}

}
my $N=110;


#Writing couplings in coupl.inc...

my @entire;
$entire[0]=$prefix[0].$VT[0];
$entire[1]=$prefix[1].$VT[1];
for(my $j=0;$j<2;$j++){
        if(length($entire[$j])>$N){
                for(my $i=0;$i<ceil(length($entire[$j])/$N);$i++){
                        print OUTFILE2 substr($entire[$j],$i*$N,$N);print OUTFILE2 "\n";
                        if($i<ceil(length($entire[$j])/$N)-1){print OUTFILE2 "     +";}
                }
        }
        elsif(length($entire[$j])>25){
                print OUTFILE2  "$entire[$j]\n";
        }
	

}
my @commonblock;
$commonblock[0]="      common /DP_COUPL/ ";
$commonblock[1]="      common /DC_COUPL/ ";
for(my $m=0;$m<2;$m++){
	$commonblock[$m]=$commonblock[$m].$VT[$m];
	$commonblock[$m]=~s/\(2\)//g;
	
}

for(my $k=0;$k<2;$k++){
if(length($commonblock[$k])>$N){
                for(my $i=0;$i<ceil(length($commonblock[$k])/$N);$i++){
                        print OUTFILE2 substr($commonblock[$k],$i*$N,$N);print OUTFILE2 "\n";
                        if($i<ceil(length($commonblock[$k])/$N)-1){print OUTFILE2 "     +";}
                }
        }
        elsif(length($entire[$k])>25){
                print OUTFILE2  "$commonblock[$k]\n";
        }
}

print OUTFILE1 "\n";
while(<INFILE5>){print OUTFILE1;}

#####Masses and widths....

my @ident;
my @pdg;
my $index=0;
my $tag=0;
my $new_particle_flag=0;
while(my $current_line=<INFILE2>){
        if($new_particle_flag==1 && !($current_line=~/^\s+/) && !($current_line=~/^#/)){
		chomp($current_line);
                my @line=split/\s+/,$current_line;
                $ident[2*$index]=$line[4];
		$pdg[2*$index]=$line[8];
		$ident[2*$index+1]=$line[5];
		$pdg[2*$index+1]=$line[8];
		
		#print OUTFILE3 "MASS                $line[8] $line[4]\n";
		#print OUTFILE3 "DECAY               $line[8] $line[5]\n";
		$index++;
        }
	if($current_line=~/EXTENSION/){$new_particle_flag=1;}
	if($current_line=~/END/){$new_particle_flag=0;}
}
my @sentence;
$prefix[2]="      double precision ";
for(my $i=0;$i<$index;$i++){
		my @tague;
		$tague[0]=0;
		$tague[1]=0;
		if($ident[2*$i] eq 'ZERO' ||  $ident[2*$i] eq 'BMASS' || $ident[2*$i] eq 'TMASS' ||  $ident[2*$i] eq 'LMASS' || $ident[2*$i] eq 'ZMASS' || $ident[2*$i] eq 'WMASS' ||  $ident[2*$i] eq 'HMASS'){$tague[0]=1;}
		if($ident[2*$i+1] eq 'ZERO' || $ident[2*$i] eq 'TWIDTH' ||  $ident[2*$i] eq 'AWIDTH' || $ident[2*$i] eq 'ZWIDTH' || $ident[2*$i] eq 'WWIDTH' ||  $ident[2*$i] eq 'HWIDTH'){$tague[1]=1;}
		#if($tag!=0){$sentence[0]=$sentence[0].",";$sentence[1]=$sentence[1].",";}
		if($tague[0]==0){$sentence[0]=$sentence[0].$ident[2*$i].",";}
		if($tague[1]==0){$sentence[1]=$sentence[1].$ident[2*$i+1].",";}
		if($tague[0]!=0 || $tague[1]!=0){$tag=1;}
}
$sentence[0]=~s/.{1}$//;
$sentence[1]=~s/.{1}$//;
$entire[0]=$prefix[2].$sentence[0];
$entire[1]=$prefix[2].$sentence[1];
for(my $j=0;$j<2;$j++){
	if(length($entire[$j])>$N){
		for(my $i=0;$i<ceil(length($entire[$j])/$N);$i++){
			print OUTFILE2 substr($entire[$j],$i*$N,$N);print OUTFILE2 "\n";
			if($i<ceil(length($entire[$j])/$N)-1){print OUTFILE2 "     +";}
		}
	}
	elsif(length($sentence[$j])>0){
		print OUTFILE2  "$entire[$j]\n";
	}

}

$commonblock[0]="      common /USR_MASS/  ";
$commonblock[1]="      common /USR_DECAY/ ";
$commonblock[0]=$commonblock[0].$sentence[0];
$commonblock[1]=$commonblock[1].$sentence[1];
for(my $k=0;$k<2;$k++){
	if(length($commonblock[$k])>$N){
                	for(my $i=0;$i<ceil(length($commonblock[$k])/$N);$i++){
                        	print OUTFILE2 substr($commonblock[$k],$i*$N,$N);print OUTFILE2 "\n";
                        	if($i<ceil(length($commonblock[$k])/$N)-1){print OUTFILE2 "     +";}
                	}
        	}
        	elsif(length($sentence[$k])>0){
                	print OUTFILE2  "$commonblock[$k]\n";
        	}
}

####################################################
while(<INFILE10b>){print OUTFILE6;}
my $massdefault=100;
my $decaydefault=1;
for(my $i=0;$i<$index;$i++){
	if($massfile==1){
		seek(INFILE17, my $curpos, 0);#Reset of Infile2
		while(my $massline=<INFILE17>){
			my @splittedmassline=split /=/,$massline;
			if($ident[2*$i] eq $splittedmassline[0]){$massdefault=$splittedmassline[1];}
		}
	}
	printf OUTFILE4 " %9i   %16.8e   # %s\n",$pdg[2*$i],$massdefault,$ident[2*$i];
	my $sentence= "      write(6,24) '$ident[2*$i]  =  ', $ident[2*$i], '$ident[2*$i+1]    = ', $ident[2*$i+1]";
	if(length($sentence)>$N){
                     for(my $i=0;$i<ceil(length($sentence)/$N);$i++){
                                print OUTFILE6 substr($sentence,$i*$N,$N);print OUTFILE6 "\n";
                                if($i<ceil(length($sentence)/$N)-1){print OUTFILE6 "     +";}
                        }
        }
        else{
        	print OUTFILE6  "$sentence\n";
        }
	my $min=lc($ident[2*$i]);
	print OUTFILE7 "       call set_it(n,ivalue,value,name,$pdg[2*$i],bn,$min,100d0)\n";
}
while(<INFILE16b>){print OUTFILE8;}
while(<INFILE10c>){print OUTFILE6;}
while(<INFILE13>){print OUTFILE7;}
while(<INFILE8b>){print OUTFILE4;}
for(my $i=0;$i<$index;$i++){
	if($massfile==1){
		seek(INFILE17, my $curpos, 0);
		while(my $massline=<INFILE17>){
			my @splittedmassline=split /=/,$massline;
			if($ident[2*$i+1] eq $splittedmassline[0]){$decaydefault=$splittedmassline[1];}
		}
	}
	printf OUTFILE4 "DECAY %9i   %16.8e   # %s\n",$pdg[2*$i+1],$decaydefault,$ident[2*$i+1];
	my $min=lc($ident[2*$i+1]);
        print OUTFILE7 "       call set_it(n,ivalue,value,name,$pdg[2*$i],bn,$min,1d0)\n";
}
while(<INFILE8c>){print OUTFILE4;}
while(<INFILE14>){print OUTFILE7;}



seek(INFILE2, my $curpos, 0);#Reset of Infile2


###################Writing MINPAR ident_card.dat part ###############################
$index=1;

print OUTFILE4 "BLOCK MGUSER\n";
while(my $current_line=<INFILE3>){
	unless($current_line=~/^\s+/){
		chomp($current_line);
		my @line=split /#/,$current_line;
		my $min=lc($line[0]);
		print OUTFILE6 "      write(6,24) '$line[$0]  =  ', $line[0]\n";
		print OUTFILE3 "MINPAR              "."$index $line[0]\n";		
		printf OUTFILE4 " %9i   %16.8e   # %s,%s\n",$index,0,$line[0],$line[1];
#		print OUTFILE4 "	$index	0.00000000E+00 # $line[1]\n";
	        print OUTFILE7 "       call set_it(n,ivalue,value,name,$index,bn,$min,0d0)\n";
		$index++;
	}	
}
while(<INFILE13>){print OUTFILE7;}
while(<INFILE15>){print OUTFILE7;}
while(<INFILE10d>){print OUTFILE6;}
###################Writing input.inc#####################

seek(INFILE3, my $curpos10, 0);#Reset of Infile3
$index=0;
$tag=0;
my $buffer;
while(my $current_line=<INFILE3>){
        unless($current_line=~/^\s+/){
		if($tag==1){$buffer=$buffer.",";}
		chomp($current_line);
                my @line=split /\#/,$current_line;
		$buffer=$buffer.$line[0];
		$tag=1;
	}
}
my $entire=$prefix[0].$buffer;
if(length($entire)>$N){
	for(my $i=0;$i<ceil(length($entire)/$N);$i++){
		print OUTFILE2 substr($entire,$i*$N,$N);print OUTFILE2 "\n";
	                        if($i<ceil(length($entire)/$N)-1){print OUTFILE2 "     +";}
        }
 }
elsif(length($buffer)>0){
	print OUTFILE5  "$entire\n";
}

$commonblock[0]="      common /USR_PARAM/  ";
$commonblock[0]=$commonblock[0].$buffer;
#print $commonblock[0];
if(length($commonblock[0])>$N){
                for(my $i=0;$i<ceil(length($commonblock[0])/$N);$i++){
                        print OUTFILE5 substr($commonblock[0],$i*$N,$N);print OUTFILE5 "\n";
                        if($i<ceil(length($commonblock[0])/$N)-1){print OUTFILE5 "     +";}
                }
        }
elsif(length($buffer)>0){
	print OUTFILE5  "$commonblock[0]\n";
}




	




########################################################################


close INFILE1;
close INFILE2;
close INFILE3;
close INFILE4;
close INFILE5;
close INFILE6;
close INFILE8a;
close INFILE8b;
close INFILE8c;
close INFILE9;
close INFILE10a;
close INFILE10b;
close INFILE10c;
close INFILE10b;
close INFILE11;
close INFILE13;
close INFILE14;
close INFILE15;
close INFILE16a;
close INFILE16b;

close OUTFILE1;
close OUTFILE2;
close OUTFILE3;
close OUTFILE4;
close OUTFILE5;
close OUTFILE6;
close OUTFILE7;

	print "File creation completed. Please edit couplings.f and param_card.dat\n";

system("./qnumbers.pl particles.dat param_card.dat");
