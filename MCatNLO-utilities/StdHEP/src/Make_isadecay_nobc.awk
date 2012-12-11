#
# Remove lines from isajet decay file for
# b and c hadrons which decay weakly.
#
# To use this file:
# awk -f Make_isadecay_nobc.awk isadecay.dat > isadecay_nobc.dat
#
#
{keep = 1}
$1 ==   "16," {keep = 0}   # tau
$1 ==  "150," {keep = 0}   # B+
$1 ==  "250," {keep = 0}   # B_d^0
$1 ==  "350," {keep = 0}   # B_s^0
$1 == "2150," {keep = 0}   # \Lambda_b^0
$1 == "3150," {keep = 0}   # \Xi_b^0
$1 == "3250," {keep = 0}   # \Xi_b^-
$1 == "3350," {keep = 0}   # \Omega_b

$1 ==  "140," {keep = 0}   # D~0
$1 ==  "240," {keep = 0}   # D-
$1 ==  "340," {keep = 0}   # D_s^-
$1 == "2140," {keep = 0}   # \Lambda_c^0
$1 == "3140," {keep = 0}   # \Xi_c^0
$1 == "3240," {keep = 0}   # \Xi_c^-
$1 == "3340," {keep = 0}   # \Omega_c

# Should also consider removing these. 
# But I do not know if qq knows how to decay
# these particles.
# 
#$1 == "3440," {keep = 0}   # \Omega_cc
#$1 == "4441," {keep = 0}   # \Omega_ccc
#$1 == "3550," {keep = 0}   # \Omega_bb
#$1 == "5551," {keep = 0}   # \Omega_bbb
#$1 == " 450," {keep = 0}   # B_c
#$1 == "4350," {keep = 0}   # \Omega_bc
#$1 == "4450," {keep = 0}   # \Omega_bcc
#$1 == "4550," {keep = 0}   # \Omega_bbc

keep == 1 {print}

