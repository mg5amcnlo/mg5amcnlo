
#include "Dire/Basics.h"

namespace Pythia8 {

//--------------------------------------------------------------------------

// Function to hash string into long integer.

ulong shash(const std::string& str) {
    ulong hash = 5381;
    for (size_t i = 0; i < str.size(); ++i)
        hash = 33 * hash + (unsigned char)str[i];
    return hash;
}

//--------------------------------------------------------------------------

// Helper function to calculate dilogarithm.

double polev(double x,double* coef,int N ) {
  double ans;
  int i;
  double *p;

  p = coef;
  ans = *p++;
  i = N;
    
  do
    ans = ans * x  +  *p++;
  while( --i );
    
  return ans;
}
  
//--------------------------------------------------------------------------

// Function to calculate dilogarithm.

double dilog(double x) {

  static double cof_A[8] = {
    4.65128586073990045278E-5,
    7.31589045238094711071E-3,
    1.33847639578309018650E-1,
    8.79691311754530315341E-1,
    2.71149851196553469920E0,
    4.25697156008121755724E0,
    3.29771340985225106936E0,
    1.00000000000000000126E0,
  };
  static double cof_B[8] = {
    6.90990488912553276999E-4,
    2.54043763932544379113E-2,
    2.82974860602568089943E-1,
    1.41172597751831069617E0,
    3.63800533345137075418E0,
    5.03278880143316990390E0,
    3.54771340985225096217E0,
    9.99999999999999998740E-1,
  };

  if( x >1. ) {
    return -dilog(1./x)+M_PI*M_PI/3.-0.5*pow2(log(x));
  }

  x = 1.-x;
  double w, y, z;
  int flag;
  if( x == 1.0 )
    return( 0.0 );
  if( x == 0.0 )
    return( M_PI*M_PI/6.0 );
    
  flag = 0;
    
  if( x > 2.0 ) {
    x = 1.0/x;
    flag |= 2;
  }
    
  if( x > 1.5 ) {
    w = (1.0/x) - 1.0;
    flag |= 2;
  }
    
  else if( x < 0.5 ) {
    w = -x;
    flag |= 1;
  }
    
  else
    w = x - 1.0;
    
  y = -w * polev( w, cof_A, 7) / polev( w, cof_B, 7 );
    
  if( flag & 1 )
    y = (M_PI * M_PI)/6.0  - log(x) * log(1.0-x) - y;
    
  if( flag & 2 ) {
    z = log(x);
    y = -0.5 * z * z  -  y;
  }
    
  return y;

}

double lABC(double a, double b, double c) { return pow2(a-b-c) - 4.*b*c;}
double bABC(double a, double b, double c) { 
  double ret = 0.;
  if      ((a-b-c) > 0.) ret = sqrt(lABC(a,b,c));
  else if ((a-b-c) < 0.) ret =-sqrt(lABC(a,b,c));
  else                   ret = 0.;
  return ret; }
double gABC(double a, double b, double c) { return 0.5*(a-b-c+bABC(a,b,c));}

int puppybort( string input, int iPuppy) {
  srand (time(NULL));
  if (iPuppy == 0) iPuppy = rand() % 7 + 1;
  cout << "\nSomething went terribly wrong in " << input << endl;
  cout << "\nMaybe this...\n" << endl;
  if (iPuppy == 1) {
    cout << "  __      _" << endl
         << "o'')}____//" << endl
         << " `_/      )" << endl
         << " (_(_/-(_/" << endl;
  } else if (iPuppy == 2) {
    cout << "    ___" << endl
         << " __/_  `.  .-\"\"\"-." << endl
         << " \\_,` | \\-'  /   )`-')" << endl
         << "  \"\") `\"`    \\  ((`\"`" << endl
         << " ___Y  ,    .'7 /|" << endl
         << "(_,___/...-` (_/_/" << endl;
  } else if (iPuppy == 3) {
    cout << "       /^-^\\         /^-----^\\" << endl
         << "      / o o \\        V  o o  V" << endl
         << "     /   Y   \\        |  Y  |" << endl
         << "     V \\ v / V         \\ Q /" << endl
         << "       / - \\           / - \\" << endl
         << "      /    |           |    \\" << endl
         << "(    /     |           |     \\     )" << endl
         << " ===/___) ||           || (___\\====" << endl;
  } else if (iPuppy == 4) {
    cout << "_     /)---(\\          /~~~\\" << endl
         << "\\\\   (/ . . \\)        /  .. \\" << endl
         << " \\\\__)-\\(*)/         (_,\\  |_)" << endl
         << " \\_       (_         /   \\@/    /^^^\\" << endl
         << " (___/-(____) _     /      \\   / . . \\" << endl
         << "              \\\\   /  `    |   V\\ Y /V" << endl
         << "               \\\\/  \\   | _\\    / - \\" << endl
         << "                \\   /__'|| \\\\_  |    \\" << endl
         << "                 \\_____)|_).\\_).||(__V" << endl;
  } else if (iPuppy == 5) {
    cout << "              ,-~~~~-," << endl
         << "        .-~~~;        ;~~~-." << endl
         << "       /    /          i\\    \\" << endl
         << "      {   .'{  O    O  }'.   }" << endl
         << "       `~`  { .-~~~~-. }  `~`" << endl
         << "            ;/        \\;" << endl
         << "           /'._  ()  _.'\\" << endl
         << "          /    `~~~~`    \\" << endl
         << "         ;                ;" << endl
         << "         {                }" << endl
         << "         {     }    {     }" << endl
         << "         {     }    {     }" << endl
         << "         /     \\    /     \\" << endl
         << "        { { {   }~~{   } } }" << endl
         << "         `~~~~~`    `~~~~~`" << endl
         << "           (`\"=======\"`)" << endl
         << "           (_.=======._)" << endl;
  } else if (iPuppy == 6) {
    cout << "                            ..,,,,,,,,,.. " << endl
         << "                     .,;%%%%%%%%%%%%%%%%%%%%;,. " << endl
         << "                   %%%%%%%%%%%%%%%%%%%%////%%%%%%, .,;%%;, " << endl
         << "            .,;%/,%%%%%/////%%%%%%%%%%%%%%////%%%%,%%//%%%, " << endl
         << "        .,;%%%%/,%%%///%%%%%%%%%%%%%%%%%%%%%%%%%%%%,////%%%%;, " << endl
         << "     .,%%%%%%//,%%%%%%%%%%%%%%%%@@%a%%%%%%%%%%%%%%%%,%%/%%%%%%%;, " << endl
         << "   .,%//%%%%//,%%%%///////%%%%%%%@@@%%%%%%///////%%%%,%%//%%%%%%%%, " << endl
         << " ,%%%%%///%%//,%%//%%%%%///%%%%%@@@%%%%%////%%%%%%%%%,/%%%%%%%%%%%%% " << endl
         << ".%%%%%%%%%////,%%%%%%%//%///%%%%@@@@%%%////%%/////%%%,/;%%%%%%%%/%%% " << endl
         << "%/%%%%%%%/////,%%%%///%%////%%%@@@@@%%%///%%/%%%%%//%,////%%%%//%%%' " << endl
         << "%//%%%%%//////,%/%a`  'a%///%%%@@@@@@%%////a`  'a%%%%,//%///%/%%%%% " << endl
         << "%///%%%%%%///,%%%%@@aa@@%//%%%@@@@S@@@%%///@@aa@@%%%%%,/%////%%%%% " << endl
         << "%%//%%%%%%%//,%%%%%///////%%%@S@@@@SS@@@%%/////%%%%%%%,%////%%%%%' " << endl
         << "%%//%%%%%%%//,%%%%/////%%@%@SS@@@@@@@S@@@@%%%%/////%%%,////%%%%%' " << endl
         << "`%/%%%%//%%//,%%%///%%%%@@@S@@@@@@@@@@@@@@@S%%%%////%%,///%%%%%' " << endl
         << "  %%%%//%%%%/,%%%%%%%%@@@@@@@@@@@@@@@@@@@@@SS@%%%%%%%%,//%%%%%' " << endl
         << "  `%%%//%%%%/,%%%%@%@@@@@@@@@@@@@@@@@@@@@@@@@S@@%%%%%,/////%%' " << endl
         << "   `%%%//%%%/,%%%@@@SS@@SSs@@@@@@@@@@@@@sSS@@@@@@%%%,//%%//%' " << endl
         << "    `%%%%%%/  %%S@@SS@@@@@Ss` .,,.    'sS@@@S@@@@%'  ///%/%' " << endl
         << "      `%%%/    %SS@@@@SSS@@S.         .S@@SSS@@@@'    //%%' " << endl
         << "               /`S@@@@@@SSSSSs,     ,sSSSSS@@@@@' " << endl
         << "             %%//`@@@@@@@@@@@@@Ss,sS@@@@@@@@@@@'/ " << endl
         << "           %%%%@@00`@@@@@@@@@@@@@'@@@@@@@@@@@'//%% " << endl
         << "       %%%%%%a%@@@@000aaaaaaaaa00a00aaaaaaa00%@%%%%% " << endl
         << "    %%%%%%a%%@@@@@@@@@@000000000000000000@@@%@@%%%@%%% " << endl
         << " %%%%%%a%%@@@%@@@@@@@@@@@00000000000000@@@@@@@@@%@@%%@%% " << endl
         << "%%%aa%@@@@@@@@@@@@@@0000000000000000000000@@@@@@@@%@@@%%%% " << endl
         << "%%@@@@@@@@@@@@@@@00000000000000000000000000000@@@@@@@@@%%%%%"  << endl;
  } else if (iPuppy == 7) {
    cout << "                          _..___" << endl
         << "                      _..xxxxxxxllllxxxx...___" << endl
         << "                   _.ssssssxxxxxxxxsssxxxxxxxxLlllxxx..._" << endl
         << "               _.ssssSSSSsssssSSSSSSSSSsxxxxxxxXxxxXxxxXxlll++._" << endl
         << "          _.sdSSSSSSSSSSSSSSSSSSSSSSsxxxxxxxXxxxXxxxXxxxXxxxxx+++." << endl
         << "       .dSSSSS$$$$$S$$SSSSSSS$$888SsxxxXxxxxXxxxXxxxxXxxxXxxxxxxxxx." << endl
         << "      j$$$$SS$$$$$$$$$$$S$SS$$888sxxxxXxxxxXxxxxXxxxxXxxXxxXxxxxxxxxx." << endl
         << "      $$$$SS$$$$$$$$$$$$$$$$$$88xxxxXXxxxXxxxxxXXxxxxXxxxXxxxXxxxxxxxx." << endl
         << "      Y$$$$SS$$$$$$$$$$$$$$$$8SsxxxxXxxXXxxxxxXXxxxxxxXxxxXxxxXxxxxS$xxh." << endl
         << "       `$$$S$S$$$$$$$$$$$$$$$SsxxxxxxxxxxxxxxxXxxxxxxxXxxxXxxxXXxxxS$$Sxx." << endl
         << "        .$$$SS$$$$$$$$$$$$$$SsxxxxxxxxxxxxxxxXxxxxxxxxXxxxXXxxXxxXxsS$$$xx." << endl
         << "        xSS$$$S$$$$$$$$$$$$SsxXxxxxxxxxxxxxxXxxxxxxxxxxXxxxxXXXxxxXxS$$$$xx." << endl
         << "       .+xSS$$$$$$$$$$$$$$$SxxxxxxxXxxxxxxxxxXXXxxxxxxxXXxxxXxxxxxxxsS$$$$xx." << endl
         << "      .++++SS$$$$$$$$$$$$$$SxXxxxxxxxxxxxxxxxxxXXXxxxxxxXxxxXxxxssSxsS$$$$$xx" << endl
         << "     .+++++xxSS$$$$$$$$$$$SxxxxxXxxxxxxxxxssSSxxxxXxxxxxxxxXXxxsSx$Ssx$$$$$Sx." << endl
         << "    .++++xxxxxxSS$$$$$$$$SxxxxsxxxxxxxxssS$$$SSsxxxxSsxxxssxxxsSsxS$SsS$$$$$Xx." << endl
         << "   .++++x++xxxxxxSS$$$$SxxxxxsSssxxxxxxxxsS$$$SssxxsSSsxsSSssSSsxxS$$SsS$$$$$xx" << endl
         << "   ++++++x+x++xxxxxxxxxxxxxssS$$SssssssssSS$$$$$SssSSSSsS$$SSSsxxsS$$SssS$$$$xx" << endl
         << "  .+++x++xxxx++xxxxxxxXxxxxsS$$$$SSSyysSS$$$$$$$$$$$$$$$$$$$$$$SSyS$$$$S$$$$$xx" << endl
         << " .++++++x+xx+x++xxxxxxxXxxsS$$$$$$d8,,n88b$$S$$$$$SSS$$$$$$$SS$$d,8b$$$$$$$$$xx" << endl
         << ".++++++x+xxXx++xxxxxxxxXxxxsS$$S$$$Y##880P$$$$$$$$SSSSSSSSSSSSSSY##P$$s'Y$$$$x'" << endl
         << "++++++++xxxxXx++xxxxxxxXxxxxsS$SS$$$$$$$$$$$$$$SSS$$$$d8####b$$SS$$$Ssx  `\"\"\"'" << endl
         << "+++++x+x+xxxXXx+x+xxxxxxXxxxxS$$$S$$$$$$$$S$$$SS$$$d#8$$8$8$8##8bSS$$sx" << endl
         << "++++x++xxxxxxXXXx+++xxxxxXxxxxS$S$$$$$$$$$$$SS$$d#8$$8$8$$8$8$$8#8b$Sx'" << endl
         << "+x+++xxxxxxxxXxXxx++xxxxxxXxxxxS$$$$$$S$$$SS$$888$$$$$8$8#8$$$$$8#P$S'" << endl
         << "+++x+xxxxxxxxxXxXxxx+xxxxxxXXxxsS$$$$$$$SS$$$88$$$$$$S$SS#SS$$$$$Y$$S" << endl
         << "+x++xxxxxxxxxxxXxXxxxxxxxxxxXXxssS$$$SSS$$$88$$$$$$SS$$SS#$SS$$$$$$$Sl" << endl
         << "+xxxxXxxxxxxxxxxXxXxxxxxxxxxXXxXxsSSS$$$$68$$$$$$$S$$SSS$#$$$$S$$$$$$$" << endl
         << "xxxxxxxxxXxxxxxxxxxXXXxxxxxxxxxXXxsS$$$$$9$$$$$$$$$$$$$$d#$$$$$$$$$$$$" << endl
         << "xxxxxxxxxxxxxxxxxxxxXxXXxxxxxxxxxXXS$$$$$$$$$$$$$$$$$$d88##6$$$$$$$$$'" << endl
         << "xxxxxxxXxxxXxxxxxxxxxxXxxxxxxxxxXXXSS$$$$$$$$$$$$$$$$d8888##b$$$$$$S'" << endl
         << "xxXxxxxxxxXxxxXxxxxxxxxxxXxxxxxxxXXXXS$$$$$$$$$$$$$d88888888##b$$$P'" << endl
         << "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxXXSS$$$$$$$$$$$$$Y8888P$$$$Y\"'" << endl
         << "xXxxxxXxxxxxxXxxxxxXxxxxxxxxxxxxxXxxxxXXsSS$$$$$$$$$$$$SSSS._" << endl
         << "xxxxxxxxxxxxxxxXxxxxxxxxxxxxxxxxxxxXxxxxxXXSSSSSSSSSSSssxx+++;" << endl
         << "xxxxxxxxxxxxxxxxxxXxXxxxXxxxxxxxx+xxXxxxxxxxsssssssssxxxxxx+'" << endl
         << "xxxxxXxxxxxxXxxxxxxxxxXxxXxxxxxx+xxxxxXXxxxxxxxxxxxxxxxxxxx." << endl
         << "xxXxxxxxxxxxxxxxxxXxxxxxxXxxxXxxxx+xxx+xxXXxxxxxxxxxxxxxxXxx." << endl
         << "xxxxxxxxxxxxxxxxxxxxxxXxxxxxxxxxxxx+x+xxxxxXXxxXxxxxxxxXxx++xx" << endl
         << "xxxxxxxxxxXxxxxxxxXxxxxxxxxxxxxxxx+xxx+xx+xxxXxxxxxxxxX+++++xx" << endl
         << "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxXxxxXxxxX++x++++." << endl
         << "xxxxXxxxxxxxxxxxxxXxxxxxxxxxxxxxxxx+xxxx++xxxxxXxxxXX+++++++++." << endl
         << "xxxxxxxxxxxxxXxxxxxxxxxXxxxxxxxxxxxxxxxxxxx+xxxx+xxX+x++++++++x." << endl
         << "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx+xxxxx++xxxXxx+++++xx++xx." << endl
         << "xxxxxxxxXxxxxxxxxxxxxxxxxxxxxxxxXxxxxxxxxxxxxxx+xXxx+x++++x+xxxxx." << endl
         << "xxxxxxxxxxxxXxxxxxXxxxxxxxxXxxxxxxxxxxxxxxxxXxxxXXxxxxx++xxxxx+xxxx." << endl
         << "xxxxXxxxxxxxxxxxxxxxxxxxxxxxxxxxxXxxxxxxXxxxxxxxXxxxxx++++++++xxx+xxxx." << endl
         << "xxxxxxxxxxxxxxxxxxxxxxXxxxxxxxxxxxxxxxxxxxxxxxxxXxxxxxxx++++++xx+xxx+xxx." << endl;
  } else {
    cout << "No puppies found. All hope is lost." << endl;
  }
  cout << "\n... will help?\n" << endl;
  abort();
}

//==========================================================================

}
