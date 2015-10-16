################################################################################
#
# Copyright (c) 2009 The MadGraph5_aMC@NLO Development team and Contributors
#
# This file is a part of the MadGraph5_aMC@NLO project, an application which 
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph5_aMC@NLO license which should accompany this 
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################

import random
if __name__=='__main__':
    # Make sure paths are accessible
    import os
    import sys
    root_path = os.path.split(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))[0]
    sys.path.insert(0, root_path)

import madgraph.various.misc as misc
from madgraph import MadGraph5Error

#===============================================================================
# MadLoopBannerStyles
#===============================================================================
class MadLoopBannerStyles(object):
    """ A container class to specify all banner styles. """
    # a dictionary contaning all banner styles
    ordered_style_keys = ['classic','classic2','classic3','big','funky',
                          'curly','bubbles','mario','wiggly',
                          'printed','fast','isometric']
    styles = {}
    styles['classic'] = \
r"""
  __  __           _ _                       
 |  \/  |         | | |                      
 | \  / | __ _  __| | |     ___   ___  _ __  
 | |\/| |/ _` |/ _` | |    / _ \ / _ \| '_ \ 
 | |  | | (_| | (_| | |___| (_) | (_) | |_) |
 |_|  |_|\__,_|\__,_|______\___/ \___/| .__/ 
                                      | |    
                                      |_|   
   %(version)s
   %(ref)s
"""

    styles['classic2'] = \
r"""                                                                 
                               ,,                                           
`7MMM.     ,MMF'             `7MM  `7MMF'                                   
  MMMb    dPMM                 MM    MM                                     
  M YM   ,M MM   ,6"Yb.   ,M""bMM    MM         ,pW"Wq.   ,pW"Wq.`7MMpdMAo. 
  M  Mb  M' MM  8)   MM ,AP    MM    MM        6W'   `Wb 6W'   `Wb MM   `Wb 
  M  YM.P'  MM   ,pm9MM 8MI    MM    MM      , 8M     M8 8M     M8 MM    M8 
  M  `YM'   MM  8M   MM `Mb    MM    MM     ,M YA.   ,A9 YA.   ,A9 MM   ,AP 
.JML. `'  .JMML.`Moo9^Yo.`Wbmd"MML..JMMmmmmMMM  `Ybmd9'   `Ybmd9'  MMbmmd'  
                                                                   MM       
                                                                 .JMML.
%(versionref)s
"""

    styles['classic3'] = \
r"""
  __  __           _ _                      
 |  \/  | __ _  __| | |    ___   ___  _ __  
 | |\/| |/ _` |/ _` | |   / _ \ / _ \| '_ \ 
 | |  | | (_| | (_| | |__| (_) | (_) | |_) |
 |_|  |_|\__,_|\__,_|_____\___/ \___/| .__/ 
                                     |_|    
        
%(version)s
%(ref)s
"""

    styles['big'] = \
r"""
88b           d88                       88  88                                                  
888b         d888                       88  88                                                  
88`8b       d8'88                       88  88                                                  
88 `8b     d8' 88  ,adPPYYba,   ,adPPYb,88  88            ,adPPYba,    ,adPPYba,   8b,dPPYba,   
88  `8b   d8'  88  ""     `Y8  a8"    `Y88  88           a8"     "8a  a8"     "8a  88P'    "8a  
88   `8b d8'   88  ,adPPPPP88  8b       88  88           8b       d8  8b       d8  88       d8  
88    `888'    88  88,    ,88  "8a,   ,d88  88           "8a,   ,a8"  "8a,   ,a8"  88b,   ,a8"  
88     `8'     88  `"8bbdP"Y8   `"8bbdP"Y8  88888888888   `"YbbdP"'    `"YbbdP"'   88`YbbdP"'   
                                                                                   88           
                                                                                   88
        %(versionref)s
"""

    styles['funky'] = \
r"""
                                                .-'''-.        .-'''-.                           
                          _______      .---.   '   _    \     '   _    \                         
 __  __   ___             \  ___ `'.   |   | /   /` '.   \  /   /` '.   \_________   _...._      
|  |/  `.'   `.            ' |--.\  \  |   |.   |     \  ' .   |     \  '\        |.'      '-.   
|   .-.  .-.   '           | |    \  ' |   ||   '      |  '|   '      |  '\        .'```'.    '. 
|  |  |  |  |  |    __     | |     |  '|   |\    \     / / \    \     / /  \      |       \     \ 
|  |  |  |  |  | .:--.'.   | |     |  ||   | `.   ` ..' /   `.   ` ..' /    |     |        |    |
|  |  |  |  |  |/ |   \ |  | |     ' .'|   |    '-...-'`       '-...-'`     |      \      /    ; 
|  |  |  |  |  |`" __ | |  | |___.' /' |   |                                |     |\`----'   .'
|__|  |__|  |__| .'.''| | /_______.'/  |   |                                |     | '-....-'`    
                / /   | |_\_______|/   '---'                               .'     '.             
                \ \._,\ '/                                               '-----------'           
                 `--'  `"                                                                        


  %(versionref)s
"""
        
    styles['curly'] = \
r"""
   __     __)              _           
  (, /|  /|        /)  ___/__)         
    / | / |  _   _(/  (, /   ________  
 ) /  |/  |_(_(_(_(_    /   (_)(_) /_)_
(_/   '                (_____   .-/    
                              )(_/    
   
%(version)s
%(ref)s
"""


    styles['keyboard'] = \
r"""
 ____ ____ ____ ____ ____ ____ ____ 
||M |||a |||d |||L |||o |||o |||p ||
||__|||__|||__|||__|||__|||__|||__||
|/__\|/__\|/__\|/__\|/__\|/__\|/__\|
        
%(version)s
%(ref)s
"""

    styles['bubbles'] = \
r"""
Oo      oO             o  o                        
O O    o o            O  O                         
o  o  O  O            o  o                         
O   Oo   O            o  o                         
O        o .oOoO' .oOoO  O       .oOo. .oOo. .oOo. 
o        O O   o  o   O  O       O   o O   o O   o 
o        O o   O  O   o  o     . o   O o   O o   O 
O        o `OoO'o `OoO'o OOoOooO `OoO' `OoO' oOoO' 
                                             O     
                                             o'    
%(version)s
%(ref)s
"""

    styles['mario'] = \
r"""
                  _   __                   
  /\/\   __ _  __| | / /  ___   ___  _ __  
 /    \ / _` |/ _` |/ /  / _ \ / _ \| '_ \ 
/ /\/\ \ (_| | (_| / /__| (_) | (_) | |_) |
\/    \/\__,_|\__,_\____/\___/ \___/| .__/ 
                                    |_|   
%(version)s
%(ref)s
"""

    styles['wiggly'] = \
r"""
   __    __       ____     ______     _____         ____       ____     _____   
   \ \  / /      (    )   (_  __ \   (_   _)       / __ \     / __ \   (  __ \  
   () \/ ()      / /\ \     ) ) \ \    | |        / /  \ \   / /  \ \   ) )_) ) 
   / _  _ \     ( (__) )   ( (   ) )   | |       ( ()  () ) ( ()  () ) (  ___/  
  / / \/ \ \     )    (     ) )  ) )   | |   __  ( ()  () ) ( ()  () )  ) )     
 /_/      \_\   /  /\  \   / /__/ /  __| |___) )  \ \__/ /   \ \__/ /  ( (      
(/          \) /__(  )__\ (______/   \________/    \____/     \____/   /__\     
                                                                        
%(versionref)s
"""

    styles['printed'] = \
r"""
__/\\\\____________/\\\\________________________/\\\___/\\\_______________________________________________________        
 _\/\\\\\\________/\\\\\\_______________________\/\\\__\/\\\_______________________________________________________       
  _\/\\\//\\\____/\\\//\\\_______________________\/\\\__\/\\\___________________________________________/\\\\\\\\\__      
   _\/\\\\///\\\/\\\/_\/\\\__/\\\\\\\\\___________\/\\\__\/\\\_________________/\\\\\________/\\\\\_____/\\\/////\\\_     
    _\/\\\__\///\\\/___\/\\\_\////////\\\_____/\\\\\\\\\__\/\\\_______________/\\\///\\\____/\\\///\\\__\/\\\\\\\\\\__    
     _\/\\\____\///_____\/\\\___/\\\\\\\\\\___/\\\////\\\__\/\\\______________/\\\__\//\\\__/\\\__\//\\\_\/\\\//////___   
      _\/\\\_____________\/\\\__/\\\/////\\\__\/\\\__\/\\\__\/\\\_____________\//\\\__/\\\__\//\\\__/\\\__\/\\\_________  
       _\/\\\_____________\/\\\_\//\\\\\\\\/\\_\//\\\\\\\/\\_\/\\\\\\\\\\\\\\\__\///\\\\\/____\///\\\\\/___\/\\\_________ 
        _\///______________\///___\////////\//___\///////\//__\///////////////_____\/////________\/////_____\///__________
        
   %(versionref)s
"""

    styles['fast'] = \
r"""
    __  ___          ____                    
   /  |/  /___ _____/ / /   ____  ____  ____ 
  / /|_/ / __ `/ __  / /   / __ \/ __ \/ __ \
 / /  / / /_/ / /_/ / /___/ /_/ / /_/ / /_/ /
/_/  /_/\__,_/\__,_/_____/\____/\____/ .___/ 
                                    /_/
%(version)s
%(ref)s
"""

    styles['isometric'] = \
r"""
      ___           ___          _____                        ___           ___           ___   
     /__/\         /  /\        /  /::\                      /  /\         /  /\         /  /\  
    |  |::\       /  /::\      /  /:/\:\                    /  /::\       /  /::\       /  /::\ 
    |  |:|:\     /  /:/\:\    /  /:/  \:\   ___     ___    /  /:/\:\     /  /:/\:\     /  /:/\:\
  __|__|:|\:\   /  /:/~/::\  /__/:/ \__\:| /__/\   /  /\  /  /:/  \:\   /  /:/  \:\   /  /:/~/:/
 /__/::::| \:\ /__/:/ /:/\:\ \  \:\ /  /:/ \  \:\ /  /:/ /__/:/ \__\:\ /__/:/ \__\:\ /__/:/ /:/ 
 \  \:\~~\__\/ \  \:\/:/__\/  \  \:\  /:/   \  \:\  /:/  \  \:\ /  /:/ \  \:\ /  /:/ \  \:\/:/  
  \  \:\        \  \::/        \  \:\/:/     \  \:\/:/    \  \:\  /:/   \  \:\  /:/   \  \::/   
   \  \:\        \  \:\         \  \::/       \  \::/      \  \:\/:/     \  \:\/:/     \  \:\   
    \  \:\        \  \:\         \__\/         \__\/        \  \::/       \  \::/       \  \:\  
     \__\/         \__\/                                     \__\/         \__\/         \__\/  

      %(versionref)s
"""

    @classmethod
    def get_raw_banner(cls,style):
        if style.lower()=='random':
            chosen = random.choice(cls.get_style_keys())
            return cls.styles[chosen]
        else:
            return cls.styles[style]

    @classmethod
    def get_style_keys(cls):
        return cls.ordered_style_keys

    @classmethod
    def get_MadLoop_Banner(cls, style='classic', color='blue', 
               top_frame_char = '=', bottom_frame_char = '=',
               left_frame_char = '{',right_frame_char = '}',
               print_frame=True, side_margin = 7, up_margin = 1):
        """ Writes out MadLoop banner."""
        
        colors = {'black':30,'red':31,'green':32,'yellow':33,
          'blue':34,'magenta':35,'cyan':36,'lightred':91,'lightgreen':92,
          'lightyellow':93,'lightblue':94,'lightmagenta':95,'lightcyan':96,
                                                           'white':97,'none':-1}
        
        if style.lower()=='random':
            color = random.choice(['blue','green','red'])
        
        reference = "Ref: arXiv:1103.0621v2, arXiv:1405.0301"
        version = "v%(version)s (%(date)s)"%misc.get_pkg_info()
        versionref = "%s, %s"%(version,reference)
        if style.lower() not in cls.get_style_keys()+['random']:
            raise MadGraph5Error('Incorrect style in MadLoopBanner. Must be'+\
          ' one of the following: %s'%str(cls.get_style_keys()+['random']))

        if isinstance(color,int):
            color_start ="char(27)//'[%im"%int
            color_end = "char(27)//'[0m"
        elif color.lower() in colors:
            if color.lower()=='none':
                color_start = ""
                color_end = ""
            else:
                color_start ="char(27)//'[%im"%colors[color.lower()]
                color_end = "char(27)//'[0m"                    
        else:
            raise MadGraph5Error('Incorrect color in MadLoopBanner. Must be and'+\
              ' intenger or one of the following: %s'%str(colors.keys()))

        def format_banner(banner):
            """ Format the raw banner text to give it a frame, colors and a 
            margin.""" 

            def fw(*args):
                """Fortran write line"""
                elems = []
                for arg in args:
                    if arg.startswith('char('):
                        elems.append("%s'"%arg)
                        continue
                    # Hard-set the single and double quotes in the text to
                    # make sure it is not processed by the FileWriter.
                    arg = arg.replace("'","'//char(39)//'")
                    arg = arg.replace('"',"'//char(34)//'")
                    if len(arg)>0:
                        elems.append("'%s'"%arg)
                return "write(*,*) %s"%("//".join(elems))
            
            banner_lines = banner.split('\n')
            formatted_lines = []

            # Determine the target width
            width = side_margin*2 + max(len(line) for line in banner_lines)
            if print_frame:
                width += 2
                
            # Print the upper frame
            if print_frame:
                formatted_lines.append(fw(" %s "%(top_frame_char*(width-2))))
            
            # Print the upper margin
            for i in range(up_margin):
                formatted_lines.append(fw("%(lside)s%(width)s%(rside)s"%
                  {'lside':left_frame_char if print_frame else '',
                   'rside':right_frame_char if print_frame else '',
                                                'width':' '*(width-2)}))
            
            # Now print the banner 
            for line in banner_lines:
                line_elements = []
                line_elements.append((left_frame_char if 
                                           print_frame else '')+' '*side_margin)
                # Colorize the logo
                line_elements.append(color_start)
                # Make sure to write the reference in black
                found = False
                for tag in [versionref, reference, version]:
                    if tag in line:
                        line_elements.extend([line[:line.index(tag)],
                                color_end,tag,color_start,
                                   line[line.index(tag)+len(tag):]+
                                       ' '*(width-2*(side_margin+1)-len(line))])
                        found = True
                        break
                if not found:
                    line_elements.append(line+
                                 ' '*(width-2*(side_margin+1)-len(line)))
                line_elements.append(color_end)
                line_elements.append(' '*side_margin+(right_frame_char 
                                                        if print_frame else ''))    
                formatted_lines.append(fw(*line_elements))

            # Print the lower margin (of height equal to up margin)
            for i in range(up_margin):
                formatted_lines.append(fw("%(lside)s%(width)s%(rside)s"%
                  {'lside':left_frame_char if print_frame else '',
                   'rside':right_frame_char if print_frame else '',
                                                        'width':' '*(width-2)}))
                                
            # Print the lower frame
            if print_frame:
                formatted_lines.append(fw(" %s "%(bottom_frame_char*(width-2))))
            
            return '\n'.join(formatted_lines)
            
        # Now we define the raw banner text for each style:
        
        return format_banner(
                 cls.get_raw_banner(style.lower())
                 %{'versionref':versionref, 'ref':reference, 'version':version})

# Below we have a small standalone code to test the MadLoop Banner output
if __name__=='__main__':
    import madgraph.iolibs.file_writers as writers
    import os
    import copy
    pjoin = os.path.join
    writer = writers.FortranWriter('test_ML_banner.f')

    styles = copy.copy(MadLoopBannerStyles.get_style_keys())
    styles.append('random')
    # Edit the line above and select here a subset of the available styles to 
    # show. Possibilities are:
    # ['classic','classic2','classic3','big','funky',
    # 'curly','keyboard','bubbles','mario','wiggly',
    # 'printed','fast','isometric','random']
    # styles = ['funky']
    
    f_code = ""
    
    for style in styles:
        f_code += "\nwrite(*,*) ''\nwrite(*,*) 'Style %s with default options.'\n"%style
        f_code += MadLoopBannerStyles.get_MadLoop_Banner(style=style)

    for style in styles:
        f_code += "\nwrite(*,*) ''\nwrite(*,*) 'Style %s in red.'\n"%style
        f_code += MadLoopBannerStyles.get_MadLoop_Banner(style=style, color='red')

    for style in styles:
        f_code += "\nwrite(*,*) ''\nwrite(*,*) 'Style %s in green.'\n"%style
        f_code += MadLoopBannerStyles.get_MadLoop_Banner(style=style, color='green')

    for style in styles:
        f_code += "\nwrite(*,*) ''\nwrite(*,*) 'Style %s without frame.'\n"%style
        f_code += MadLoopBannerStyles.get_MadLoop_Banner(style=style, print_frame=False)
    
    for style in styles:
        f_code += "\nwrite(*,*) ''\nwrite(*,*) 'Style %s with a different frame.'\n"%style
        f_code += MadLoopBannerStyles.get_MadLoop_Banner(
            side_margin=10, up_margin=3,
            top_frame_char = '-',
            bottom_frame_char = '-',
            left_frame_char = '*',
            right_frame_char = '*',
            style=style)
    
    writer.writelines("program testMLBanner\n%s\nend\n"%f_code)
    writer.close()
    # Now compile and run the code
    if os.path.isfile(pjoin(os.getcwd(),'test_ML_banner')):
        os.remove(pjoin(os.getcwd(),'test_ML_banner'))
    misc.call('gfortran -o test_ML_banner test_ML_banner.f',
                                                     cwd=os.getcwd(),shell=True)
    misc.call('./test_ML_banner',cwd=os.getcwd(),shell=True)
