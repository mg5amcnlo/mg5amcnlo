import logging

BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

COLORS = {
    'WARNING'  : BLUE,
    'INFO'     : BLACK,
    'DEBUG'    : GREEN,
    'CRITICAL' : RED,
    'ERROR'    : RED,
    'BLACK'    : BLACK,
    'RED'      : RED,
    'GREEN'    : GREEN,
    'YELLOW'   : YELLOW,
    'BLUE'     : BLUE,
    'MAGENTA'  : MAGENTA,
    'CYAN'     : CYAN,
    'WHITE'    : WHITE,
}

for i in range(0,11):
    COLORS['Level %i'%i] = COLORS['DEBUG']

RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ  = "\033[1m"

class ColorFormatter(logging.Formatter):

    def __init__(self, *args, **kwargs):
        # can't do super(...) here because Formatter is an old school class
        logging.Formatter.__init__(self, *args, **kwargs)

    def format(self, record):
        levelname = record.levelname
        try:
            color_choice = COLORS[levelname]
        except KeyError:
            color_choice = COLORS['INFO']
        new_args=[]
        # A not-so-nice but working way of passing arguments to this formatter
        # from MadGraph.
        color_specified = False
        for arg in record.args:
            if isinstance(arg,str) and arg.startswith('$MG'):
                elems=arg.split(':')
                if len(elems)>2:
                    if elems[1]=='color':
                        color_specified = True                            
                        color_choice = COLORS[elems[2]]
                    if color_choice == 0:
                        color_choice = 30
            else:
                new_args.append(arg)
        record.args = tuple(new_args)
        color     = COLOR_SEQ % (30 + color_choice)
        message   = logging.Formatter.format(self, record)
        if not message.endswith('$RESET'):
            message +=  '$RESET'
        for k,v in COLORS.items():
            color_flag = COLOR_SEQ % (v+30)
            message = message.replace("$" + k, color_flag)\
                         .replace("$BG" + k,  COLOR_SEQ % (v+40))\
                         .replace("$BG-" + k, COLOR_SEQ % (v+40))        
        
        if levelname == 'INFO':
            message   = message.replace("$RESET", '' if not color_specified else RESET_SEQ)\
                           .replace("$BOLD",  '')\
                           .replace("$COLOR", color if color_specified else '')
            return message
        else:    
            message   = message.replace("$RESET", RESET_SEQ)\
                           .replace("$BOLD",  BOLD_SEQ)\
                           .replace("$COLOR", color)

        return message 

logging.ColorFormatter = ColorFormatter
