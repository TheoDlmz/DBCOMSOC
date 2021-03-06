from rply import ParserGenerator
from ast import Query

# Parser : parse the language

class Parser():
    def __init__(self):
        self.pg = ParserGenerator(
            # A list of all token names accepted by the parser.
            ['NUMBER', 'VARIABLE', 'OPEN_PAREN', 'CLOSE_PAREN',
             'SEMI_COLON', 'COMMA',  'EQUAL', 'QUOTE', 'FALSE', 'TRUE']
        )

    def parse(self):
        
        # query of the kind "q() = [...];"
        @self.pg.production('program : VARIABLE OPEN_PAREN CLOSE_PAREN EQUAL expression SEMI_COLON')
        
        def program(p):
            return (p[0].value,p[4])
        
        # list of atoms : "at_1,..,at_k"
        @self.pg.production('expression : atom COMMA expression')
        @self.pg.production('expression : atom')
        def expression(p):
            if len(p) == 1:
                return [p[0]]
            else:
                return [p[0]] + p[2]
                
        
        # One atom is At([...]) with a tuple inside
        @self.pg.production('atom : VARIABLE OPEN_PAREN tuple CLOSE_PAREN')
        
        def atom(p):
            return (p[0].value,p[2])
            
                
        # The tuple inside an atom is a list of element "x,y,..."
        @self.pg.production('tuple : element COMMA tuple')
        @self.pg.production('tuple : element ')
        
        def tuple(p):
            if len(p) == 1:
                return [p[0]]
            else:
                return [p[0]] + p[2]

        # Anelement is either : "x=n" with n a number, "x=y", "x= 'string'" or "x = True/False"
        @self.pg.production('element : VARIABLE EQUAL NUMBER')
        @self.pg.production('element : VARIABLE EQUAL VARIABLE')
        @self.pg.production('element : VARIABLE EQUAL QUOTE VARIABLE QUOTE')
        @self.pg.production('element : VARIABLE EQUAL bool')
        
        def element(p):
            if len(p) == 3:
                if p[2] == "True" or p[2] == "False":
                    return ("BOOL",p[0].value,p[2])
                else:
                    return (p[2].gettokentype(),p[0].value,p[2].value)
            else:
                return ("STRING",p[0].value,p[3].value)
                
        @self.pg.production('bool : TRUE')
        @self.pg.production('bool : FALSE')
        
        def bool(p):
            return p[0].value
            
        @self.pg.error
        def error_handle(token):
            raise ValueError(token)

    def get_parser(self):
        return self.pg.build()