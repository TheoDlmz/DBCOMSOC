from rply import LexerGenerator


class Lexer():
    def __init__(self):
        self.lexer = LexerGenerator()

    def _add_tokens(self):
        self.lexer.add('OPEN_PAREN', r'\(')
        self.lexer.add('CLOSE_PAREN', r'\)')
        self.lexer.add('COMMA', r'\,')
        self.lexer.add('SEMI_COLON',r'\;')
        self.lexer.add('EQUAL',r'\=')
        self.lexer.add('QUOTE',r'\"')
        self.lexer.add('TRUE',r'True')
        self.lexer.add('FALSE',r'False')
        self.lexer.add('NUMBER', r'\d+')
        self.lexer.add('VARIABLE', r'[\w|\d]+') 
        self.lexer.ignore('\s+')

    def get_lexer(self):
        self._add_tokens()
        return self.lexer.build()
        
