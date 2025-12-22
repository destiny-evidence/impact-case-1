import re

from lark import Lark, Tree, Token

grammar = Lark(
    '''
    ?start: or
    
    ?or: and 
        | or "OR" and           -> or
        | or near and           -> near
    
    ?near: "NEAR/" NUMBER
         | "W/" NUMBER
    
    ?and: atom
        | and "AND NOT" atom    -> not
        | and "NOT" atom        -> not
        | and "AND" atom        -> and
    
    ?phrase_inner: token
         | token phrase_inner
         | token "-" token
    ?phrase: _QUOT phrase_inner _QUOT
    
    ?token: WORD
         | WORD "*"             -> wild_pre
         | WORD "?"             -> wild_one
         | "*" WORD             -> wild_post
         | "*" WORD "*"         -> wild_prepost
         | WORD "$" WORD        -> wild_in
         | WORD "?" WORD        -> wild_in
    
    ?atom: token
         | phrase
         | "(" or ")"
    
    _QUOT : "\\""
    
    
    DIGIT: "0".."9"
    INT: DIGIT+
    SIGNED_INT: ["+"|"-"] INT
    DECIMAL: INT "." INT? | "." INT
    FLOAT: DECIMAL
    NUMBER: FLOAT | INT
    
    LCASE_LETTER: "a".."z"
    UCASE_LETTER: "A".."Z"
    
    LETTER: UCASE_LETTER | LCASE_LETTER | NUMBER | "."
    WORD: LETTER+
    
    %import common.WS
    
    %ignore WS
    ''', parser='earley', start='start',
)

NEAR_OFFSET = 0


def expand_wildcard(token: str, prefix: str, postfix: str, expansions: dict[str, list[str]]):
    if token in expansions:
        return f'({' OR '.join(expansions[token])})'
    return f'{prefix}{token}{postfix}'

def parse(query, expansions: dict[str, list[str]] | None = None):
    comm = re.compile(r'# .*\n')
    tree = grammar.parse(comm.sub('', query))
    expansions = {} if expansions is None else expansions

    def recurse(subtree: Tree | Token, indent: str = '') -> str:
        if isinstance(subtree, Token):
            return subtree.value

        if not isinstance(subtree, Tree):
            raise SyntaxError('This is not a tree!')

        if subtree.data == 'or':
            return (
                '(\n'
                f'{indent}   {recurse(subtree.children[0], indent=indent + '  ')}\n'
                f'{indent}OR {recurse(subtree.children[1], indent=indent + '  ')}\n'
                f'{indent[:-2]})'
            )

        if subtree.data == 'and':
            return (
                '(\n'
                f'{indent}    {recurse(subtree.children[0], indent=indent + '  ')}\n'
                f'{indent}AND {recurse(subtree.children[1], indent=indent + '  ')}\n'
                f'{indent[:-2]})'
            )

        if subtree.data == 'not':
            return (
                '(\n'
                f'{indent}    {recurse(subtree.children[0], indent=indent + '  ')}\n'
                f'{indent}NOT {recurse(subtree.children[1], indent=indent + '  ')}\n'
                f'{indent[:-2]})'
            )

        if subtree.data == 'near':
            near = int(subtree.children[1].value) + NEAR_OFFSET
            near_str = 'W' if near < 2 else f'{near}W'
            return (
                '(\n'
                f'{indent}  {recurse(subtree.children[0], indent=indent + '  ')}\n'
                f'{indent}{near_str} '
                f'{recurse(subtree.children[2], indent=indent + '  ')}\n'
                f'{indent[:-2]})'
            )

        if subtree.data == 'phrase_inner':
            return f'({' W '.join([recurse(child) for child in subtree.children])})'

        if subtree.data == 'wild_pre':
            return expand_wildcard(subtree.children[0].value, prefix='', postfix='*', expansions=expansions)
        if subtree.data == 'wild_post':
            return expand_wildcard(subtree.children[0].value, prefix='', postfix='', expansions=expansions) # FIXME: postfix not available in surround
        if subtree.data == 'wild_prepost':
            # FIXME: postfix not available in surround
            # return ('('
            #         f'{expand_wildcard(subtree.children[0].value, prefix='', postfix='*', expansions=expansions)} OR '
            #         f'{expand_wildcard(subtree.children[0].value, prefix='*', postfix='', expansions=expansions)}'
            #         ')')
            return expand_wildcard(subtree.children[0].value, prefix='', postfix='', expansions=expansions)
        if subtree.data == 'wild_in':
            return f'{subtree.children[0].value}?{subtree.children[1].value}'
        if subtree.data == 'wild_one':
            return f'{subtree.children[0].value}?'

        raise SyntaxError("You shouldn't end up here.")

    return recurse(tree)
