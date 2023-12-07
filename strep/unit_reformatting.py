import pint


SPECIAL_SYMBOLS = {
    'NUMBER': r'#',
    'PERCENT': r'%',
}

class CustomUnitReformater(pint.UnitRegistry):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.define('FLOPS = 1 = FLOPS')
        self.define('wattseconds = watthours * 3600 = Ws')
        self.define('number = 1 = NUMBER')
        self.define('percent = 1 = PERCENT')

    def reformat_value(self, value, unit_from=None, unit_to=None):
        symbol = ''
        for unit, short_unit in SPECIAL_SYMBOLS.items(): # remap num and percent units
            if short_unit == unit_from:
                unit_from = unit
            if short_unit == unit_to:
                unit_to = unit
        # run unit conversion, else only format
        if unit_from is not None:
            try:
                val = value * self[unit_from]
                if unit_to is not None:
                    conv = val.to(unit_to)
                else:
                    conv = val.to_compact()
                value = conv.magnitude
                symbol = self.get_unit_symbol(conv.u)
            except pint.errors.UndefinedUnitError as e:
                print(e)
        # string formatting
        if value < 1000 and value >= 0.0001:
            value = f'{value:8.6f}'[:8]
        else:
            value = f'{value:.2e}'
        return value, symbol

    def get_unit_symbol(self, input, with_brackets=True):
        if isinstance(input, str): # unit string input
            input = 0 * self[input]
            input = input.u
        symbol = self._get_symbol(str(input))
        if symbol in SPECIAL_SYMBOLS:
            symbol = SPECIAL_SYMBOLS[symbol]
        if with_brackets:
            symbol = f'[{symbol}]'
        return symbol
