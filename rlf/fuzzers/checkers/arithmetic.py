from .checker import Checker
from ...ethereum import ADD, MUL, SUB



class Arithmetic(Checker):
    def __init__(self):
        super().__init__()

    def check(self, logger):
        for i, log in enumerate(logger.logs):
            # Checking for overflow in ADD and MUL
            if log.op in (ADD, MUL):
                try:
                    op1 = int(log.stack[-1], 16)
                    op2 = int(log.stack[-2], 16)
                    
                    if log.op == ADD:
                        result = op1 + op2
                    elif log.op == MUL:
                        result = op1 * op2

                    # Check if result is less than either operand (indicates overflow)
                    if result < op1 or result < op2:
                        return True
                except (ValueError, IndexError):
                    # Stack access failure or invalid values
                    continue

            # Checking for underflow in SUB
            elif log.op == SUB:
                try:
                    op1 = int(log.stack[-1], 16)
                    op2 = int(log.stack[-2], 16)
                    
                    # Check for underflow (op2 should not be greater than op1)
                    if op2 > op1:
                        return True
                except (ValueError, IndexError):
                    # Stack access failure or invalid values
                    continue

        return False
