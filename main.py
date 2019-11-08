
num = 1234
num_str = str(num)
num_reverse_str = ''.join([digit for digit in reversed(num_str)])
num_reverse = int(num_reverse_str)

# --------------------------------------------------

num = 2345
sign = 1 if num >= 0 else -1
num_abs = num * sign
num_reverse = 0
while num_abs > 0:
    current_last_digit = num_abs % 10
    num_abs //= 10
    num_reverse = 10 * num_reverse + current_last_digit
num_reverse *= sign
