from string import ascii_uppercase


def get_upper_kv(module):
    return {k: getattr(module, k)
            for k in dir(module)
            if k[0] in ascii_uppercase}
