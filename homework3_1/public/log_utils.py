from time import time

RESET = "\033[0m"
MAGENTA = "\033[35m"


def log_time(func):
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        print(f"{MAGENTA}{func.__name__} took {(end - start):.3f} seconds{RESET}")
        return result

    return wrapper
