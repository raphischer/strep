import subprocess

def execute(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line 
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)
    
def print_colored_block(message, ok=True):
    col = '\033[92m' if ok else '\033[91m'
    print(col + (u"\u2588"*60 + '\n')*4 + '\n')
    print(f"{message}\n")
    print((u"\u2588"*60 + '\n')*4 + '\033[0m')