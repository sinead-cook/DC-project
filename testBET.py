import os
import subprocess
import sys

sys.stdout = open('stdout.txt', 'w')
sys.stderr = open('stdout.txt', 'w')
# from nipype.interfaces import fsl
# from distutils.core import setup
# setup(
#     scripts=['/usr/local/fsl/bin/bet','/usr/local/fsl/bin/remove_ext']
# )
outputFileName = 'betlog.txt'
outputFile = open(outputFileName, "w")
p = subprocess.Popen(['which', 'bet'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.PIPE)
result = p.stdout.read()
os.environ['PATH'] += os.path.split(result)[0]
os.chdir(os.path.split(result)[0])
proc = subprocess.Popen(['bet /Users/Sinead/temp.nii.gz out.nii.gz -f -v 0.3'], shell=True, cwd=os.path.split(result)[0], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.PIPE)
proc.stdin.close()
proc.wait()
result = proc.returncode
outputFile.write(proc.stdout.read())
# fsl.BET(in_file='/Users/Sinead/temp.nii.gz', out_file='/Users/Sinead/out.nii.gz', frac=0.5)