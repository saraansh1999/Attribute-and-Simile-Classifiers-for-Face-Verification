import os

os.system('sbatch run.sh linear Smiling _feat')
os.system('sbatch run.sh CNN Smiling')
os.system('sbatch run.sh mobilenet Smiling')

os.system('sbatch run.sh linear Mustache _feat')
os.system('sbatch run.sh CNN Mustache')
os.system('sbatch run.sh mobilenet Mustache')

os.system('sbatch run.sh linear Youth _feat')
os.system('sbatch run.sh CNN Youth')
os.system('sbatch run.sh mobilenet Youth')

os.system('sbatch run.sh linear White _feat')
os.system('sbatch run.sh CNN White')
os.system('sbatch run.sh mobilenet White')

os.system('sbatch run.sh linear Goatee _feat')
os.system('sbatch run.sh CNN Goatee')
os.system('sbatch run.sh mobilenet Goatee')
