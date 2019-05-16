import os
import sys
import itertools

dry_run = '--dry-run' in sys.argv
clear = '--clear' in sys.argv
local = False #'--local' in sys.argv
double_book = False #'--double-book' in sys.argv

if not os.path.exists("slurm_logs"):
    os.makedirs("slurm_logs")
if not os.path.exists("slurm_scripts"):
    os.makedirs("slurm_scripts")
# code_dir = '/misc/vlgscratch4/FergusGroup/wwhitney'
code_dir = '..'

basename = "PRF64"
grids = [
    # raw
    {
        "main_file": ['main_pixels'],
        "env_name": [
            'ReacherVertical-v2',
            'ReacherTurn-v2',
            'ReacherPush-v2',
        ],
        "arch": [
            'ilya_bn',
        ],
        "init": [False],
        "stack": [4],
        "img_width": [64],

        "start_timesteps": [1e4],
        "max_timesteps": [1e4],
        "eval_freq": [5e3],
        "render_freq": [1e10],
        "seed": list(range(8)),
    },
]


jobs = []
for grid in grids:
    individual_options = [[{key: value} for value in values]
                          for key, values in grid.items()]
    product_options = list(itertools.product(*individual_options))
    jobs += [{k: v for d in option_set for k, v in d.items()}
             for option_set in product_options]

if dry_run:
    print("NOT starting {} jobs:".format(len(jobs)))
else:
    print("Starting {} jobs:".format(len(jobs)))

all_keys = set().union(*[g.keys() for g in grids])
merged = {k: set() for k in all_keys}
for grid in grids:
    for key in all_keys:
        grid_key_value = grid[key] if key in grid else ["<<NONE>>"]
        merged[key] = merged[key].union(grid_key_value)
varying_keys = {key for key in merged if len(merged[key]) > 1}

excluded_flags = {'main_file'}

job_specs = []
for job in jobs:
    jobname = basename
    flagstring = ""
    for flag in job:

        # construct the string of arguments to be passed to the script
        if not flag in excluded_flags:
            if isinstance(job[flag], bool):
                if job[flag]:
                    flagstring = flagstring + " --" + flag
                else:
                    print("WARNING: Excluding 'False' flag " + flag)
            else:
                flagstring = flagstring + " --" + flag + " " + str(job[flag])

        # construct the job's name
        if flag in varying_keys:
            jobname = jobname + "_" + flag + str(job[flag])
    flagstring = flagstring + " --name " + jobname

    slurm_log_dir = 'slurm_logs/' + jobname 
    os.makedirs(os.path.dirname(slurm_log_dir), exist_ok=True)

    true_source_dir = code_dir + '/TD3' 
    job_source_dir = code_dir + '/TD3-clones/' + jobname
    try:
        os.makedirs(job_source_dir)
        os.system('cp *.py ' + job_source_dir)
        os.system('cp -R reacher_family ' + job_source_dir)
        os.system('cp -R pointmass ' + job_source_dir)
    except FileExistsError:
        # with the 'clear' flag, we're starting fresh
        # overwrite the code that's already here
        if clear:
            print("Overwriting existing files.")
            os.system('cp *.py ' + job_source_dir)
            os.system('cp -R reacher_family ' + job_source_dir)
            os.system('cp -R pointmass ' + job_source_dir)

    jobcommand = "python {}/{}.py{}".format(job_source_dir, job['main_file'], flagstring)

    # jobcommand += " --restart-command '{}'".format(job_start_command)
    job_specs.append((jobname, jobcommand))

increment = 1 if not double_book else 2

# def build_joint_name(a, b):
#     import difflib
#     matches = difflib.SequenceMatcher(None, a, b).get_matching_blocks()
#     a_i, b_i = 0, 0
#     for match in matches:
#         if a_i 

i = 0
while i < len(job_specs):
    current_jobs = job_specs[i : i + increment]

    for job_spec in current_jobs: print(job_spec[1])
    print('')

    joint_name = ""
    for job_spec in current_jobs: 
        if len(joint_name) > 0: joint_name += "__"
        joint_name += job_spec[0]

    joint_name = joint_name[:200]

    if local:
        gpu_id = i % 1
        log_path = "slurm_logs/" + job_spec[0]
        os.system("env CUDA_VISIBLE_DEVICES={gpu_id} {command} > {log_path}.out 2> {log_path}.err &".format(
                gpu_id=gpu_id, command=job_spec[1], log_path=log_path))

    else:
        slurm_script_path = 'slurm_scripts/' + joint_name + '.slurm'
        slurm_script_dir = os.path.dirname(slurm_script_path)
        os.makedirs(slurm_script_dir, exist_ok=True)

        job_start_command = "sbatch " + slurm_script_path

        with open(slurm_script_path, 'w') as slurmfile:
            slurmfile.write("#!/bin/bash\n")
            slurmfile.write("#SBATCH --job-name" + "=" + joint_name + "\n")
            slurmfile.write("#SBATCH --open-mode=append\n")
            slurmfile.write("#SBATCH --output=slurm_logs/" +
                            joint_name + ".out\n")
            slurmfile.write("#SBATCH --error=slurm_logs/" + joint_name + ".err\n")
            slurmfile.write("#SBATCH --export=ALL\n")
            # slurmfile.write("#SBATCH --signal=USR1@600\n")
            # slurmfile.write("#SBATCH --time=0-02\n")
            # slurmfile.write("#SBATCH --time=0-12\n")
            slurmfile.write("#SBATCH --time=2-00\n")
            # slurmfile.write("#SBATCH -p dev\n")
            # slurmfile.write("#SBATCH -p uninterrupted,dev\n")
            # slurmfile.write("#SBATCH -p uninterrupted\n")
            # slurmfile.write("#SBATCH -p dev,uninterrupted,priority\n")
            slurmfile.write("#SBATCH -N 1\n")
            slurmfile.write("#SBATCH --mem=64gb\n")

            slurmfile.write("#SBATCH -c 4\n")
            slurmfile.write("#SBATCH --gres=gpu:1\n")

            # slurmfile.write("#SBATCH -c 40\n")
            slurmfile.write("#SBATCH --constraint=pascal|turing|volta\n")
            # slurmfile.write("#SBATCH --exclude=lion[1-26]\n")

            slurmfile.write("cd " + true_source_dir + '\n')

            for job_i, job_spec in enumerate(current_jobs):
                # srun_comm = "srun --job-name={name} --output=slurm_logs/{name}.out --error=slurm_logs/{name}.err {command} &".format(name=job_spec[0], command=job_spec[1])
                srun_comm = "{command} &".format(name=job_spec[0], command=job_spec[1])
                slurmfile.write(srun_comm)

                # slurmfile.write("srun --job-name=" + job_spec[0] + " --output=slurm_logs/" + job_spec[0] + ".out --error=slurm_logs/" + job_spec[0] + ".err" + jobcommand)
                slurmfile.write("\n")
            slurmfile.write("wait\n")

        if not dry_run:
            os.system(job_start_command + " &")

    i += increment
