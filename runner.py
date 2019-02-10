import os
import sys
import itertools

dry_run = '--dry-run' in sys.argv
clear = '--clear' in sys.argv

if not os.path.exists("slurm_logs"):
    os.makedirs("slurm_logs")
if not os.path.exists("slurm_scripts"):
    os.makedirs("slurm_scripts")
code_dir = '/private/home/willwhitney/code'

basename = "manipulator"
grids = [
    # raw
    {
        "main_file": ['main'],
        "policy_name": ['TD3', 'DDPG'],
        "env_name": ['dm.manipulator.bring_ball'],

        "start_timesteps": [0],
        "eval_freq": [5e3],
        "render_freq": [1e10],
        "seed": list(range(4)),
    },

    # learned embedding
    {
        "main_file": ['main_embedded'],
        "env_name": ['dm.manipulator.bring_ball'],
        "decoder": [
            "prior_traj4_z6_kl1e4_lr1e4_norm1e4",
            "marginal_traj4_z6_kl1e4_lr1e4_norm1e4",
        ],

        "start_timesteps": [0],
        "eval_freq": [5e3],
        "render_freq": [1e10],
        "seed": list(range(4)),
    },
]

# basename = "Pusher_start_prior"
# grids = [
#     # raw
#     # {
#     #     "main_file": ['main'],
#     #     "env_name": ['Pusher-v2'],

#     #     "start_timesteps": [1e4],
#     #     # "eval_freq": [1e2],
#     #     "seed": list(range(4)),
#     # },

#     # learned embedding
#     {
#         "main_file": ['main_embedded'],
#         "env_name": ['Pusher-v2'],
#         "decoder": [
#             "prior_traj4_z7_kl1e4_lr1e4_norm1e4",
#             "marginal_traj4_z7_kl1e4_lr1e4_norm1e4",
#         ],

#         "start_timesteps": [1e4],
#         # "eval_freq": [1e2],
#         "seed": list(range(4)),
#     },
# ]



# basename = "SR_nostart3_fixedreward"
# grids = [
#     # raw
#     {
#         "main_file": ['main'],
#         "env_name": ['SparseReacher-v2'],
#         "expl_noise": [0.2, 0.4],

#         "start_timesteps": [0],
#         "eval_freq": [5e3],
#         "seed": list(range(8)),
#     },

#     # # dummy decoder, 4 steps
#     # {
#     #     "main_file": ['main_embedded'],
#     #     "env_name": ['SparseReacher-v2'],
#     #     "dummy_decoder": [True],
#     #     "dummy_traj_len": [4],
#     #     "expl_noise": [0.05, 0.1, 0.2],
#     #     "policy_noise": [0.1, 0.2, 0.4],

#     #     "start_timesteps": [0],
#     #     "eval_freq": [2e3],
#     #     "seed": list(range(4)),
#     # },

#     # learned embedding
#     {
#         "main_file": ['main_embedded'],
#         "env_name": ['SparseReacher-v2'],
#         "decoder": [
#             'prior_traj4_z2',
#             'marginal_traj4_z2',
#             # 'rawstate_prior_traj4_z2',
#             # 'rawstate_marginal_traj4_z2',
#             # 'qposqvel_prior_traj4_z2',
#             # 'qposqvel_marginal_traj4_z2',
#         ],
#         "expl_noise": [0.2, 0.4],
#         "policy_noise": [0.2],

#         "start_timesteps": [0],
#         "eval_freq": [5e3],
#         "seed": list(range(8)),
#     },
#     # {
#     #     "main_file": ['main_embedded'],
#     #     "env_name": ['SparsePointMass-v0'],
#     #     "decoder": ['LPM_traj4'],
#     #     "max_e_action": [3],
#     #     "start_timesteps": [0],
#     #     "seed": list(range(8)),
#     # },
    
# ]

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

    slurm_script_path = 'slurm_scripts/' + jobname + '.slurm'
    slurm_script_dir = os.path.dirname(slurm_script_path)
    os.makedirs(slurm_script_dir, exist_ok=True)

    slurm_log_dir = 'slurm_logs/' + jobname 
    os.makedirs(os.path.dirname(slurm_log_dir), exist_ok=True)

    true_source_dir = code_dir + '/TD3' 
    job_source_dir = code_dir + '/TD3-clones/' + jobname
    try:
        os.makedirs(job_source_dir)
        os.system('cp *.py ' + job_source_dir)
    except FileExistsError:
        # with the 'clear' flag, we're starting fresh
        # overwrite the code that's already here
        if clear:
            print("Overwriting existing files.")
            os.system('cp *.py ' + job_source_dir)

    jobcommand = "python {}/{}.py{}".format(job_source_dir, job['main_file'], flagstring)

    job_start_command = "sbatch " + slurm_script_path
    # jobcommand += " --restart-command '{}'".format(job_start_command)

    print(jobcommand)
    with open(slurm_script_path, 'w') as slurmfile:
        slurmfile.write("#!/bin/bash\n")
        slurmfile.write("#SBATCH --job-name" + "=" + jobname + "\n")
        slurmfile.write("#SBATCH --open-mode=append\n")
        slurmfile.write("#SBATCH --output=slurm_logs/" +
                        jobname + ".out\n")
        slurmfile.write("#SBATCH --error=slurm_logs/" + jobname + ".err\n")
        slurmfile.write("#SBATCH --export=ALL\n")
        slurmfile.write("#SBATCH --signal=USR1@600\n")
        # slurmfile.write("#SBATCH --time=0-06\n")
        slurmfile.write("#SBATCH --time=1-00\n")
        # slurmfile.write("#SBATCH -p dev\n")
        slurmfile.write("#SBATCH -p dev,uninterrupted\n")
        # slurmfile.write("#SBATCH -p priority\n")
        slurmfile.write("#SBATCH -N 1\n")
        slurmfile.write("#SBATCH --mem=32gb\n")

        slurmfile.write("#SBATCH -c 3\n")
        slurmfile.write("#SBATCH --gres=gpu:1\n")

        # slurmfile.write("#SBATCH -c 40\n")
        # slurmfile.write("#SBATCH --constraint=pascal\n")

        slurmfile.write("cd " + true_source_dir + '\n')
        slurmfile.write("srun " + jobcommand)
        slurmfile.write("\n")

    if not dry_run:
        os.system(job_start_command + " &")
