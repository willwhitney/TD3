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

# basename = "SPMish"
# grids = [
#     # raw
#     {
#         "main_file": ['main'],
#         "env_name": [
#             'SparsishPointMass-v0',
#         ],

#         "start_timesteps": [0],
#         "max_timesteps": [1e6],
#         "eval_freq": [1e3],
#         "render_freq": [1e10],
#         "seed": list(range(8)),
#     },

#     # learned embedding
#     {
#         "main_file": ['main_embedded'],
#         "env_name": [
#             'SparsishPointMass-v0',
#         ],
#         "decoder": [
#             # "qvel_marg",
#             # "qvel_margscale",
#             "qpos_marg2",
#             "qpos_margscale2",
#         ],

#         "start_timesteps": [0],
#         "max_timesteps": [1e6],
#         "eval_freq": [1e3],
#         "render_freq": [1e10],
#         "seed": list(range(4)),
#     },
# ]

# basename = "SPMAgain_noise"
# grids = [
#     # raw
#     # {
#     #     "main_file": ['main'],
#     #     "env_name": [
#     #         'SparsePointMass-v0',
#     #     ],

#     #     "start_timesteps": [0],
#     #     "max_timesteps": [1e6],
#     #     "eval_freq": [1e3],
#     #     "render_freq": [1e10],
#     #     "seed": list(range(8)),
#     # },

#     # learned embedding
#     {
#         "main_file": ['main_embedded'],
#         "env_name": [
#             'SparsePointMass-v0',
#         ],
#         "decoder": [
#             # "qvel_marg",
#             # "qvel_margscale",
#             "qpos_marg2",
#             "qpos_margscale2",
#         ],

#         "policy_noise": [0.05, 0.1, 0.2],

#         "start_timesteps": [0],
#         "max_timesteps": [1e6],
#         "eval_freq": [1e2],
#         "render_freq": [1e10],
#         "seed": list(range(6)),
#     },
# ]

# basename = "dm.easy_lownoise"
# grids = [
#     # raw
#     {
#         "main_file": ['main'],
#         "policy_name": ['TD3'],
#         "env_name": [
#             'dm.manipulator.reach',
#             'dm.manipulator.chase',
#         ],

#         "expl_noise": [0.05],
#         "policy_noise": [0.1],
#         "start_timesteps": [0],
#         "eval_freq": [1e4],
#         "render_freq": [2e4],
#         "max_timesteps": [1e8],
#         "seed": list(range(8)),
#     },

#     # learned embedding
#     {
#         "main_file": ['main_embedded'],
#         "env_name": [
#             'dm.manipulator.reach',
#             'dm.manipulator.chase',
#         ],
#         "decoder": [
#             "raw_prior_traj4_z5_norm1e4",
#             "raw_prior_traj8_z5_norm1e4",
#             "raw_prior_traj16_z5_norm1e4",
#         ],

#         "expl_noise": [0.05],
#         "policy_noise": [0.1],
#         "start_timesteps": [0],
#         "eval_freq": [1e4],
#         "render_freq": [2e4],

#         "max_timesteps": [1e8],
#         "seed": list(range(8)),
#     },
# ]

# basename = "Thrower_Striker_transfer_start"
# grids = [
#     # raw
#     {
#         "main_file": ['main'],
#         "env_name": [
#             'Striker-v2',
#             'Thrower-v2',
#         ],

#         "max_timesteps": [1e7],
#         "render_freq": [1e10],
#         "seed": list(range(8)),
#     },

#     # learned embedding
#     {
#         "main_file": ['main_embedded'],
#         "env_name": [
#             'Striker-v2',
#             'Thrower-v2',
#         ],
#         "decoder": [
#             "prior_traj4_z7_kl1e4_lr1e4_norm1e4",
#             "marginal_traj4_z7_kl1e4_lr1e4_norm1e4",
#         ],

#         "max_timesteps": [1e7],
#         "render_freq": [1e10],
#         "seed": list(range(8)),
#     },
# ]

# basename = "RFS_nostart_redo_pnoise_scale"
# grids = [
#     # raw
#     # {
#     #     "main_file": ['main'],
#     #     "env_name": [
#     #         'ReacherVerticalSparse-v2',
#     #         'ReacherPushSparse-v2',
#     #         'ReacherSpinSparse-v2',
#     #     ],

#     #     "start_timesteps": [0],
#     #     "max_timesteps": [1e6],
#     #     "render_freq": [1e10],
#     #     "seed": list(range(8)),
#     # },

#     # learned embedding
#     {
#         "main_file": ['main_embedded'],
#         "env_name": [
#             'ReacherVerticalSparse-v2',
#             'ReacherPushSparse-v2',
#             'ReacherSpinSparse-v2',
#         ],
#         "decoder": [
#             "qpos_marg_whitemax",
#             "qpos_margscale_whitemax",
#         ],

#         "start_timesteps": [0],
#         "max_timesteps": [1e6],
#         "render_freq": [1e10],
#         "seed": list(range(8)),
#     },
# ]


# basename = "RVS_dfix_decoderqvel_white_dfix_traj8_z2"
# grids = [
#     # raw
#     # {
#     #     "main_file": ['main'],
#     #     "env_name": [
#     #         'ReacherVerticalSparse-v2',
#     #         'ReacherPushSparse-v2',
#     #         'ReacherSpinSparse-v2',
#     #     ],

#     #     "start_timesteps": [0],
#     #     "max_timesteps": [1e6],
#     #     "render_freq": [1e10],
#     #     "seed": list(range(8)),
#     # },

#     # learned embedding
#     {
#         "main_file": ['main_embedded'],
#         "env_name": [
#             'ReacherVerticalSparse-v2',
#         ],
#         "decoder": [
            # "qvel_white_dfix_z2",
            # "qvel_white_dfix_z3",
            # "qvel_white_dfix_z4",
            # "qvel_white_dfix_traj8_z2",
            # "qvel_white_dfix_traj8_z4",

#             # "qvel_white",
#             # "qvel_white_z3",
#             # "qvel_white_z4",
#             # "qvel_white_traj8_z4",
#         ],

#         "start_timesteps": [0],
#         "max_timesteps": [1e6],
#         "eval_freq": [1e3],
#         "render_freq": [5e3],
#         "seed": list(range(4)),
#     },
# ]

# basename = "RFS_dfix"
# grids = [

#     # learned embedding
#     {
#         "main_file": ['main_embedded'],
#         "env_name": [
#             # 'ReacherVerticalSparse-v2',
#             'ReacherPushSparse-v2',
#             'ReacherSpinSparse-v2',
#         ],
#         "decoder": [
#             "qvel_white_dfix_z2",
#             # "qvel_white_dfix_z3",
#         ],

#         "policy_noise": [0.3, 0.2, 0.1],
#         "expl_noise": [0.2, 0.1, 0.05],
#         "start_timesteps": [0],
#         "max_timesteps": [1e6],
#         "eval_freq": [1e3],
#         "render_freq": [5e3],
#         "seed": list(range(8)),
#     },
#     # {
#     #     "main_file": ['main_embedded'],
#     #     "env_name": [
#     #         'ReacherVerticalSparse-v2',
#     #     ],
#     #     "decoder": [
#     #         "qvel_white_dfix_z3",
#     #         "qvel_white_dfix_z4",
#     #     ],

#     #     "expl_noise": [0.2, 0.05],
#     #     "start_timesteps": [0],
#     #     "max_timesteps": [1e6],
#     #     "eval_freq": [1e3],
#     #     "render_freq": [5e3],
#     #     "seed": list(range(4)),
#     # },
# ]

# basename = "RealSishPMx4_repro_part2"
# grids = [
#     # raw
#     # {
#     #     "main_file": ['main'],
#     #     "env_name": [
#     #         'SparsishPointMass-v0',
#     #     ],

#     #     "start_timesteps": [0],
#     #     "max_timesteps": [1e6],
#     #     "eval_freq": [1e3],
#     #     "render_freq": [1e10],
#     #     "seed": list(range(8)),
#     # },


#     # learned embedding
#     {
#         "main_file": ['main_embedded'],
#         "env_name": [
#             'SparsishPointMass-v0',
#         ],
#         "decoder": [
#             "real_x4_qvel_z2",
#             "real_x4_qvel_z2_take2",
#             "real_x4_qvel_z2_take3",
#         ],

#         # "policy_noise": [0.3, 0.2, 0.1],
#         # "expl_noise": [0.2, 0.1, 0.05],
#         "start_timesteps": [0],
#         "max_timesteps": [1e6],
#         "eval_freq": [1e3],
#         "render_freq": [1e10],
#         "seed": list(range(8, 16)),
#     },
    # {
    #     "main_file": ['main_embedded'],
    #     "env_name": [
    #         'SparsishPointMass-v0',
    #     ],
    #     "decoder": [
    #         "x4_qvel_dfix_z2",
    #         "x4_qpos_dfix_z2",
    #     ],

    #     # "policy_noise": [0.3, 0.2, 0.1],
    #     "expl_noise": [0.2, 0.1, 0.05],
    #     "start_timesteps": [0],
    #     "max_timesteps": [1e6],
    #     "eval_freq": [5e2],
    #     "render_freq": [2e3],
    #     "seed": list(range(8)),
    # },
# ]

# basename = "RV_white_repro_part2"
# grids = [
#     # raw
#     {
#         "main_file": ['main'],
#         "env_name": [
#             'ReacherVertical-v2',
#         ],

#         "start_timesteps": [0],
#         "max_timesteps": [1e6],
#         "eval_freq": [1e2],
#         "render_freq": [1e10],
#         "seed": list(range(8, 16)),
#     },


#     # learned embedding
#     {
#         "main_file": ['main_embedded'],
#         "env_name": [
#             'ReacherVertical-v2',
#         ],
#         "decoder": [
#             "white_qvel",
#             "white_qvel_take2",
#         ],

#         # "policy_noise": [0.3, 0.2, 0.1],
#         # "expl_noise": [0.2, 0.1, 0.05],
#         "start_timesteps": [0],
#         "max_timesteps": [1e6],
#         "eval_freq": [1e2],
#         "render_freq": [1e10],
#         "seed": list(range(8, 16)),
#     },
# ]

# basename = "RVnocollide_highpenalty_step001_gear200_sweep"
# grids = [
#     # raw
#     # {
#     #     "main_file": ['main'],
#     #     "env_name": [
#     #         'ReacherVertical-v2',
#     #     ],

#     #     "start_timesteps": [0],
#     #     "max_timesteps": [1e6],
#     #     "eval_freq": [1e2],
#     #     "render_freq": [1e4],
#     #     "seed": list(range(8)),
#     # },


#     # learned embedding
#     {
#         "main_file": ['main_embedded'],
#         "env_name": [
#             'ReacherVertical-v2',
#         ],
#         "decoder": [
#             "nocollide_step001_gear200_white_qvel",
#             # "nocollide_white_qvel",
#         ],

#         "policy_noise": [0.4, 0.2, 0.1],
#         # "expl_noise": [0.2, 0.1, 0.05],
#         "start_timesteps": [0],
#         "max_timesteps": [1e6],
#         "eval_freq": [1e2],
#         "render_freq": [1e4],
#         "seed": list(range(8)),
#     },
#     {
#         "main_file": ['main_embedded'],
#         "env_name": [
#             'ReacherVertical-v2',
#         ],
#         "decoder": [
#             "nocollide_step001_gear200_white_qvel",
#             # "nocollide_white_qvel",
#         ],

#         # "policy_noise": [0.3, 0.2, 0.1],
#         "expl_noise": [0.2, 0.1, 0.05],
#         "start_timesteps": [0],
#         "max_timesteps": [1e6],
#         "eval_freq": [1e2],
#         "render_freq": [1e4],
#         "seed": list(range(8)),
#     },
# ]

basename = "RFnew3"
grids = [
    # raw
    {
        "main_file": ['main'],
        "env_name": [
            'ReacherVertical-v2',
            'ReacherPush-v2',
            'ReacherTurn-v2',
        ],

        "start_timesteps": [0],
        "max_timesteps": [1e7],
        "eval_freq": [1e2],
        "render_freq": [1e4],
        "seed": list(range(8)),
    },


    # learned embedding
    {
        "main_file": ['main_embedded'],
        "env_name": [
            'ReacherVertical-v2',
            'ReacherPush-v2',
            'ReacherTurn-v2',
        ],
        "decoder": [
            "nocollide_step001_gear200_white_qvel",
            # "nocollide_white_qvel",
        ],

        # "policy_noise": [0.4, 0.2, 0.1],
        # "expl_noise": [0.2, 0.1, 0.05],
        "start_timesteps": [0],
        "max_timesteps": [1e7],
        "eval_freq": [1e2],
        "render_freq": [1e4],
        "seed": list(range(8)),
    },
]

# basename = "RF_white_refix"
# grids = [
#     # raw
#     {
#         "main_file": ['main'],
#         "env_name": [
#             'ReacherPush-v2',
#             'ReacherSpin-v2',
#         ],

#         "start_timesteps": [0],
#         "max_timesteps": [1e6],
#         "eval_freq": [1e2],
#         "render_freq": [1e4],
#         "seed": list(range(8)),
#     },


#     # learned embedding
#     {
#         "main_file": ['main_embedded'],
#         "env_name": [
#             'ReacherPush-v2',
#             'ReacherSpin-v2',
#         ],
#         "decoder": [
#             "white_qvel",
#             "white_qpos",
#         ],

#         # "policy_noise": [0.3, 0.2, 0.1],
#         # "expl_noise": [0.2, 0.1, 0.05],
#         "start_timesteps": [0],
#         "max_timesteps": [1e6],
#         "eval_freq": [1e2],
#         "render_freq": [1e4],
#         "seed": list(range(8)),
#     },
# ]

# basename = "VanillaReacher_decoderwhite_qvel_traj8"
# grids = [
#     # raw
#     # {
#     #     "main_file": ['main'],
#     #     "env_name": [
#     #         'Reacher-v2',
#     #     ],

#     #     "start_timesteps": [0],
#     #     "max_timesteps": [1e6],
#     #     "eval_freq": [1e2],
#     #     "render_freq": [1e10],
#     #     "seed": list(range(8)),
#     # },


#     # learned embedding
#     {
#         "main_file": ['main_embedded'],
#         "env_name": [
#             'Reacher-v2',
#         ],
#         "decoder": [
#             # "white_qvel",
#             # "white_qpos",
#             "white_qvel_traj8",
#         ],

#         # "policy_noise": [0.3, 0.2, 0.1],
#         # "expl_noise": [0.2, 0.1, 0.05],
#         "start_timesteps": [0],
#         "max_timesteps": [1e6],
#         "eval_freq": [1e2],
#         "render_freq": [1e10],
#         "seed": list(range(8)),
#     },
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
        os.system('cp -R reacher_family ' + job_source_dir)
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
        # slurmfile.write("#SBATCH --time=0-02\n")
        slurmfile.write("#SBATCH --time=1-00\n")
        # slurmfile.write("#SBATCH -p dev\n")
        # slurmfile.write("#SBATCH -p uninterrupted,dev\n")
        # slurmfile.write("#SBATCH -p uninterrupted\n")
        slurmfile.write("#SBATCH -p priority\n")
        slurmfile.write("#SBATCH --comment='ICLR workshop 3/7'\n")
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
