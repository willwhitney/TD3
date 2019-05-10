import os
import sys
import itertools

dry_run = '--dry-run' in sys.argv
clear = '--clear' in sys.argv

if not os.path.exists("slurm_logs"):
    os.makedirs("slurm_logs")
if not os.path.exists("slurm_scripts"):
    os.makedirs("slurm_scripts")
code_dir = '/misc/vlgscratch4/FergusGroup/wwhitney'

# basename = "PFnew_start_traj1"
# grids = [
#     # raw
#     # {
#     #     "main_file": ['main'],
#     #     "env_name": [
#     #         'Pusher-v2',
#     #         'Striker-v2',
#     #         'Thrower-v2',
#     #     ],

#     #     # "start_timesteps": [0],
#     #     "max_timesteps": [1e7],
#     #     "eval_freq": [5e3],
#     #     "render_freq": [1e5],
#     #     "seed": list(range(8)),
#     # },


#     # learned embedding
#     {
#         "main_file": ['main_embedded'],
#         "env_name": [
#             'Pusher-v2',
#             'Striker-v2',
#             'Thrower-v2',
#         ],
#         "decoder": [
#             # "white_qvel_traj8_z7",
#             "white_qvel_traj1_z7",
#         ],

#         # "start_timesteps": [0],
#         "max_timesteps": [1e7],
#         "eval_freq": [5e3],
#         "render_freq": [1e5],
#         "seed": list(range(8)),
#     },
# ]

# basename = "newenvs_qvel_traj4_norm1e5"
# grids = [
#     # raw
#     # {
#     #     "main_file": ['main'],
#     #     "env_name": [
#     #         'Ant-v2',
#     #         'Walker2d-v2',
#     #     ],

#     #     # "start_timesteps": [0],
#     #     "max_timesteps": [1e7],
#     #     "eval_freq": [5e3],
#     #     "render_freq": [1e5],
#     #     "seed": list(range(4)),
#     # },


#     # learned embedding
#     {
#         "main_file": ['main_embedded'],
#         "env_name": [
#             'Ant-v2',
#             'Walker2d-v2',
#         ],
#         "decoder": [
#             # "qvel",
#             # "qvel_traj1",
#             # "qvel_traj8",
#             "qvel_traj4_norm1e5",
#         ],

#         # "start_timesteps": [0],
#         "max_timesteps": [1e7],
#         "eval_freq": [5e3],
#         "render_freq": [1e5],
#         "seed": list(range(4)),
#     },
# ]


# basename = "RFnew3_trajs"
# grids = [
# #     # raw
# #     {
# #         "main_file": ['main'],
# #         "env_name": [
# #             'ReacherVertical-v2',
# #             'ReacherPush-v2',
# #             'ReacherTurn-v2',
# #         ],

# #         "start_timesteps": [0],
# #         "max_timesteps": [1e7],
# #         "eval_freq": [1e2],
# #         "render_freq": [1e4],
# #         "seed": list(range(8)),
# #     },


# #     # learned embedding
#     {
#         "main_file": ['main_embedded'],
#         "env_name": [
#             'ReacherVertical-v2',
#             'ReacherPush-v2',
#             'ReacherTurn-v2',
#         ],
#         "decoder": [
#             # "nocollide_step001_gear200_white_qvel",
#             "nocollide_step001_gear200_white_qvel_traj1",
#             "nocollide_step001_gear200_white_qvel_traj8",
#             # "nocollide_white_qvel",
#         ],

#         # "policy_noise": [0.4, 0.2, 0.1],
#         # "expl_noise": [0.2, 0.1, 0.05],
#         "start_timesteps": [0],
#         "max_timesteps": [5e6],
#         "eval_freq": [1e2],
#         "render_freq": [1e4],
#         "seed": list(range(8)),
#     },
# ]

# basename = "RF_transfergap"
# grids = [
# #     # raw
# #     {
# #         "main_file": ['main'],
# #         "env_name": [
# #             'ReacherVertical-v2',
# #             'ReacherPush-v2',
# #             'ReacherTurn-v2',
# #         ],

# #         "start_timesteps": [0],
# #         "max_timesteps": [1e7],
# #         "eval_freq": [1e2],
# #         "render_freq": [1e4],
# #         "seed": list(range(8)),
# #     },


# #     # learned embedding
#     {
#         "main_file": ['main_embedded'],
#         "env_name": [
#             'ReacherVertical-v2',
#             'ReacherPush-v2',
#             # 'ReacherTurn-v2',
#         ],
#         "decoder": [
#             "transfertest",
#         ],
#         "source_env": [
#             "ReacherVertical-v2",
#             "ReacherPush-v2",
#         ],

#         # "policy_noise": [0.4, 0.2, 0.1],
#         # "expl_noise": [0.2, 0.1, 0.05],
#         "start_timesteps": [0],
#         "max_timesteps": [5e6],
#         "eval_freq": [1e2],
#         "render_freq": [1e4],
#         "seed": list(range(8)),
#     },
# ]

# basename = "Thrower_transfergap"
# grids = [
#     # raw
#     # {
#     #     "main_file": ['main'],
#     #     "env_name": [
#     #         'Pusher-v2',
#     #         'Striker-v2',
#     #         'Thrower-v2',
#     #     ],

#     #     # "start_timesteps": [0],
#     #     "max_timesteps": [1e7],
#     #     "eval_freq": [5e3],
#     #     "render_freq": [1e5],
#     #     "seed": list(range(8)),
#     # },


#     # learned embedding
#     {
#         "main_file": ['main_embedded'],
#         "env_name": [
#             'Thrower-v2',
#         ],
#         "decoder": [
#             "transfertest",
#         ],
#         "source_env": [
#             "Thrower-v2",
#         ],

#         # "start_timesteps": [0],
#         "max_timesteps": [1e7],
#         "eval_freq": [5e3],
#         "render_freq": [1e5],
#         "seed": list(range(8)),
#     },
# ]

# basename = "manip_white_traj8"
# grids = [
#     # raw
#     # {
#     #     "main_file": ['main'],
#     #     "env_name": [
#     #         'dm.manipulator.bring_ball',
#     #     ],

#     #     # "start_timesteps": [0],
#     #     "max_timesteps": [1e7],
#     #     "eval_freq": [5e3],
#     #     "render_freq": [1e6],
#     #     "seed": list(range(8)),
#     # },


#     # learned embedding
#     {
#         "main_file": ['main_embedded'],
#         "env_name": [
#             'dm.manipulator.bring_ball',
#         ],
#         "decoder": [
#             # "white_traj4_z5",
#             "white_traj8_z5",
#         ],

#         # "start_timesteps": [0],
#         "max_timesteps": [1e7],
#         "eval_freq": [5e3],
#         "render_freq": [1e6],
#         "seed": list(range(8)),
#     },
# ]

# basename = "PRV128r32_stack4_t2_ddpglr_init"
# grids = [
#     # raw
#     {
#         "main_file": ['main_pixels'],
#         "env_name": [
#             'ReacherVertical-v2',
#         ],
#         # "arch": ['mine_bn', 'ilya_bn'],
#         # "arch": ['minev2_bn', 'minev3_bn'],
#         "arch": ['mine_bn', 'minev2_bn', 'minev3_bn', 'minev4_bn'],
#         "init": [True],

#         # "start_timesteps": [0],
#         "max_timesteps": [1e7],
#         "eval_freq": [1e3],
#         "render_freq": [1e4],
#         "seed": list(range(4)),
#     },
# ]

# basename = "PRV128r64_stack4_embed_t3"
# grids = [
#     # raw
#     {
#         "main_file": ['main_embedded_pixels'],
#         "env_name": [
#             'ReacherVertical-v2',
#         ],
#         # "arch": ['mine_bn', 'ilya_bn'],
#         # "arch": ['minev2_bn', 'minev3_bn'],
#         "arch": ['mine_bn', 'minev2_bn', 'minev3_bn', 'minev4_bn'],
#         "init": [False],
#         "stack": [4],
#         "img_width": [64],

#         "decoder": ["nocollide_step001_gear200_white_qvel"],

#         # "start_timesteps": [0],
#         "max_timesteps": [1e7],
#         "eval_freq": [1e3],
#         "render_freq": [1e4],
#         "seed": list(range(4)),
#     },
# ]

# basename = "PRVr64_noshadow_replaydisk_lrschedfix"
# grids = [
#     {
#         "main_file": ['main_pixels'],
#         "env_name": [
#             'ReacherVertical-v2',
#         ],
#         "arch": ['ilya_bn'],
#         "init": [False],
#         "stack": [4],
#         "img_width": [64],
#         "lr_schedule": [True],
#         "replay_type": ['disk'],

#         "start_timesteps": [10000, 100000],
#         "max_timesteps": [1e6],
#         "eval_freq": [1e3],
#         "render_freq": [1e5],
#         "seed": list(range(4)),
#     },
# ]

# basename = "PRV256d32_actrepeat_ddpglrnodecay"
# grids = [
#     # raw
#     {
#         "main_file": ['main_dummy_pixels'],
#         "env_name": [
#             'ReacherVertical-v2',
#         ],
#         "arch": [
#             'dcgan_bn',
#             # 'dcgandeep_bn',
#             'mine_bn',
#             # 'lin1_extrawide', 
#             # 'lin2_extrawide', 
#             # 'lin3_extrawide', 
#             # 'conv1_1_wide', 
#             # 'conv2_1_wide'
#         ],
#         "init": [False],
#         "ddpglr": [True],
#         "stack": [4],
#         "img_width": [32],

#         "start_timesteps": [0],
#         "max_timesteps": [1e6],
#         "eval_freq": [1e3],
#         "render_freq": [1e10],
#         "seed": list(range(4)),
#     },
# ]

# basename = "PRV256r32_finetune"
# grids = [
#     # raw
#     {
#         "main_file": ['main_pixels_finetune'],
#         "env_name": [
#             'ReacherVertical-v2',
#         ],
#         "arch": [
#             'dcgan_bn',
#             # 'lin1_extrawide', 
#             # 'lin2_extrawide', 
#             # 'lin3_extrawide', 
#             # 'conv1_1_wide', 
#             # 'conv2_1_wide'
#         ],
#         "init": [False],
#         "stack": [4],
#         "img_width": [32],

#         "start_timesteps": [0],
#         "max_timesteps": [1e6],
#         "eval_freq": [1e3],
#         "render_freq": [1e10],
#         "seed": list(range(4)),
#     },
# ]

# basename = "PRV256r64_reg"
# grids = [
#     # raw
#     {
#         "main_file": ['main_pixels_regression'],
#         "env_name": [
#             'ReacherVertical-v2',
#         ],

#         "stack": [4],
#         "img_width": [64],

#         "start_timesteps": [0],
#         "max_timesteps": [1e6],
#         "eval_freq": [1e3],
#         "render_freq": [1e10],
#         "seed": list(range(4)),
#     },
# ]

# basename = "LPM64_start10k_fixdisk_t2"
# grids = [
#     # raw
#     {
#         "main_file": ['main_pixels'],
#         "env_name": [
#             'LinearPointMass-v0',
#         ],
#         "arch": [
#             # 'ilya',
#             'ilya_bn',
#             # 'ilya_coord',
#             # 'ilya_coord_bn'
#             # 'ilya_bn',
#             # 'mine_bn'
#         ],
#         "replay_type": ['dataset', 'disk'],
#         "lr_schedule": [False],
#         "stack": [4],
#         "img_width": [64],

#         "start_timesteps": [10000],
#         "max_timesteps": [1e5],
#         "eval_freq": [1e3],
#         "render_freq": [1e4],
#         "seed": list(range(4)),
#     },
# ]

# basename = "LPMvar64_slow"
# grids = [
#     {
#         "main_file": ['main_pixels'],
#         "env_name": [
#             # 'FastPointMass-v0',
#             'SlowPointMass-v0',
#             # 'GoalLinearPointMass-v0',
#         ],
#         "arch": [
#             'ilya_bn',
#         ],

#         "start_timesteps": [10000],
#         "max_timesteps": [1e6],
#         "eval_freq": [1e3],
#         "render_freq": [1e4],
#         "seed": list(range(4)),
#     },

#     {
#         "main_file": ['main'],
#         "env_name": [
#             # 'FastPointMass-v0',
#             'SlowPointMass-v0',
#             # 'GoalLinearPointMass-v0',
#         ],

#         "start_timesteps": [10000],
#         "max_timesteps": [1e6],
#         "eval_freq": [1e3],
#         "render_freq": [1e4],
#         "seed": list(range(4)),
#     },
# ]

basename = "RFSparse"
grids = [
    # raw
    {
        "main_file": ['main'],
        "env_name": [
            'ReacherVerticalSparse-v2',
            'ReacherTurnSparse-v2',
            'ReacherPushSparse-v2',
        ],

        # "start_timesteps": [0],
        "max_timesteps": [5e6],
        "eval_freq": [1e3],
        "render_freq": [1e4],
        "seed": list(range(4)),
    },


    # learned embedding
    {
        "main_file": ['main_embedded'],
        "env_name": [
            'ReacherVerticalSparse-v2',
            'ReacherTurnSparse-v2',
            'ReacherPushSparse-v2',
        ],
        "decoder": [
            "nocollide_step001_gear200_white_qvel",
        ],

        # "policy_noise": [0.4, 0.2, 0.1],
        # "expl_noise": [0.2, 0.1, 0.05],
        # "start_timesteps": [0],
        "max_timesteps": [5e6],
        "eval_freq": [1e3],
        "render_freq": [1e4],
        "seed": list(range(4)),
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
        # slurmfile.write("#SBATCH --signal=USR1@600\n")
        # slurmfile.write("#SBATCH --time=0-02\n")
        # slurmfile.write("#SBATCH --time=0-12\n")
        slurmfile.write("#SBATCH --time=1-00\n")
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

        slurmfile.write("cd " + true_source_dir + '\n')
        slurmfile.write("srun " + jobcommand)
        slurmfile.write("\n")

    if not dry_run:
        os.system(job_start_command + " &")
