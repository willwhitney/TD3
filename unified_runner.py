import os
import sys
import itertools
import subprocess

dry_run = '--dry-run' in sys.argv
clear = '--clear' in sys.argv
local = '--local' in sys.argv
double_book = '--double-book' in sys.argv
quad_book = '--quad-book' in sys.argv

if double_book:
    increment = 2
elif quad_book:
    increment = 4
else:
    increment = 1


if not os.path.exists("slurm_logs"):
    os.makedirs("slurm_logs")
if not os.path.exists("slurm_scripts"):
    os.makedirs("slurm_scripts")
# code_dir = '/misc/vlgscratch4/FergusGroup/wwhitney'
code_dir = '..'
excluded_flags = {'main_file'}



# embed_grid = [
#     # only make one job in each grid — 
#     #   not sure if aligning the RL jobs with these will work otherwise
#     {
#         'main_file': ['main'],
#         'qpos-qvel': [True],
#         'epochs': [200],
#         'embed-every': [10000000],
#         'render-every': [10000000],

#         'env': ['ReacherVertical-v2'],

#         'model-type': ['deterministic'],
#         'kl': [0]

#         # 'epochs': [10],
#         # 'dataset-size': [1000],
#     }
# ]

# rl_grid = [
#     {
#         "main_file": ['main_embedded'],
#         "env_name": [
#             'ReacherVertical-v2',
#             'ReacherPush-v2',
#             'ReacherTurn-v2',
#         ],

#         "_embed_job": list(range(len(embed_grid))),

#         "start_timesteps": [0],
#         "max_timesteps": [5e6],
#         "eval_freq": [1e3],
#         "render_freq": [1e4],
#         "seed": list(range(8)),

#     },
# ]

# embed_grid = [
#     # only make one job in each grid — 
#     #   not sure if aligning the RL jobs with these will work otherwise
#     # {
#     #     "main_file": ['main_darla'],
#     #     "env": ['ReacherTurn-v2'],
#     #     "epochs": [200],
#     #     "embed-every": [200000],
#     # },
#     {
#         "main_file": ['main_darla'],
#         "env": ['ReacherPush-v2'],
#         "epochs": [200],
#         "embed-every": [200000],
#     },
# ]

# rl_grid = [
#     # {
#     #     "main_file": ['main_pixels_vae'],
#     #     "env_name": [
#     #         # 'ReacherVertical-v2',
#     #         'ReacherTurn-v2',
#     #         # 'ReacherPush-v2',
#     #     ],

#     #     "_embed_job": [0],
#     #     # "decoder": ["darla", "darla_skl01"],
#     #     "source_env": ['PixelReacherTurn-v2'],
#     #     "source_img_width": [64],

#     #     "max_timesteps": [5e6],
#     #     "eval_freq": [5e3],
#     #     "render_freq": [1e5],
#     #     "seed": list(range(8)),
#     # },
#     {
#         "main_file": ['main_pixels_vae'],
#         "env_name": [
#             # 'ReacherVertical-v2',
#             # 'ReacherTurn-v2',
#             'ReacherPush-v2',
#         ],

#         "_embed_job": [0],
#         # "decoder": ["darla", "darla_skl01"],
#         "source_env": ['PixelReacherPush-v2'],
#         "source_img_width": [64],

#         "max_timesteps": [5e6],
#         "eval_freq": [5e3],
#         "render_freq": [1e5],
#         "seed": list(range(8)),
#     },
# ]



# basename = "state_pixel_deterministic"
# embed_grid = [
#     # only make one job in each grid — 
#     #   not sure if aligning the RL jobs with these will work otherwise
# ]

# rl_grid = [
#     {
#         "main_file": ['main_pixels_encode'],
#         "env_name": [
#             'ReacherVertical-v2',
#             # 'ReacherTurn-v2',
#             # 'ReacherPush-v2',
#         ],

#         "decoder": ["detpred"],
#         "source_env": ['PixelReacherVertical-v2'],
#         "source_img_width": [64],

#         "max_timesteps": [5e6],
#         "eval_freq": [5e3],
#         "render_freq": [1e5],
#         "seed": list(range(8)),
#     },
#     {
#         "main_file": ['main_pixels_encode'],
#         "env_name": [
#             # 'ReacherVertical-v2',
#             'ReacherTurn-v2',
#             # 'ReacherPush-v2',
#         ],

#         "decoder": ["embed_pixel_deterministic_rf_envReacherTurn-v2"],
#         "source_env": ['PixelReacherTurn-v2'],
#         "source_img_width": [64],

#         "max_timesteps": [5e6],
#         "eval_freq": [5e3],
#         "render_freq": [1e5],
#         "seed": list(range(8)),
#     },
#     {
#         "main_file": ['main_pixels_encode'],
#         "env_name": [
#             # 'ReacherVertical-v2',
#             # 'ReacherTurn-v2',
#             'ReacherPush-v2',
#         ],

#         "decoder": ["embed_pixel_deterministic_rf_envReacherPush-v2"],
#         "source_env": ['PixelReacherPush-v2'],
#         "source_img_width": [64],

#         "max_timesteps": [5e6],
#         "eval_freq": [5e3],
#         "render_freq": [1e5],
#         "seed": list(range(8)),
#     },
# ]


# basename = "gym_sweep_fixedasize"
# embed_grid = [
#     # only make one job in each grid — 
#     #   not sure if aligning the RL jobs with these will work otherwise
#     {
#         'main_file': ['main_pixels'],
#         'epochs': [200],
#         'embed-every': [200],

#         'env': ['Hopper-v2'],
#         'action-embed-size': [3],

#         'model-type': ['variational'],
#         'state-kl': [5e-7],
#         'state-embed-size': [100],
#     },
#     {
#         'main_file': ['main_pixels'],
#         'epochs': [200],
#         'embed-every': [200],

#         'env': ['Swimmer-v2'],
#         'action-embed-size': [2],

#         'model-type': ['variational'],
#         'state-kl': [5e-7],
#         'state-embed-size': [100],
#     },
#     {
#         'main_file': ['main_pixels'],
#         'epochs': [200],
#         'embed-every': [200],

#         'env': ['Walker2d-v2'],
#         'action-embed-size': [6],

#         'model-type': ['variational'],
#         'state-kl': [5e-7],
#         'state-embed-size': [100],
#     },
# ]

# rl_grid = [
#     {
#         "main_file": ['main_embedded_pixels_encode'],
#         "env_name": [
#             'Hopper-v2',
#         ],

#         "_embed_job": [0],
#         "source_env": ['PixelHopper-v2'],
#         "source_img_width": [64],

#         "max_timesteps": [5e6],
#         "eval_freq": [5e3],
#         "render_freq": [1e5],
#         "seed": list(range(8)),
#     },
#     {
#         "main_file": ['main_embedded_pixels_encode'],
#         "env_name": [
#             'Swimmer-v2',
#         ],

#         "_embed_job": [1],
#         "source_env": ['PixelSwimmer-v2'],
#         "source_img_width": [64],

#         "max_timesteps": [5e6],
#         "eval_freq": [5e3],
#         "render_freq": [1e5],
#         "seed": list(range(8)),
#     },
#     {
#         "main_file": ['main_embedded_pixels_encode'],
#         "env_name": [
#             'Walker2d-v2',
#         ],

#         "_embed_job": [2],
#         "source_env": ['PixelWalker2d-v2'],
#         "source_img_width": [64],

#         "max_timesteps": [5e6],
#         "eval_freq": [5e3],
#         "render_freq": [1e5],
#         "seed": list(range(8)),
#     },
# ]


# basename = "RP_ksweep_seed2"
# embed_grid = [
#     # only make one job in each grid — 
#     #   not sure if aligning the RL jobs with these will work otherwise
#     {
#         'main_file': ['main'],
#         'seed': [2],
#         'epochs': [200],
#         'embed-every': [10000],
#         'qpos-qvel': [True],
#         'traj-len': [2],

#         'env': ['ReacherPush-v2'],
#     },
#     {
#         'main_file': ['main'],
#         'seed': [2],
#         'epochs': [200],
#         'embed-every': [10000],
#         'qpos-qvel': [True],
#         'traj-len': [4],

#         'env': ['ReacherPush-v2'],
#     },
#     {
#         'main_file': ['main'],
#         'seed': [2],
#         'epochs': [200],
#         'embed-every': [10000],
#         'qpos-qvel': [True],
#         'traj-len': [6],

#         'env': ['ReacherPush-v2'],
#     },
#     {
#         'main_file': ['main'],
#         'seed': [2],
#         'epochs': [200],
#         'embed-every': [10000],
#         'qpos-qvel': [True],
#         'traj-len': [8],

#         'env': ['ReacherPush-v2'],
#     },
#     {
#         'main_file': ['main'],
#         'seed': [2],
#         'epochs': [200],
#         'embed-every': [10000],
#         'qpos-qvel': [True],
#         'traj-len': [10],

#         'env': ['ReacherPush-v2'],
#     },
# ]

# rl_grid = [
#     {
#         "main_file": ['main_embedded'],
#         "env_name": [
#             'ReacherPush-v2',
#         ],

#         "_embed_job": [0],
#         "source_env": ['ReacherPush-v2'],

#         "start_timesteps": [0],
#         "max_timesteps": [1e6],
#         "eval_freq": [1e2],
#         "render_freq": [1e4],
#         "seed": list(range(4)),
#     },
#     {
#         "main_file": ['main_embedded'],
#         "env_name": [
#             'ReacherPush-v2',
#         ],

#         "_embed_job": [1],
#         "source_env": ['ReacherPush-v2'],

#         "start_timesteps": [0],
#         "max_timesteps": [1e6],
#         "eval_freq": [1e2],
#         "render_freq": [1e4],
#         "seed": list(range(4)),
#     },
#     {
#         "main_file": ['main_embedded'],
#         "env_name": [
#             'ReacherPush-v2',
#         ],

#         "_embed_job": [2],
#         "source_env": ['ReacherPush-v2'],

#         "start_timesteps": [0],
#         "max_timesteps": [1e6],
#         "eval_freq": [1e2],
#         "render_freq": [1e4],
#         "seed": list(range(4)),
#     },
#     {
#         "main_file": ['main_embedded'],
#         "env_name": [
#             'ReacherPush-v2',
#         ],

#         "_embed_job": [3],
#         "source_env": ['ReacherPush-v2'],

#         "start_timesteps": [0],
#         "max_timesteps": [1e6],
#         "eval_freq": [1e2],
#         "render_freq": [1e4],
#         "seed": list(range(4)),
#     },
#     {
#         "main_file": ['main_embedded'],
#         "env_name": [
#             'ReacherPush-v2',
#         ],

#         "_embed_job": [4],
#         "source_env": ['ReacherPush-v2'],

#         "start_timesteps": [0],
#         "max_timesteps": [1e6],
#         "eval_freq": [1e2],
#         "render_freq": [1e4],
#         "seed": list(range(4)),
#     },
# ]

# basename = "states_gym_sweep"
# embed_grid = [
#     # only make one job in each grid — 
#     #   not sure if aligning the RL jobs with these will work otherwise
#     # {
#     #     'main_file': ['main'],
#     #     'epochs': [200],
#     #     'embed-every': [10000],

#     #     'env': ['Hopper-v2'],
#     #     'embed-size': [3],
#     #     'qpos-qvel': [True],
#     # },
#     # {
#     #     'main_file': ['main'],
#     #     'epochs': [200],
#     #     'embed-every': [10000],

#     #     'env': ['Swimmer-v2'],
#     #     'embed-size': [2],
#     #     'qpos-qvel': [True],
#     # },
#     # {
#     #     'main_file': ['main'],
#     #     'epochs': [200],
#     #     'embed-every': [10000],

#     #     'env': ['Walker2d-v2'],
#     #     'embed-size': [6],
#     #     'qpos-qvel': [True],
#     # },
#     # {
#     #     'main_file': ['main'],
#     #     'epochs': [200],
#     #     'embed-every': [10000],

#     #     'env': ['HalfCheetah-v2'],
#     #     'embed-size': [6],
#     #     'qpos-qvel': [True],
#     # },
#     # {
#     #     'main_file': ['main'],
#     #     'epochs': [200],
#     #     'embed-every': [10000],

#     #     'env': ['Ant-v2'],
#     #     'embed-size': [8],
#     #     'qpos-qvel': [True],
#     # },
    
# ]

# rl_grid = [
#     # {
#     #     "main_file": ['main_embedded'],
#     #     "env_name": ['Hopper-v2',],

#     #     "_embed_job": [0],
#     #     "source_env": ['Hopper-v2'],

#     #     "start_timesteps": [0],
#     #     "max_timesteps": [1e6],
#     #     "eval_freq": [1e3],
#     #     "render_freq": [1e5],
#     #     "seed": list(range(2)),
#     # },
#     {
#         "main_file": ['main_embedded'],
#         "env_name": ['Swimmer-v2',],

#         # "_embed_job": [1],
#         "decoder": ["embed_states_gym_sweep_envSwimmer-v2_embed-size2"],
#         "source_env": ['Swimmer-v2'],

#         "start_timesteps": [0],
#         "max_timesteps": [1e6],
#         "eval_freq": [1e3],
#         "render_freq": [1e5],
#         "seed": list(range(2)),
#     },
#     # {
#     #     "main_file": ['main_embedded'],
#     #     "env_name": ['Walker2d-v2',],

#     #     "_embed_job": [2],
#     #     "source_env": ['Walker2d-v2'],

#     #     "start_timesteps": [0],
#     #     "max_timesteps": [1e6],
#     #     "eval_freq": [1e3],
#     #     "render_freq": [1e5],
#     #     "seed": list(range(2)),
#     # },
#     {
#         "main_file": ['main_embedded'],
#         "env_name": ['HalfCheetah-v2', ],

#         # "_embed_job": [2],
#         "decoder": ["embed_states_gym_sweep_envHalfCheetah-v2_embed-size6"],
#         "source_env": ['HalfCheetah-v2'],

#         "start_timesteps": [0],
#         "max_timesteps": [1e6],
#         "eval_freq": [1e3],
#         "render_freq": [1e5],
#         "seed": list(range(2)),
#     },
#     {
#         "main_file": ['main_embedded'],
#         "env_name": ['Ant-v2', ],

#         # "_embed_job": [2],
#         "decoder": ["embed_states_gym_sweep_envAnt-v2_embed-size8"],
#         "source_env": ['Ant-v2'],

#         "start_timesteps": [0],
#         "max_timesteps": [1e6],
#         "eval_freq": [1e3],
#         "render_freq": [1e5],
#         "seed": list(range(2)),
#     },
# ]


# basename = "RT_ksweep"
# embed_grid = [
#     # only make one job in each grid — 
#     #   not sure if aligning the RL jobs with these will work otherwise
#     {
#         'main_file': ['main'],
#         'seed': list(range(4)),
#         'epochs': [200],
#         'embed-every': [10000],
#         'qpos-qvel': [True],
#         'traj-len': [1, 2, 4, 6, 8, 10],

#         'env': ['ReacherTurn-v2'],
#     },    
# ]

# rl_grid = [
#     {
#         "main_file": ['main_embedded'],
#         "env_name": [
#             'ReacherTurn-v2',
#         ],

#         # "decoder": ["embed_{}_seed{}_traj-len{}".format(basename, s, k)
#         #             # for s in range(4) for k in [1, 2, 4, 6, 8, 10]],
#         #             for s in range(4) for k in [15, 20, 30]],
#         "_embed_job": list(range(24)),
#         "source_env": ['ReacherTurn-v2'],

#         "start_timesteps": [0],
#         "max_timesteps": [1e6],
#         "eval_freq": [1e2],
#         "render_freq": [1e4],
#         "seed": list(range(1)),
#     },
# ]

basename = "7DPusher_ksweep"
embed_grid = [
    # only make one job in each grid — 
    #   not sure if aligning the RL jobs with these will work otherwise
    {
        'main_file': ['main'],
        'seed': list(range(4)),
        'epochs': [200],
        'embed-every': [10000],
        'qpos-qvel': [True],
        'traj-len': [1, 2, 4, 6, 8, 10],
        'embed-size': [7],

        'env': ['Pusher-v2'],
    },
]

rl_grid = [
    {
        "main_file": ['main_embedded'],
        "env_name": [
            'Pusher-v2',
        ],

        # "decoder": ["embed_{}_seed{}_traj-len{}".format(basename, s, k)
        #             # for s in range(4) for k in [1, 2, 4, 6, 8, 10]],
        #             for s in range(4) for k in [15, 20, 30]],
        "_embed_job": list(range(24)),
        "source_env": ['Pusher-v2'],

        "start_timesteps": [0],
        "max_timesteps": [2e6],
        "eval_freq": [1e4],
        "render_freq": [1e5],
        "seed": list(range(1)),
    },
]


def construct_varying_keys(grids):
    all_keys = set().union(*[g.keys() for g in grids])
    merged = {k: set() for k in all_keys}
    for grid in grids:
        for key in all_keys:
            grid_key_value = grid[key] if key in grid else ["<<NONE>>"]
            merged[key] = merged[key].union(grid_key_value)
    varying_keys = {key for key in merged if len(merged[key]) > 1}
    return varying_keys


def construct_jobs(grids):
    jobs = []
    for grid in grids:
        individual_options = [[{key: value} for value in values]
                              for key, values in grid.items()]
        product_options = list(itertools.product(*individual_options))
        jobs += [{k: v for d in option_set for k, v in d.items()}
                 for option_set in product_options]
    return jobs



def construct_flag_string(job):
    """construct the string of arguments to be passed to the script"""
    flagstring = ""
    for flag in job:
        if not flag in excluded_flags and not flag.startswith('_'):
            if isinstance(job[flag], bool):
                if job[flag]:
                    flagstring = flagstring + " --" + flag
                else:
                    print("WARNING: Excluding 'False' flag " + flag)
            else:
                flagstring = flagstring + " --" + flag + " " + str(job[flag])
    return flagstring

def construct_name(job, varying_keys):
    """construct the job's name out of the varying keys in this sweep"""
    jobname = basename
    for flag in job:
        if flag in varying_keys and not flag.startswith('_'):
            jobname = jobname + "_" + flag + str(job[flag])
    return jobname

# if dry_run:
#     print("NOT starting {} jobs:".format(len(jobs)))
# else:
#     print("Starting {} jobs:".format(len(jobs)))

embed_jobs = construct_jobs(embed_grid)
embed_varying_keys = construct_varying_keys(embed_grid)

for i, job in enumerate(embed_jobs):
    jobname = construct_name(job, embed_varying_keys)
    jobname = "embed_" + jobname
    flagstring = construct_flag_string(job)
    flagstring = flagstring + " --name " + jobname

    slurm_log_dir = 'slurm_logs/' + jobname 
    os.makedirs(os.path.dirname(slurm_log_dir), exist_ok=True)

    true_source_dir = code_dir + '/action-embedding' 
    job_source_dir = code_dir + '/action-embedding-clones/' + jobname
    try:
        os.makedirs(job_source_dir)
        os.system('cp {}/*.py {}'.format(true_source_dir, job_source_dir))
    except FileExistsError:
        # with the 'clear' flag, we're starting fresh
        # overwrite the code that's already here
        if clear:
            print("Overwriting existing files.")
            os.system('cp {}/*.py {}'.format(true_source_dir, job_source_dir))

    jobcommand = "python -u {}/{}.py{}".format(job_source_dir, job['main_file'], flagstring)

    embed_jobs[i]['name'] = jobname
    if local:
        gpu_id = i % 4
        log_path = "slurm_logs/" + jobname
        os.system("env CUDA_VISIBLE_DEVICES={gpu_id} {command} > {log_path}.out 2> {log_path}.err &".format(
                gpu_id=gpu_id, command=jobcommand, log_path=log_path))

    else:
        slurm_script_path = 'slurm_scripts/' + jobname + '.slurm'
        slurm_script_dir = os.path.dirname(slurm_script_path)
        os.makedirs(slurm_script_dir, exist_ok=True)

        job_start_command = "sbatch --parsable " + slurm_script_path

        with open(slurm_script_path, 'w') as slurmfile:
            slurmfile.write("#!/bin/bash\n")
            slurmfile.write("#SBATCH --job-name" + "=" + jobname + "\n")
            slurmfile.write("#SBATCH --open-mode=append\n")
            slurmfile.write("#SBATCH --output=slurm_logs/" +
                            jobname + ".out\n")
            slurmfile.write("#SBATCH --error=slurm_logs/" + jobname + ".err\n")
            slurmfile.write("#SBATCH --export=ALL\n")
            slurmfile.write("#SBATCH --time=1-00\n")
            slurmfile.write("#SBATCH -N 1\n")
            # slurmfile.write("#SBATCH --mem=128gb\n")
            slurmfile.write("#SBATCH --mem=32gb\n")

            slurmfile.write("#SBATCH -c 4\n")
            slurmfile.write("#SBATCH --gres=gpu:1\n")
            slurmfile.write("#SBATCH --constraint=pascal|turing|volta\n")
            slurmfile.write("#SBATCH --exclude=lion[1-26]\n")

            slurmfile.write("cd " + true_source_dir + '\n')

            slurmfile.write(jobcommand)
            slurmfile.write("\n")

        if not dry_run:
            job_subproc_cmd = ["sbatch", "--parsable", slurm_script_path]
            start_result = subprocess.run(job_subproc_cmd, stdout=subprocess.PIPE)
            jobid = start_result.stdout.decode('utf-8').strip()
            embed_jobs[i]['jobid'] = jobid
            print("Started embed job with id {}".format(jobid))


jobs = construct_jobs(rl_grid)
varying_keys = construct_varying_keys(rl_grid)
job_specs = []
for job in jobs:
    dependency = None
    if '_embed_job' in job and job['_embed_job'] is not None:
        embed_job = embed_jobs[job['_embed_job']]
        dependency = embed_job['jobid']
        job['decoder'] = embed_job['name']
        if 'source_env' not in job: job['source_env'] = embed_job['env']

    jobname = construct_name(job, varying_keys.union({'source_env', 'decoder'}))
    flagstring = construct_flag_string(job)

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

    jobcommand = "python -u {}/{}.py{}".format(job_source_dir, job['main_file'], flagstring)

    # jobcommand += " --restart-command '{}'".format(job_start_command)
    job_specs.append((jobname, jobcommand, dependency))


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

    deps = [j[2] for j in current_jobs]
    if any(deps):
        joint_deps = ':'.join([d for d in deps if d is not None])
    else:
        joint_deps = None

    if local:
        gpu_id = i % 4
        log_path = "slurm_logs/" + job_spec[0]
        os.system("env CUDA_VISIBLE_DEVICES={gpu_id} {command} > {log_path}.out 2> {log_path}.err &".format(
                gpu_id=gpu_id, command=job_spec[1], log_path=log_path))

    else:
        slurm_script_path = 'slurm_scripts/' + joint_name + '.slurm'
        slurm_script_dir = os.path.dirname(slurm_script_path)
        os.makedirs(slurm_script_dir, exist_ok=True)

        job_start_command = "sbatch " 
        if joint_deps is not None:
            job_start_command += "--dependency=afterany:{} ".format(joint_deps)
        job_start_command += slurm_script_path

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
            slurmfile.write("#SBATCH --mem=32gb\n")

            slurmfile.write("#SBATCH -c 4\n")
            slurmfile.write("#SBATCH --gres=gpu:1\n")

            # slurmfile.write("#SBATCH -c 40\n")
            slurmfile.write("#SBATCH --constraint=pascal|turing|volta\n")
            slurmfile.write("#SBATCH --exclude=lion[1-26]\n")

            slurmfile.write("cd " + true_source_dir + '\n')

            for job_i, job_spec in enumerate(current_jobs):
                srun_comm = "{command} &".format(name=job_spec[0], command=job_spec[1])
                slurmfile.write(srun_comm)

                slurmfile.write("\n")
            slurmfile.write("wait\n")

        if not dry_run:
            os.system(job_start_command + " &")

    i += increment
