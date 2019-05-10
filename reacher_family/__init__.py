from gym.envs.registration import registry, register, make, spec

from reacher_family.reacher_push import ReacherPushEnv, ReacherPushSparseEnv
from reacher_family.reacher_vertical import ReacherVerticalEnv, ReacherVerticalSparseEnv
from reacher_family.reacher_spin import ReacherSpinEnv, ReacherSpinSparseEnv
from reacher_family.reacher_turn import ReacherTurnEnv, ReacherTurnSparseEnv
from reacher_family.reacher_test import ReacherTestEnv, ReacherTestSparseEnv


register(
    id='ReacherVertical-v2',
    entry_point='reacher_family:ReacherVerticalEnv',
    max_episode_steps=100,
    # reward_threshold=-3.75,
)

register(
    id='ReacherTest-v2',
    entry_point='reacher_family:ReacherTestEnv',
    max_episode_steps=100,
    # reward_threshold=-3.75,
)

register(
    id='ReacherTestSparse-v2',
    entry_point='reacher_family:ReacherTestSparseEnv',
    max_episode_steps=100,
    # reward_threshold=-3.75,
)

register(
    id='ReacherPush-v2',
    entry_point='reacher_family:ReacherPushEnv',
    max_episode_steps=100,
    # reward_threshold=-3.75,
)

register(
    id='ReacherSpin-v2',
    entry_point='reacher_family:ReacherSpinEnv',
    max_episode_steps=100,
    # reward_threshold=-3.75,
)

register(
    id='ReacherTurn-v2',
    entry_point='reacher_family:ReacherTurnEnv',
    max_episode_steps=100,
    # reward_threshold=-3.75,
)

register(
    id='ReacherVerticalSparse-v2',
    entry_point='reacher_family:ReacherVerticalSparseEnv',
    max_episode_steps=100,
    # reward_threshold=-3.75,
)
register(
    id='ReacherPushSparse-v2',
    entry_point='reacher_family:ReacherPushSparseEnv',
    max_episode_steps=100,
    # reward_threshold=-3.75,
)
register(
    id='ReacherSpinSparse-v2',
    entry_point='reacher_family:ReacherSpinSparseEnv',
    max_episode_steps=100,
    # reward_threshold=-3.75,
)
register(
    id='ReacherTurnSparse-v2',
    entry_point='reacher_family:ReacherTurnSparseEnv',
    max_episode_steps=100,
    # reward_threshold=-3.75,
)
