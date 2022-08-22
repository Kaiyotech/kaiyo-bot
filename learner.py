import wandb
import torch.jit

from torch.nn import Linear, Sequential, GELU

from redis import Redis
from rocket_learn.agent.actor_critic_agent import ActorCriticAgent
from rocket_learn.agent.discrete_policy import DiscretePolicy
from rocket_learn.ppo import PPO
from rocket_learn.rollout_generator.redis.redis_rollout_generator import RedisRolloutGenerator
from rlgym_tools.extra_obs.advanced_padder import AdvancedObsPadder

from N_Parser import NectoAction
import numpy as np
from zero_sum_rewards import ZeroSumReward
from Constants import FRAME_SKIP

import os
from torch import set_num_threads
from rocket_learn.utils.stat_trackers.common_trackers import Speed, Demos, TimeoutRate, Touch, EpisodeLength, Boost, \
    BehindBall, TouchHeight, DistToBall, AirTouch, AirTouchHeight, BallHeight, BallSpeed, CarOnGround, GoalSpeed,\
    MaxGoalSpeed

set_num_threads(1)

if __name__ == "__main__":
    frame_skip = FRAME_SKIP
    half_life_seconds = 12  # 8 -> 12 at 12.53b
    fps = 120 / frame_skip
    gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))
    print(f"_gamma is: {gamma}")
    config = dict(
        actor_lr=1e-5,
        critic_lr=1e-5,

        n_steps=2_000_000,  # polishing at 13.1b
        batch_size=200_000,
        minibatch_size=50_000,
        epochs=30,
        gamma=gamma,
        save_every=5,
        model_every=30,
        ent_coef=0.01,
    )

    run_id = "V01"
    wandb.login(key=os.environ["WANDB_KEY"])
    logger = wandb.init(dir="./wandb_store",
                        name="KaiBumBot_v01",
                        project="KaiBumBot",
                        entity="kaiyotech",
                        id=run_id,
                        config=config,
                        )
    redis = Redis(username="user1", password=os.environ["redis_user1_key"])  # host="192.168.0.201",
    redis.delete("worker-ids")

    stat_trackers = [
        Speed(), Demos(), TimeoutRate(), Touch(), EpisodeLength(), Boost(), BehindBall(), TouchHeight(), DistToBall(),
        AirTouch(), AirTouchHeight(), BallHeight(), BallSpeed(), CarOnGround(), GoalSpeed(), MaxGoalSpeed(),
    ]

    rollout_gen = RedisRolloutGenerator("KaiBumBot",
                                        redis,
                                        lambda: AdvancedObsPadder(team_size=3, expanding=True),
                                        lambda: ZeroSumReward(),
                                        lambda: NectoAction(),
                                        save_every=logger.config.save_every,
                                        model_every=logger.config.model_every,
                                        logger=logger,
                                        clear=False,
                                        stat_trackers=stat_trackers,
                                        # gamemodes=("1v1", "2v2", "3v3"),
                                        max_age=1,
                                        )

    critic = Sequential(Linear(237, 512), GELU(), Linear(512, 512), GELU(),

                        Linear(512, 512), GELU(), Linear(512, 512), GELU(), Linear(512, 512),
                        GELU(), Linear(512, 512), GELU(), Linear(512, 1))

    actor = Sequential(Linear(237, 512), GELU(), Linear(512, 512), GELU(),
                       Linear(512, 512), GELU(), Linear(512, 512), GELU(), Linear(512, 90))

    actor = DiscretePolicy(actor, (90,))

    optim = torch.optim.Adam([
        {"params": actor.parameters(), "lr": logger.config.actor_lr},
        {"params": critic.parameters(), "lr": logger.config.critic_lr}

    ])

    agent = ActorCriticAgent(actor=actor, critic=critic, optimizer=optim)

    model_parameters = filter(lambda p: p.requires_grad, agent.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"There are {params} trainable parameters")
    alg = PPO(
        rollout_gen,
        agent,
        ent_coef=logger.config.ent_coef,
        n_steps=logger.config.n_steps,
        batch_size=logger.config.batch_size,
        minibatch_size=logger.config.minibatch_size,
        epochs=logger.config.epochs,
        gamma=logger.config.gamma,
        logger=logger,
        zero_grads_with_none=True,
    )

    alg.load("model_saves/KaiBumBot_1661133768.9422624/KaiBumBot_15730/checkpoint.pt")
    alg.agent.optimizer.param_groups[0]["lr"] = logger.config.actor_lr
    alg.agent.optimizer.param_groups[1]["lr"] = logger.config.critic_lr

    alg.run(iterations_per_save=logger.config.save_every, save_dir="model_saves")



