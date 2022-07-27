import sys
from redis import Redis
from redis.retry import Retry
from redis.backoff import ExponentialBackoff
from redis.exceptions import ConnectionError, TimeoutError
from rlgym.envs import Match
from rlgym_tools.extra_obs.advanced_padder import AdvancedObsPadder
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, NoTouchTimeoutCondition, GoalScoredCondition
from rocket_learn.rollout_generator.redis.redis_rollout_worker import RedisRolloutWorker
from kaiyo_setter import KaiyoSetter
from N_Parser import NectoAction
#from kaiyo_rewards import KaiyoRewards
from zero_sum_rewards import ZeroSumReward
from pretrained_agents.necto.necto_v1 import NectoV1
from pretrained_agents.necto.nexto_v2 import NextoV2
from torch import set_num_threads
import os
set_num_threads(1)


if __name__ == "__main__":
    rew = ZeroSumReward()
    frame_skip = 8
    fps = 120 // frame_skip
    name = "Default"
    send_gamestate = False
    local = True
    host = "127.0.0.1"
    if len(sys.argv) > 1:
        host = sys.argv[1]
        if host != "127.0.0.1" and host != "localhost":
            local = False
    if len(sys.argv) > 2:
        name = sys.argv[2]
    if len(sys.argv) > 3:
        if sys.argv[3] == 'GAMESTATE':
            send_gamestate = True

    match = Match(
        game_speed=100,
        spawn_opponents=True,
        team_size=3,
        state_setter=KaiyoSetter(),
        obs_builder=AdvancedObsPadder(team_size=3, expanding=True),
        action_parser=NectoAction(),
        terminal_conditions=[TimeoutCondition(fps * 300), NoTouchTimeoutCondition(fps * 45), GoalScoredCondition()],
        reward_function=rew
    )

    # local Redis
    if local:
        r = Redis(host=host,
                  username="user1",
                  password=os.environ["redis_user1_key"],
                  )

    # remote Redis
    else:
        # noinspection PyArgumentList
        r = Redis(host=host,
                  username="user1",
                  password=os.environ["redis_user1_key"],
                  retry_on_error=[ConnectionError, TimeoutError],
                  retry=Retry(ExponentialBackoff(cap=10, base=1), 25)
                  )

    model_name = "necto-model-30Y.pt"
    nectov1 = NectoV1(model_string=model_name, n_players=6)
    model_name = "nexto-model.pt"
    nexto = NextoV2(model_string=model_name, n_players=6)

    pretrained_agents = {nectov1: 0, nexto: .1}

    RedisRolloutWorker(r, name, match,
                       past_version_prob=0.2,
                       sigma_target=2,
                       evaluation_prob=0.01,
                       force_paging=True,
                       dynamic_gm=True,
                       send_obs=True,
                       auto_minimize=True,
                       send_gamestates=send_gamestate,
                       pretrained_agents=pretrained_agents,
                       ).run()


