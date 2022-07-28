from rlgym.utils.common_values import BLUE_TEAM, BLUE_GOAL_BACK, ORANGE_GOAL_BACK, ORANGE_TEAM, BALL_MAX_SPEED, \
    CAR_MAX_SPEED, BALL_RADIUS, GOAL_HEIGHT, CEILING_Z
from rlgym.utils.reward_functions.common_rewards.misc_rewards import *
from rlgym.utils.reward_functions.common_rewards.conditional_rewards import *
from rlgym.utils.reward_functions import RewardFunction

from numpy.linalg import norm

from typing import Tuple, List


def _closest_to_ball(state: GameState) -> Tuple[int, int]:
    #returns [blue_closest, orange_closest]
    length = len(state.players)
    dist_list: List[float] = [100_000] * length
    blue_closest = -1
    orange_closest = -1
    for i, player in enumerate(state.players):
        dist = np.linalg.norm((a - b for a, b in zip(player.car_data.position, state.ball.position)))
        dist_list[i] = dist
        if state.players[i].team_num == BLUE_TEAM and blue_closest == -1:
            blue_closest = i
        elif state.players[i].team_num == ORANGE_TEAM and orange_closest == -1:
            orange_closest = i
        elif state.players[i].team_num == BLUE_TEAM and dist <= dist_list[blue_closest]:
            if dist == dist_list[blue_closest]:
                if state.players[i].car_data.position[0] > state.players[blue_closest].car_data.position[0]:
                    blue_closest = i
                    continue
            else:
                blue_closest = i
                continue
        elif state.players[i].team_num == ORANGE_TEAM and dist <= dist_list[orange_closest]:
            if dist == dist_list[orange_closest]:
                if state.players[i].car_data.position[0] < state.players[orange_closest].car_data.position[0]:
                    orange_closest = i
                    continue
            else:
                orange_closest = i
                continue
    return blue_closest, orange_closest

# def _closest_to_ball(player: PlayerData, state: GameState) -> bool:
#     player_dist = np.linalg.norm(player.car_data.position - state.ball.position)
#     for p in state.players:
#         if p.team_num == player.team_num and p.car_id != p.car_id:
#             dist = np.linalg.norm(p.car_data.position - state.ball.position)
#             if dist < player_dist:
#                 return False
#             if dist == player_dist:  # left goes!
#                 if player.team_num == BLUE_TEAM and player.car_data.position[0] > p.car_data.position[0]:
#                     return True
#                 if player.team_num == ORANGE_TEAM and player.car_data.position[0] < p.car_data.position[0]:
#                     return True
#     return True


class ZeroSumReward(RewardFunction):

    def __init__(
        self,
        goal_w=10,
        # concede_w=0,
        # velocity_pb_w=0,
        # dist_pb_w=0,
        # face_ball_w=0,
        velocity_bg_w=0.05,
        # kickoff_w=0,
        # ball_touch_w=0,
        # touch_grass_w=0,
        acel_ball_w=0.1,
        boost_gain_w=1,  # 1.5 -> 0.5 at 7.18b, was maybe 1 originally?
        boost_spend_w=0.025,  # 0.033 -> 0.025 7.18b
        # ball_touch_dribble_w=0,
        jump_touch_w=2,
        # wall_touch_w=0,
        cons_air_touches_w=5,
        demo_w=4,  # 3->4 at 6.87b
        # got_demoed_w=0,
        kickoff_w=0.05,
        tick_skip=8,
        team_spirit=1,
    ):
        self.goal_w = goal_w
        # self.concede_w = concede_w
        # self.velocity_pb_w = velocity_pb_w
        self.velocity_bg_w = velocity_bg_w
        # self.ball_touch_w = ball_touch_w
        # self.kickoff_w = kickoff_w
        # self.touch_grass_w = touch_grass_w
        self.acel_ball_w = acel_ball_w
        self.boost_gain_w = boost_gain_w
        self.boost_spend_w = boost_spend_w
        # self.ball_touch_dribble_w = ball_touch_dribble_w
        self.jump_touch_w = jump_touch_w
        # self.wall_touch_w = wall_touch_w
        self.cons_air_touches_w = cons_air_touches_w
        self.demo_w = demo_w
        # self.got_demoed_w = got_demoed_w
        # self.dist_pb_w = dist_pb_w
        # self.face_ball_w = face_ball_w
        self.kickoff_w = kickoff_w
        self.rewards = None
        self.current_state = None
        self.last_state = None
        self.touch_timeout = 8 * 120 // tick_skip  # 120 ticks at 8 tick skip is 8 seconds
        self.kickoff_timeout = 5 * 120 // tick_skip
        self.kickoff_timer = 0
        self.closest_reset_blue = -1
        self.closest_reset_orange = -1
        self.blue_touch_timer = self.touch_timeout + 1
        self.orange_touch_timer = self.touch_timeout + 1
        self.blue_toucher = None
        self.orange_toucher = None
        # self.last_toucher = None
        self.team_spirit = team_spirit
        self.n = 0
        self.cons_touches = 0

    def pre_step(self, state: GameState):
        if state != self.current_state:
            self.last_state = self.current_state
            self.current_state = state
            self.n = 0
            self.blue_touch_timer += 1
            self.orange_touch_timer += 1
            self.kickoff_timer += 1
        # Calculate rewards
        player_rewards = [0] * (len(state.players))
        player_self_rewards = [0] * (len(state.players))
        # ball_height = state.ball.position[2]

        for i, player in enumerate(state.players):
            last = self.last_state.players[i]

            if player.ball_touched:
                # self.last_toucher = i
                if player.team_num == BLUE_TEAM:
                    self.blue_toucher = i
                    self.blue_touch_timer = 0
                else:
                    self.orange_toucher = i
                    self.orange_touch_timer = 0

                # ball touch
                # player_rewards[i] += self.ball_touch_w
                # if state.ball.position[2] > 145 and player.on_ground and player.car_data.position[2] < 20:
                #     player_self_rewards[i] += self.ball_touch_dribble_w

                # acel_ball
                vel_difference = abs(np.linalg.norm((a - b for a, b in zip(self.last_state.ball.linear_velocity,
                                                                          self.current_state.ball.linear_velocity))))
                player_rewards[i] += vel_difference / 4600.0

                # jump touch
                min_height = 120
                max_height = CEILING_Z - BALL_RADIUS
                rnge = max_height - min_height
                if not player.on_ground and state.ball.position[2] > min_height:
                    player_rewards[i] += self.jump_touch_w * (state.ball.position[2] - min_height) / rnge

                # wall touch
                # min_height = 350
                # if player.on_ground and state.ball.position[2] > min_height:
                #     player_self_rewards[i] += self.wall_touch_w * (state.ball.position[2] - min_height) / rnge

                # cons air touches, max reward of 5, normalized to 1, initial reward 1.4, only starts at second touch
                if state.ball.position[2] > 140 and last.ball_touched and not player.on_ground:
                    self.cons_touches += 1
                    player_rewards[i] += self.cons_air_touches_w * min((1.4 ** self.cons_touches), 5) / 5
                else:
                    self.cons_touches = 0
            # not touched by this player
            else:
                if last.ball_touched:
                    self.cons_touches = 0

            # vel bg
            if self.blue_toucher is not None or self.orange_toucher is not None:
                if player.team_num == BLUE_TEAM:
                    objective = ORANGE_GOAL_BACK
                else:
                    objective = BLUE_GOAL_BACK
                vel = state.ball.linear_velocity
                pos_diff = (a - b for a, b in zip(objective, state.ball.position))
                norm_pos_diff = (a / b for a, b in zip(pos_diff, np.linalg.norm(pos_diff)))
                norm_vel = (a / b for a, b in zip(vel, BALL_MAX_SPEED))
                vel_bg_reward = float(np.dot(norm_pos_diff, norm_vel))
                player_rewards[i] += self.velocity_bg_w * vel_bg_reward

            # # face ball
            # pos_diff = state.ball.position - player.car_data.position
            # norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
            # player_self_rewards[i] += self.face_ball_w * float(np.dot(player.car_data.forward(), norm_pos_diff))
            #
            # # player dist ball
            # dist = np.linalg.norm(player.car_data.position - state.ball.position) - BALL_RADIUS
            # player_self_rewards[i] += self.dist_pb_w * np.exp(-0.5 * dist / CAR_MAX_SPEED)

            # boost
            # don't punish or reward boost when above  approx single jump height
            if player.car_data.position[2] < 2 * BALL_RADIUS:
                boost_diff = player.boost_amount - last.boost_amount
                if boost_diff > 0:
                    player_rewards[i] += self.boost_gain_w * boost_diff
                # elif norm(player.car_data.linear_velocity) >= CAR_MAX_SPEED * 0.9504:  # greater than supersonic?
                #     player_rewards[i] += 5 * self.boost_spend_w * boost_diff
                else:
                    player_rewards[i] += self.boost_spend_w * boost_diff

            # # touch_grass
            # if player.on_ground and player.car_data.position[2] < 25:
            #     player_self_rewards[i] += self.touch_grass_w

            # demo
            # if player.is_demoed and not last.is_demoed:
            #     player_rewards[i] += self.got_demoed_w
            if player.match_demolishes > last.match_demolishes:
                player_rewards[i] += self.demo_w

            # # vel pb - removing the negative clamping
            # vel = player.car_data.linear_velocity
            # pos_diff = state.ball.position - player.car_data.position
            # norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
            # norm_vel = vel / CAR_MAX_SPEED
            # speed_rew = float(np.dot(norm_pos_diff, norm_vel))
            # # if speed_rew < 0:
            # #     speed_rew /= 10
            # #     speed_rew = max(speed_rew, -0.25)
            # player_self_rewards[i] += self.velocity_pb_w * speed_rew

            # # kickoff extra reward for speed to ball
            # if state.ball.position[0] == 0 and state.ball.position[1] == 0 and \
            #         self.kickoff_timer < self.kickoff_timeout:
            #     player_self_rewards[i] += self.kickoff_w * speed_rew

            # acel_car in forward direction - this doesn't work yet, needs forward direction maybe?
            # but half flip is good, maybe just a bad idea
            # curr_car_vel = player.car_data.linear_velocity
            # last_car_vel = last.car_data.linear_velocity
            # player_rewards[i] += self.acel_car_w * (norm(curr_car_vel - last_car_vel) / CAR_MAX_SPEED)

            # kickoff reward
            if state.ball.position[0] == 0 and state.ball.position[1] == 0 and \
                    (self.closest_reset_blue == i or self.closest_reset_orange == i) and \
                    self.kickoff_timer < self.kickoff_timeout:
                player_self_rewards[i] += self.kickoff_w * -1

        mid = len(player_rewards) // 2

        # Handle goals with no scorer for critic consistency,
        # random state could send ball straight into goal
        if self.blue_touch_timer < self.touch_timeout or self.orange_touch_timer < self.touch_timeout:
            d_blue = state.blue_score - self.last_state.blue_score
            d_orange = state.orange_score - self.last_state.orange_score
            if d_blue > 0:
                goal_speed = norm(self.last_state.ball.linear_velocity)
                goal_reward = self.goal_w * (goal_speed / (CAR_MAX_SPEED * 1.25))
                if self.blue_touch_timer < self.touch_timeout:
                    player_rewards[self.blue_toucher] += (1 - self.team_spirit) * goal_reward
                    player_rewards[:mid] += self.team_spirit * goal_reward
                # player_rewards[mid:] += self.concede_w
            if d_orange > 0:
                goal_speed = norm(self.last_state.ball.linear_velocity)
                goal_reward = self.goal_w * (goal_speed / (CAR_MAX_SPEED * 1.25))
                if self.orange_touch_timer < self.touch_timeout:
                    player_rewards[self.orange_toucher] += (1 - self.team_spirit) * goal_reward
                    player_rewards[mid:] += self.team_spirit * goal_reward
                # player_rewards[:mid] += self.concede_w

        # zero mean
        orange_mean = np.mean(player_rewards[mid:])
        blue_mean = np.mean(player_rewards[:mid])
        player_rewards[:mid] -= orange_mean
        player_rewards[mid:] -= blue_mean

        self.last_state = state
        self.rewards = player_rewards + player_self_rewards
        # print(self.rewards)

    def reset(self, initial_state: GameState):
        self.n = 0
        self.last_state = None
        self.rewards = None
        self.current_state = initial_state
        self.blue_toucher = None
        self.orange_toucher = None
        # self.last_toucher = None
        self.blue_touch_timer = self.touch_timeout + 1
        self.orange_touch_timer = self.touch_timeout + 1
        self.cons_touches = 0
        self.kickoff_timer = 0
        self.closest_reset_blue, self.closest_reset_orange = _closest_to_ball(initial_state)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        rew = self.rewards[self.n]
        self.n += 1
        return float(rew)
