#  Copyright (c) ProrokLab.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
import sys

import pytest
import torch
from tqdm import tqdm

from vmas import make_env


class TestFootball:
    def setup_env(self, n_envs, **kwargs) -> None:
        self.continuous_actions = True

        self.env = make_env(
            scenario="football",
            num_envs=n_envs,
            device="cpu",
            continuous_actions=True,
            # Environment specific variables
            **kwargs,
        )
        self.env.seed(0)

    @pytest.mark.skipif(
        sys.platform.startswith("win32"), reason="Test does not work on windows"
    )
    def test_ai_vs_random(self, n_envs=4, n_agents=3, scoring_reward=1):
        self.setup_env(
            n_red_agents=n_agents,
            n_blue_agents=n_agents,
            ai_red_agents=True,
            ai_blue_agents=False,
            dense_reward=False,
            n_envs=n_envs,
            scoring_reward=scoring_reward,
        )
        all_done = torch.full((n_envs,), False)
        obs = self.env.reset()
        total_rew = torch.zeros(self.env.num_envs, n_agents)
        with tqdm(total=n_envs) as pbar:
            while not all_done.all():
                pbar.update(all_done.sum().item() - pbar.n)
                actions = []
                for _ in range(n_agents):
                    actions.append(torch.rand(n_envs, 2))

                obs, rews, dones, _ = self.env.step(actions)
                for i in range(n_agents):
                    total_rew[:, i] += rews[i]
                if dones.any():
                    # Done envs should have exactly sum of rewards equal to num_agents
                    actual_rew = -scoring_reward * n_agents
                    assert torch.equal(
                        total_rew[dones].sum(-1).to(torch.long),
                        torch.full((dones.sum(),), actual_rew, dtype=torch.long),
                    )
                    total_rew[dones] = 0
                    all_done += dones
                    for env_index, done in enumerate(dones):
                        if done:
                            self.env.reset_at(env_index)

    @pytest.mark.skipif(
        sys.platform.startswith("win32"), reason="Test does not work on windows"
    )
    def test_soft_reset_and_kickoff_transitions(self, n_envs=2, n_agents=2):
        self.setup_env(
            n_red_agents=n_agents,
            n_blue_agents=n_agents,
            ai_red_agents=False,
            ai_blue_agents=False,
            dense_reward=False,
            terminate_on_goal=False,
            observe_is_kickoff=True,
            max_steps=50,
            n_envs=n_envs,
            scoring_reward=10.0,
        )
        obs = self.env.reset()
        for agent_obs in obs:
            assert torch.all(agent_obs[:, -1] == 1.0), "Initial obs must have is_kickoff=1.0"

        actions = [torch.zeros(n_envs, 2) for _ in range(n_agents * 2)]

        # Step 1: Normal step -> is_kickoff transitions to 0.0
        obs, rews, dones, infos = self.env.step(actions)
        for agent_obs in obs:
            assert torch.all(agent_obs[:, -1] == 0.0), "Active play obs must have is_kickoff=0.0"
        assert not dones.any()

        # Place ball into right goal in env 0
        self.env.world.ball.state.pos[0, 0] = self.env.scenario.pitch_length / 2 + 0.1
        self.env.world.ball.state.pos[0, 1] = 0.0

        # Step 2: Goal scored in env 0
        obs, rews, dones, infos = self.env.step(actions)
        assert infos[0]["blue_score"][0] == True
        assert infos[0]["blue_score"][1] == False
        assert infos[0]["blue_goals"][0] == 1
        assert infos[0]["blue_goals"][1] == 0
        assert not dones.any(), "dones must remain False on goal when terminate_on_goal=False"

        # Env 0 had soft reset: ball back to center, is_kickoff=1.0
        assert torch.allclose(self.env.world.ball.state.pos[0], torch.zeros(2), atol=1e-3)
        for agent_obs in obs:
            assert agent_obs[0, -1] == 1.0, "Env 0 must have is_kickoff=1.0 after soft reset"
            assert agent_obs[1, -1] == 0.0, "Env 1 must have is_kickoff=0.0"

        # Step 3: Next step in env 0 transitions is_kickoff back to 0.0
        obs, rews, dones, infos = self.env.step(actions)
        for agent_obs in obs:
            assert torch.all(agent_obs[:, -1] == 0.0)

        # Place ball into left goal in env 1
        self.env.world.ball.state.pos[1, 0] = -self.env.scenario.pitch_length / 2 - 0.1
        self.env.world.ball.state.pos[1, 1] = 0.0

        # Step 4: Goal scored in env 1 by Red
        obs, rews, dones, infos = self.env.step(actions)
        assert infos[0]["red_score"][1] == True
        assert infos[0]["red_goals"][1] == 1
        assert not dones.any()
        assert torch.allclose(self.env.world.ball.state.pos[1], torch.zeros(2), atol=1e-3)
        for agent_obs in obs:
            assert agent_obs[1, -1] == 1.0
            assert agent_obs[0, -1] == 0.0

    @pytest.mark.skipif(
        sys.platform.startswith("win32"), reason="Test does not work on windows"
    )
    def test_continuous_match_with_ai(self, n_envs=4, n_agents=3, max_steps=200):
        self.setup_env(
            n_red_agents=n_agents,
            n_blue_agents=n_agents,
            ai_red_agents=True,
            ai_blue_agents=False,
            dense_reward=False,
            terminate_on_goal=False,
            observe_is_kickoff=True,
            max_steps=max_steps,
            n_envs=n_envs,
            scoring_reward=10.0,
        )
        obs = self.env.reset()
        for agent_obs in obs:
            assert torch.all(agent_obs[:, -1] == 1.0)

        goals_seen = 0
        for step in range(max_steps):
            actions = [torch.rand(n_envs, 2) for _ in range(n_agents)]
            obs, rews, dones, infos = self.env.step(actions)

            if step < max_steps - 1:
                assert not dones.any(), f"Dones should remain False before max_steps, got {dones} at step {step}"
            else:
                assert dones.all(), f"Dones should be True at max_steps, got {dones}"

            red_scored = infos[0]["red_score"]
            blue_scored = infos[0]["blue_score"]
            goal_scored = red_scored | blue_scored

            if goal_scored.any():
                goals_seen += goal_scored.sum().item()
                for agent_obs in obs:
                    assert torch.all(agent_obs[goal_scored, -1] == 1.0)
                    if (~goal_scored).any():
                        assert torch.all(agent_obs[~goal_scored, -1] == 0.0)
                ball_pos = self.env.world.ball.state.pos[goal_scored]
                assert torch.allclose(
                    ball_pos,
                    torch.zeros_like(ball_pos),
                    atol=1e-3,
                )

        assert goals_seen > 0, "AI red agents should have scored at least one goal in 200 steps"
        total_red_goals = infos[0]["red_goals"].sum().item()
        total_blue_goals = infos[0]["blue_goals"].sum().item()
        assert total_red_goals + total_blue_goals == goals_seen

    @pytest.mark.skipif(
        sys.platform.startswith("win32"), reason="Test does not work on windows"
    )
    def test_dict_obs_kickoff(self, n_envs=2, n_agents=2):
        self.setup_env(
            n_red_agents=n_agents,
            n_blue_agents=n_agents,
            ai_red_agents=True,
            ai_blue_agents=False,
            terminate_on_goal=False,
            observe_is_kickoff=True,
            dict_obs=True,
            n_envs=n_envs,
        )
        obs = self.env.reset()
        for agent_obs in obs:
            assert "is_kickoff" in agent_obs
            assert agent_obs["is_kickoff"].shape == (n_envs, 1)
            assert torch.all(agent_obs["is_kickoff"] == 1.0)
