# File: drlfusion/modelling/utils/rollout_tools.py

""" 
A unified DRLFusion rollout and batch processing toolkit combining:

- safe environment handling
- on-policy data collection
- off-policy replay filling
- batch formatting (GAE & replay)
"""

from typing import Callable, Dict, List, Tuple, Any, Union
import numpy as np
from drlfusion.core.utils.logger import get_logger

logger = get_logger("rollout_tools")


# === 1. ENVIRONMENT UNPACKING ===

def safe_step(step_result: Union[tuple, list]) -> Tuple[Any, float, bool, bool]:
    """
    Unpacks env.step() result safely across Gym/Gymnasium versions.

    Args:
        step_result (tuple | list): Output from env.step()

    Returns:
        next_state (Any): Next observation
        reward (float): Scalar reward
        done (bool): Done flag
        truncated (bool): Truncated flag
    """
    if len(step_result) == 5:
        next_state, reward, done, truncated, _ = step_result
    elif len(step_result) == 4:
        next_state, reward, done, truncated = step_result
    elif len(step_result) == 3:
        next_state, reward, done = step_result
        truncated = False
    else:
        raise ValueError(f"Unexpected step return: {step_result}")
    return next_state, reward, done, truncated

def safe_reset(env: Any) -> Any:
    """
    Unpacks env.reset() result safely across Gym/Gymnasium versions.

    Args:
        env (Any): Gym-compatible environment

    Returns:
        Any: Initial observation
    """
    result = env.reset()
    return result[0] if isinstance(result, (tuple, list)) and len(result) == 2 else result


# === 2. ON-POLICY EXPERIENCE COLLECTION ===

def collect_on_policy(
    env,
    choose_action_fn: Callable,
    batch_size: int = 2048
) -> Dict[str, list]:
    """
    Collects trajectory rollout for PPO/A2C.

    Args:
        env: Gym environment.
        choose_action_fn: agent.choose_action(state)
        batch_size: Max steps to collect.

    Returns:
        memory dict with: states, actions, log_probs, values, rewards, dones
    """
    memory = {"states": [], "actions": [], "log_probs": [], "values": [], "rewards": [], "dones": []}
    state = safe_reset(env)
    done, truncated = False, False
    steps = 0

    while steps < batch_size and not done and not truncated:
        action, log_prob, value, _ = choose_action_fn(state)
        step_result = env.step(action)
        next_state, reward, done, truncated = safe_step(step_result)

        memory["states"].append(state)
        memory["actions"].append(action)
        memory["log_probs"].append(log_prob)
        memory["values"].append(value)
        memory["rewards"].append(reward)
        memory["dones"].append(done or truncated)

        state = next_state
        steps += 1

    logger.debug(f"✅ On-policy: {steps} steps collected.")
    return memory


# === 3. OFF-POLICY REPLAY COLLECTION ===

def collect_off_policy(
    env,
    choose_action_fn: Callable,
    buffer: List[Tuple],
    max_steps: int = 2048
) -> float:
    """
    Collects experience from one rollout and appends valid (s, a, r, s', done) samples to a replay buffer.

    Args:
        env: Gym-compatible environment
        choose_action_fn: Function(state) -> action OR (action, ...)
        buffer (List[Tuple]): The replay buffer to populate
        max_steps (int): Max number of steps to collect

    Returns:
        float: Total episode reward
    """
    state = safe_reset(env)
    done, truncated = False, False
    total_reward = 0.0
    steps = 0

    while steps < max_steps and not done and not truncated:
        try:
            output = choose_action_fn(state)

            # ✅ Extract action (first element if tuple, else assume raw action)
            action = output[0] if isinstance(output, tuple) else output

            # ✅ Convert single-item arrays to scalars
            if isinstance(action, (list, tuple, np.ndarray)) and np.array(action).size == 1:
                action = np.array(action).item()

            step_result = env.step(action)
            next_state, reward, done, truncated = safe_step(step_result)

            sample = (state, action, reward, next_state, done or truncated)

            if len(sample) == 5:
                buffer.append(sample)
            else:
                logger.warning(f"⚠️ Malformed sample skipped (len={len(sample)}): {sample}")

            total_reward += reward
            state = next_state
            steps += 1

        except Exception as e:
            logger.error(f"❌ Error during rollout step {steps}: {e}")
            break

    logger.debug(f"✅ Off-policy: {steps} steps collected | Reward: {total_reward:.2f}")
    return total_reward


# === 4. GAE BATCH BUILDER ===

def build_gae_batch(
    memory: Dict[str, List],
    gamma: float = 0.99,
    lam: float = 0.95,
    is_ppo: bool = False
) -> Dict[str, List]:
    """
    Creates advantage-based batch for A2C/PPO.
    """
    if not memory or len(memory.get("rewards", [])) == 0:
        logger.warning("❌ Empty memory for GAE.")
        return {"states": [], "actions": [], "returns": [], "advantages": []}

    rewards, values, dones = memory["rewards"], memory["values"], memory["dones"]
    last_value = values[-1] if values else 0
    returns, advantages = [], []
    gae = 0.0

    for t in reversed(range(len(rewards))):
        next_val = values[t + 1] if t + 1 < len(values) else last_value
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        returns.insert(0, gae + values[t])
        advantages.insert(0, gae)

    batch = {
        "states": memory["states"],
        "actions": memory["actions"],
        "returns": returns,
        "advantages": advantages
    }
    if is_ppo:
        batch["log_probs"] = memory["log_probs"]

    logger.debug(f"✅ GAE batch built ({len(returns)} steps).")
    return batch


# === 5. REPLAY BATCH BUILDER ===

def build_replay_batch(replay: List[Tuple]) -> Dict[str, list]:
    """
    Converts a list of replay tuples into a training batch dictionary for SAC/DQN.

    Args:
        replay (List[Tuple]): Each entry must be (state, action, reward, next_state, done)

    Returns:
        Dict[str, list]: Batch with keys: states, actions, rewards, next_states, dones
    """
    if not replay:
        logger.warning("❌ Replay buffer is empty.")
        return {
            "states": [], "actions": [], "rewards": [], "next_states": [], "dones": []
        }

    batch = {
        "states": [],
        "actions": [],
        "rewards": [],
        "next_states": [],
        "dones": []
    }

    for i, sample in enumerate(replay):
        if not isinstance(sample, tuple) or len(sample) != 5:
            logger.error(f"❌ Invalid replay sample at index {i} (len={len(sample)}): {sample}")
            continue

        state, action, reward, next_state, done = sample
        batch["states"].append(state)
        batch["actions"].append(action)
        batch["rewards"].append(reward)
        batch["next_states"].append(next_state)
        batch["dones"].append(done)

    logger.debug(f"✅ Replay batch built with {len(batch['states'])} valid samples.")
    return batch

