"""AI Gym Tower Defense — a tower defense game with a Gymnasium API for AI research."""

from ai_gym_td import gym_register  # noqa: F401  (registers TowerDefense-v0)
from ai_gym_td.config import EnemySpec, GameConfig, TowerSpec
from ai_gym_td.engine import TowerDefenseGame
from ai_gym_td.env import TowerDefenseEnv
from ai_gym_td.legality import ActionLegalityTracker, get_legal_actions, is_action_legal, repair_action

# Optional LLM agent imports (require API keys and external packages)
try:
    from ai_gym_td.llm_agents import (
        AnthropicAgent,
        GoogleAgent,
        LLMAgent,
        OllamaAgent,
        OpenAIAgent,
        create_llm_agent,
    )
    from ai_gym_td.obs_format import format_obs_for_llm

    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

__version__ = "0.2.0"
__all__ = [
    "LLM_AVAILABLE",
    "ActionLegalityTracker",
    "AnthropicAgent",
    "EnemySpec",
    "GameConfig",
    "GoogleAgent",
    "LLMAgent",
    "OllamaAgent",
    "OpenAIAgent",
    "TowerDefenseEnv",
    "TowerDefenseGame",
    "TowerSpec",
    "create_llm_agent",
    "format_obs_for_llm",
    "get_legal_actions",
    "is_action_legal",
    "repair_action",
]
