"""Unit and integration tests for segmented PPO training loop and fallback."""

import os

from towermind.experiments.manifest import ExperimentConfig, ExperimentManifest
from towermind.llm.gateway import GatewayConfig, LLMGateway
from towermind.llm.mock_provider import MockLLMProvider
from towermind.meta_controller.loop import run_segmented_experiment
from towermind.rewards.registry import RewardProfileRegistry


def test_segmented_training_loop_end_to_end(tmp_path):
    provider = MockLLMProvider()
    gateway = LLMGateway(provider, GatewayConfig(cache_enabled=False))
    registry = RewardProfileRegistry()

    config = ExperimentConfig(
        experiment_id="test_exp_001",
        seed=10,
        total_timesteps=512,  # 2 segments of 256
        segment_timesteps=256,
        eval_episodes=2,
        num_steps=128,
    )

    ckpt_dir = str(tmp_path / "checkpoints")

    manifest = run_segmented_experiment(
        config=config,
        gateway=gateway,
        registry=registry,
        checkpoint_dir=ckpt_dir,
    )

    assert isinstance(manifest, ExperimentManifest)
    assert manifest.experiment_id == "test_exp_001"
    assert manifest.baseline_metrics is not None
    assert manifest.final_metrics is not None
    assert len(manifest.segments) == 2
    assert len(manifest.checkpoint_paths) == 2

    # Verify checkpoint files exist
    for p in manifest.checkpoint_paths:
        assert os.path.exists(p)

    # Verify manifest was saved
    manifest_file = os.path.join(ckpt_dir, "manifest.json")
    assert os.path.exists(manifest_file)

    loaded = ExperimentManifest.load_json(manifest_file)
    assert loaded.experiment_id == "test_exp_001"
    assert len(loaded.segments) == 2


def test_segmented_training_fallback_on_llm_failure(tmp_path):
    # LLM will timeout on every request
    provider = MockLLMProvider(simulate_timeout=True)
    gateway = LLMGateway(
        provider, GatewayConfig(max_retries=1, retry_delay_seconds=0.01, cache_enabled=False)
    )
    registry = RewardProfileRegistry()

    config = ExperimentConfig(
        experiment_id="fallback_test_002",
        seed=20,
        total_timesteps=256,
        segment_timesteps=256,
        eval_episodes=1,
        num_steps=128,
    )

    manifest = run_segmented_experiment(
        config=config,
        gateway=gateway,
        registry=registry,
        checkpoint_dir=str(tmp_path / "fallback_ckpt"),
    )

    # Training should succeed without throwing an unhandled exception
    assert len(manifest.segments) == 1
    assert manifest.segments[0].decision == "fallback"
    assert manifest.telemetry["total_fallbacks"] >= 1
