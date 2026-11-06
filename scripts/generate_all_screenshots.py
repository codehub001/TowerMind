#!/usr/bin/env python3
"""
Generate comprehensive high-resolution media screenshots for TowerMind.

Captures:
  1. Web Dashboard Views:
     - Live Visualizer & Interactive Engine
     - Controlled Benchmark Comparison
     - Meta-Controller Review & Decision Timeline
     - Multi-Map Curriculum Generalization
     - Counterfactual Post-Episode Diagnostics
  2. Engine Gameplay Snapshots:
     - Early wave placement
     - Mid wave combat & projectiles
     - Boss wave endgame defense
  3. Curriculum Maps Showcase:
     - 5 distinct map topologies side-by-side
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

from ai_gym_td.engine import TowerDefenseGame
from ai_gym_td.render import PILRenderer
from towermind.curriculum.maps import MAP_FACTORIES
from towermind.strategy.adapter import StrategicAgent


def capture_dashboard_tabs(output_dir: Path) -> None:
    """Capture screenshots of each tab in dashboard.html using headless Firefox."""
    dash_path = Path("benchmark_results/dashboard.html")
    if not dash_path.exists():
        print("Exporting fresh dashboard.html...")
        from towermind.ui.generator import export_dashboard

        dash_path = export_dashboard()

    html_content = dash_path.read_text(encoding="utf-8")
    profile_dir = Path("/tmp/ff_screenshot_profile")
    profile_dir.mkdir(parents=True, exist_ok=True)

    tabs = [
        ("tab-sim", "dashboard_visualizer.png"),
        ("tab-benchmarks", "dashboard_benchmarks.png"),
        ("tab-meta", "dashboard_meta_controller.png"),
        ("tab-generalization", "dashboard_generalization.png"),
        ("tab-counterfactual", "dashboard_counterfactual.png"),
    ]

    for tab_id, filename in tabs:
        dest_path = output_dir / filename
        # Modify active tab in HTML
        mod_html = html_content
        # Remove current active from tabs and contents
        mod_html = re.sub(r'class="tab-btn active"', 'class="tab-btn"', mod_html)
        mod_html = re.sub(r'class="tab-content active"', 'class="tab-content"', mod_html)
        # Add active to target tab
        mod_html = mod_html.replace(
            f"onclick=\"switchTab('{tab_id}')\"", f'class="tab-btn active" onclick="switchTab(\'{tab_id}\')"'
        )
        mod_html = mod_html.replace(
            f'id="{tab_id}" class="tab-content"', f'id="{tab_id}" class="tab-content active"'
        )

        with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as f:
            f.write(mod_html)
            temp_html = f.name

        try:
            cmd = [
                "firefox",
                "--headless",
                "--profile",
                str(profile_dir),
                "--screenshot",
                str(dest_path.resolve()),
                "--window-size",
                "1360,950",
                f"file://{temp_html}",
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            print(f"Captured: {dest_path.name} ({dest_path.stat().st_size // 1024} KB)")
        except Exception as e:
            print(f"Warning: Failed to capture {filename}: {e}")
        finally:
            if os.path.exists(temp_html):
                os.unlink(temp_html)


def generate_gameplay_snapshots(output_dir: Path) -> None:
    """Run an episode with StrategicAgent in TowerDefenseEnv and save high-res gameplay snapshots."""
    from ai_gym_td.env import TowerDefenseEnv

    env = TowerDefenseEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=42)
    agent = StrategicAgent()

    early_saved = False
    mid_saved = False
    boss_saved = False

    for _step_i in range(800):
        act = agent.act(obs, info)
        obs, reward, term, trunc, info = env.step(act)
        frame = env.render()

        wave = info.get("wave", 0)
        towers = len(env.game.towers)
        enemies = len(env.game.enemies)

        # Early game snapshot (Wave 2, first towers placed)
        if not early_saved and wave >= 2 and towers >= 2 and enemies >= 2:
            dest = output_dir / "gameplay_wave_early.png"
            Image.fromarray(frame).save(dest)
            print(f"Saved: {dest.name}")
            early_saved = True

        # Mid game snapshot (Wave 7-9, dense combat, projectiles firing)
        if not mid_saved and wave >= 7 and enemies >= 3 and towers >= 5:
            dest = output_dir / "gameplay_wave_mid.png"
            Image.fromarray(frame).save(dest)
            print(f"Saved: {dest.name}")
            mid_saved = True

        # Boss/late game snapshot (Wave 12+)
        if not boss_saved and wave >= 12 and towers >= 6:
            dest = output_dir / "gameplay_wave_boss.png"
            Image.fromarray(frame).save(dest)
            print(f"Saved: {dest.name}")
            boss_saved = True

        if term or trunc:
            break

    # Fallbacks if wave conditions were missed
    frame = env.render()
    if not early_saved:
        Image.fromarray(frame).save(output_dir / "gameplay_wave_early.png")
    if not mid_saved:
        Image.fromarray(frame).save(output_dir / "gameplay_wave_mid.png")
    if not boss_saved:
        Image.fromarray(frame).save(output_dir / "gameplay_wave_boss.png")
    env.close()


def generate_curriculum_maps_preview(output_dir: Path) -> None:
    """Generate a combined preview collage of all 5 curriculum maps."""
    map_names = ["s_curve", "straight", "zigzag", "chokepoint", "spiral"]

    map_frames = []
    for m in map_names:
        factory = MAP_FACTORIES[m]
        cfg = factory()
        game = TowerDefenseGame(cfg)
        renderer = PILRenderer(cfg, cell_px=22, hud_height=36)
        frame = renderer.render_frame(game)
        im = Image.fromarray(frame)
        # Draw label on top
        draw = ImageDraw.Draw(im)
        draw.text((12, 10), f"Map: {m.upper()}", fill=(240, 240, 240))
        map_frames.append(im)

    # Arrange in a 3 + 2 grid collage
    single_w, single_h = map_frames[0].size
    grid_w = single_w * 3 + 40
    grid_h = single_h * 2 + 50
    collage = Image.new("RGB", (grid_w, grid_h), (25, 25, 25))

    positions = [
        (10, 15),
        (single_w + 20, 15),
        (single_w * 2 + 30, 15),
        (single_w // 2 + 15, single_h + 30),
        (single_w // 2 + single_w + 25, single_h + 30),
    ]

    for im, pos in zip(map_frames, positions):
        collage.paste(im, pos)

    dest = output_dir / "curriculum_maps.png"
    collage.save(dest)
    print(f"Saved: {dest.name} ({dest.stat().st_size // 1024} KB)")


def main():
    media_dir = Path("media")
    media_dir.mkdir(exist_ok=True)

    print("=================================================================")
    print("Capturing TowerMind Media Assets & Project Screenshots")
    print("=================================================================")

    print("\n[1/3] Capturing Web Dashboard tabs...")
    capture_dashboard_tabs(media_dir)

    print("\n[2/3] Simulating StrategicAgent for gameplay snapshots...")
    generate_gameplay_snapshots(media_dir)

    print("\n[3/3] Generating curriculum maps collage...")
    generate_curriculum_maps_preview(media_dir)

    print("\nAll media assets successfully generated in media/ directory!")


if __name__ == "__main__":
    main()
