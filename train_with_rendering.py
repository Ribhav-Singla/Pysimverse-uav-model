#!/usr/bin/env python3
"""
Test script for UAV RL training with rendering
Demonstrates the agent learning to navigate with visual feedback
"""

from train_uav_rl import UAVTrainer

def main():
    """Run a short training session with rendering enabled"""
    print("🚀 Starting UAV RL Training with Rendering Demo...")
    print("📺 This will show the agent learning to navigate in real-time!")
    print("🎬 Episodes will be rendered when they start...")
    
    # Configuration for demo with rendering
    demo_config = {
        'total_episodes': 10,  # Very short demo
        'max_episode_steps': 100,  # Limit episode length
        'update_frequency': 50,  # Frequent updates
        'save_frequency': 5,
        'eval_frequency': 5,
        'learning_rate': 1e-3,  # Faster learning for demo
        'gamma': 0.95,
        'clip_epsilon': 0.3,
        'k_epochs': 5,
        'entropy_coef': 0.02,
        'value_coef': 0.5,
        'device': 'cpu',
        'random_goal': False,
        'render': True,  # Enable rendering
        'render_episodes': [0, 1, 2, 4, 6, 8, 9],  # Render most episodes
        'render_frequency': 1  # Render every episode
    }
    
    print(f"\n🎯 Demo Configuration:")
    print(f"  Episodes: {demo_config['total_episodes']}")
    print(f"  Max steps per episode: {demo_config['max_episode_steps']}")
    print(f"  Rendering enabled: {demo_config['render']}")
    print(f"  Episodes to render: {demo_config['render_episodes']}")
    
    print(f"\n🎮 What you'll see:")
    print(f"  - Red UAV quadcopter learning to navigate")
    print(f"  - Green start marker and blue goal marker")
    print(f"  - Colored obstacles to avoid")
    print(f"  - Agent behavior improving over episodes")
    
    # Create trainer and run demo
    trainer = UAVTrainer(config=demo_config)
    trainer.train()
    
    print(f"\n🎉 Demo completed!")
    print(f"📊 Final performance:")
    
    # Quick evaluation
    eval_results = trainer.evaluate_agent(num_episodes=3)
    print(f"  Success rate: {eval_results['success_rate']:.1%}")
    print(f"  Average reward: {eval_results['avg_reward']:.2f}")

if __name__ == "__main__":
    main()