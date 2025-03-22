===Mario AI Agent with PyTorch===
- A self-learning AI that masters Super Mario Bros using Deep Q-Learning (DQN).

===Objective===
- Train an AI agent to play Super Mario Bros autonomously using PyTorch and GPU acceleration (NVIDIA RTX 2060+).

===Key Features===
- 🎮 Real-time gameplay visualization during training
- 🧠 Deep Q-Learning with experience replay and target networks
- ⚡ GPU-accelerated training (CUDA support)
- 📊 Live performance tracking with reward graphs
- 💾 Auto-save system for progress protection

===Technical Components===
- Neural Network: 3-layer CNN + 2-layer FC
- Input: 84x84 grayscale frames (4-frame stack)
- Output: 7 possible Mario actions
- Training: 100,000+ episodes with ε-greedy exploration

=== INSTALLATION ===
1. Install Python 3.8+
2. Run these commands:
- "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117"
- "pip install gym==0.25.2 gym-super-mario-bros==7.3.0 numpy matplotlib opencv-python nes-py"

=== HOW TO RUN ===
1. Save code as mario_ai.py
2. Open terminal
3. Run: python mario_ai.py
4. Wait 10-15 minutes for initial learning
5. Watch Mario improve over hours

=== TROUBLESHOOTING ===
Common Issues:
- Black screen: Update graphics drivers
- CUDA errors: Reduce batch_size to 32
- Gym errors: Reinstall with pip install gym==0.25.2
- Memory issues: Close other applications

=== HARDWARE REQUIREMENTS ===
- NVIDIA GPU (RTX 2060+ recommended)
- 8GB+ RAM
- 2GB+ free disk space

=== TRAINING TIPS ===
1. Let run overnight for best results
2. Monitor average reward curve
3. Stop/restart anytime (model auto-saves)
4. For faster training: Close other programs
