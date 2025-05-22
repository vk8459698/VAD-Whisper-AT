# For VAD functionality
pip install torch>=1.9.0 torchaudio>=0.9.0
sudo apt update && sudo apt install ffmpeg

# For audio tagging functionality
pip install --upgrade pip
pip install --force-reinstall --no-cache-dir tqdm numba numpy torch more-itertools
curl https://sh.rustup.rs -sSf | sh -s -- -y
source $HOME/.cargo/env
pip install --no-cache-dir tiktoken==0.3.3
pip install --no-deps whisper-at
