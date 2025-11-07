conda create --name vings_vio python=3.9.19
pip3 install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip3 install torch-scatter==2.0.2 -f https://data.pyg.org/whl/torch-2.0.2+cu118.html
pip3 install -r requirements.txt

# Build dbaf.
cd submodules/dbaf
sudo python3 setup.py install
