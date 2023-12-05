# Installation (MACOS)

```
python3.10 -m venv venv-metal
source venv-metal/bin/activate
python3.10 -m pip install -U pip
python3.10 -m pip install tensorflow==2.14
python3.10 -m pip install tensorflow-metal==1.1.0
python3.10 -m pip install --upgrade tensorflow_hub
python3.10 -m pip install -r requirements.txt
```

# Running Pose Correction

```
python3.10 alignment_feedback.py
```
