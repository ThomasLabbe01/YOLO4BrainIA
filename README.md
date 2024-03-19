# YOLO4BrainIA

!!DONT FORGET THE weights FOLDER REQUIRED IN THE MAIN DIRECTORY!!

STEPS:

1. Install Anaconda https://www.anaconda.com/download
2. Open anaconda prompt

    
    press Windows keys (search)
    write anaconda3
    press Enter

4. Enter:

   
    create --name ENVS_NAME
    conda install python


5. Copy - Paste the following list


    python -m pip install fiftyone
    python -m pip install opencv-python
    python -m pip install torch==2.2.1+cu121 torchvision==0.17.1+cu121 --index-url https://download.pytorch.org/whl/cu121
    python -m pip install ultralytics
    python -m pip install websockets

6. Run BrainIA/python_server.py
