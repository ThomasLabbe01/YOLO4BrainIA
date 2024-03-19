# YOLO4BrainIA

!!! DONT FORGET THE weights FOLDER REQUIRED IN THE MAIN DIRECTORY !!!

STEPS:
1. Create a directory in YOLO4BrainIA called weights
2. Add the weights in that folder (will not be push to git)
   1. Edit weights name in python_server.py line 13
3. Install Anaconda https://www.anaconda.com/download
4. Open anaconda prompt:


      Press Windows keys (search)
      Write anaconda3
      Press Enter



6. Enter:

   
    create --name ENVS_NAME
    conda activate ENVS_NAME
    conda install python


5. Copy - Paste the following list:


    python -m pip install fiftyone
    python -m pip install opencv-python
    python -m pip install torch==2.2.1+cu121 torchvision==0.17.1+cu121 --index-url https://download.pytorch.org/whl/cu121
    python -m pip install ultralytics
    python -m pip install websockets

6. Verify that your IDE is using the proper interpreter (ENVS)
7. Run BrainIA/python_server.py
