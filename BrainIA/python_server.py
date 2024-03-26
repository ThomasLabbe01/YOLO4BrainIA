import asyncio
import websockets
import websockets.exceptions
import json
import os

from data_config import *
from Predictor import Predictor
from Quantifier import Quantifier


class Server:
    def __init__(self):
        self.fileName = "CustomModelSmall.pt"
        self.script_directory = os.path.dirname(os.path.abspath(__file__))
        self.repo_directory = os.path.dirname(self.script_directory)
        self.weights_path = os.path.join(self.repo_directory, "weights", self.fileName)
        self.userWorkspacePath = os.path.expandvars("%USERPROFILE%\\Documents\\Brain_Projects")

        self.PREDICTOR = Predictor(self.weights_path)
        self.QUANTIFIER = Quantifier()

    async def handler(self, websocket, path):
        print("\n")
        print("-" * 60)
        print("Client is connected to Server")
        while True:
            try:
                message = await websocket.recv()
                message_json = json.loads(message)
                task = message_json["task"]
                folder_path = message_json["path"]
                args = message_json["args"]

                print("TASK IS : ", task)
                print("FILE AT : ", folder_path)
                print("ARGS ARE: ", args)

                if task == "splitFile":
                    extension, base_name, frame_name, directory = split_path(folder_path)
                    self.PREDICTOR.extract_frames(base_name, directory)
                    final_dict = {"working": True}
                    await websocket.send(json.dumps(final_dict))
                    await websocket.close()

                if task == "plotOccurenceNumber" or task == "plotSpreadness" or task == "plotOccurenceOverTime":
                    self.QUANTIFIER.prepare_object_list(folder_path)
                    self.QUANTIFIER.quantify(folder_path)
                    self.QUANTIFIER.plotGraphs(task)

                # TODO: call Train_button.py function and see if answer is necessary
                if task == "trainDataset":
                    pass

            except websockets.exceptions.ConnectionClosedError as e:
                print("Client closed connection")
                break

            except websockets.exceptions.ConnectionClosedOK:
                print("Caught ConnectionCloseOK")
                break


    async def runserver(self):
        print("Waiting for client's connection...")
        try:
            server = await websockets.serve(self.handler, "localhost", 10000)
            await server.wait_closed()
        except Exception as e:
            print(f"Error starting server: {e}")
            pass
        finally:
            pass


if __name__ == "__main__":
    SERVER = Server()
    asyncio.run(SERVER.runserver())
