import os
from time import sleep

if __name__ == '__main__':
    while True:
        try:
            os.system("python3 Predict_LSTM-1min.py")
            sleep(60)
        except:
            print('An error occurred.')
