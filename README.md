# Real-time-Hand-Gesture-Recognition-Model


Link to the  corpus dataset:
https://drive.google.com/drive/folders/153nPh_j3l8gPkcb9Eh2eBHXX-GMgj05W

Link to the  hand signs dataset: https://github.com/hse-sl/rsl-alphabet-dataset


The google drive link to the files:
https://drive.google.com/drive/folders/1dEw9BNNKe1fHDlnzEVujJYhfxLLUzXI2

## Setup environment:
- python -m venv env 
- source ./env/bin/activate 
- python3 -m pip install -r requirements.txt 

## To Run Hand Gesture to letters code
- python3 App.py 

## To Run Hand Gesture to text along with closest recommendation
- python3 App.py 
- Press 'S' to start recording the word. <br />
- press 'S' to stop recording the word. The closest words will be printed on the terminal.

## To Run the next word prediction using LSTM
- Run LSTM_training_new.ipynb
- Run the Inference execution cells
