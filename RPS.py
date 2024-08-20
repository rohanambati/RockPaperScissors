import tkinter as tk
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np
import random

# Initialize moves
moves = {'rock': 0, 'paper': 1, 'scissors': 2}
reverse_moves = {0: 'rock', 1: 'paper', 2: 'scissors'}

# Initialize dataset
X = []
y = []

# Populate initial dataset with random moves
for _ in range(100):
    player_move = random.randint(0, 2)
    rohan_move = (player_move + 1) % 3  # Simple initial strategy
    X.append([player_move])
    y.append(rohan_move)

X = np.array(X)
y = np.array(y)

# Create a pipeline with standard scaling and logistic regression
model = make_pipeline(StandardScaler(), LogisticRegression())
model.fit(X, y)

# Maintain a history of moves
history = []

def get_rohan_move(player_move_num):
    if len(history) > 1:
        X_history = np.array([move[0] for move in history]).reshape(-1, 1)
        try:
            predicted_move = model.predict([X_history.flatten()])[0]
            return reverse_moves[predicted_move]
        except Exception as e:
            print(f"Prediction error: {e}")
            return reverse_moves[random.randint(0, 2)]
    else:
        return reverse_moves[random.randint(0, 2)]  # Random move if no prediction

def update_model(player_move_num, rohan_move_num):
    global X, y
    X = np.append(X, [[player_move_num]], axis=0)
    y = np.append(y, [rohan_move_num])
    try:
        model.fit(X, y)
    except Exception as e:
        print(f"Model update error: {e}")

def update_history(player_move_num, rohan_move_num):
    global history
    history.append((player_move_num, rohan_move_num))
    if len(history) > 10:
        history.pop(0)

class RockPaperScissorsGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Rock Paper Scissors")

        self.label = tk.Label(root, text="Choose your move:", font=('Helvetica', 16), fg='blue')
        self.label.pack()

        self.rock_button = tk.Button(root, text="Rock", width=20, command=lambda: self.play('rock'), bg='lightgray')
        self.rock_button.pack()

        self.paper_button = tk.Button(root, text="Paper", width=20, command=lambda: self.play('paper'), bg='lightgray')
        self.paper_button.pack()

        self.scissors_button = tk.Button(root, text="Scissors", width=20, command=lambda: self.play('scissors'), bg='lightgray')
        self.scissors_button.pack()

        self.result_label = tk.Label(root, text="", font=('Helvetica', 16), fg='green')
        self.result_label.pack()

        self.end_button = tk.Button(root, text="End Game", width=20, command=self.end_game, bg='red', fg='white')
        self.end_button.pack()

    def play(self, player_move):
        player_move_num = moves[player_move]
        rohan_move = get_rohan_move(player_move_num)
        rohan_move_num = moves[rohan_move]

        # Update the model and history with the latest moves
        update_model(player_move_num, rohan_move_num)
        update_history(player_move_num, rohan_move_num)

        # Determine the winner
        if player_move == rohan_move:
            result = "It's a tie!"
        elif (player_move == 'rock' and rohan_move == 'scissors') or \
             (player_move == 'paper' and rohan_move == 'rock') or \
             (player_move == 'scissors' and rohan_move == 'paper'):
            result = "You win!"
        else:
            result = "Rohan wins!"

        self.result_label.config(text=f"You chose: {player_move}\nRohan chose: {rohan_move}\n{result}")

    def end_game(self):
        self.root.destroy()

# Main loop
if __name__ == "__main__":
    root = tk.Tk()
    game = RockPaperScissorsGame(root)
    root.mainloop()

