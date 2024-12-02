"""
author - Brendan McMullen
"""


# import the random module
# use "random_int = randint(1, 13)" to generate a random int from 1 - 13 and store in a variable "random_int"
from random import randint

from django.http import HttpResponse
from django.views.decorators.csrf import ensure_csrf_cookie

import torch
import numpy as np
import torch.nn as nn

# Set up AI model
class Simple21Net(nn.Module):
    def __init__(self, input_size, action_size):
        super(Simple21Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the trained model
state_size = 5
action_size = 2  
model = Simple21Net(input_size=state_size, action_size=action_size)
model.load_state_dict(torch.load("Simple21model_with_opponent_with_target.pth"))
model.eval() 


# Test HttpResponse for testing and monitoring
@ensure_csrf_cookie
def hello_world(request):
    return HttpResponse("Hello World!")


class Game:
    
    def __init__(self):

        # Global Variables
        self.username = ''
        self.computer_name = "Computer"
        self.card_deck = []
        self.is_computer_passed = False
        self.cards = {}
        self.user_visible_card_total_values = []
        self.user_hidden_card_value = []
        self.computer_visible_card_total_values = []
        self.computer_hidden_card_value = []
        self.target_score = 0


    # Use AI model to determine if computer takes another card
    def AI_take_another_card(self):

        # Get computer's score
        self.computer_total_points = self.calculate_score(self.computer_visible_card_total_values + self.computer_hidden_card_value, True)

        # Get user's score
        self.user_visible_points = self.calculate_score(self.user_visible_card_total_values, True)

        # Compile game state for model
        state = self.computer_total_points + self.user_visible_points + [self.target_score]

        # Convert state to PyTorch tensor
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        # Get the Q-values predicted by the model
        with torch.no_grad():
            q_values = model(state_tensor).squeeze().numpy()

        # Choose the best action based on Q-values
        best_action = np.argmax(q_values)

        # Calculate surety based on difference between Q-values
        surety = float(abs(q_values[0] - q_values[1]))

        # Computer does take card
        if best_action == 1:
            return [True, surety]
        
        # Computer does not take card
        if best_action == 0:
            return [False, surety]

    def calculate_score(self, stack, includeAs=False):
        score = 0
        num_of_As = 0

        for card in stack:

            # Add values of numbered cards
            if isinstance(card['card_value'], int):
                score += card['card_value']

            # Track number of aces and add 11 for each
            elif card['card_value'] == 'A':
                score += 11
                num_of_As += 1

            # Add 10 for each other lettered card
            else:
                score += 10
            
        # Drop ace values to 10 if score is over target
        if score > self.target_score:
            for i in range(0, num_of_As):
                score -= 10
                if score <= self.target_score:
                    break

        # return score and number of aces if aces are included
        if includeAs == True:
            return [score, num_of_As]
        else:
            return score

    def next_card(self, surety=None):
        """
        Draw a new card for the deck
        """

        # Return None if deck is empty
        if(len(self.card_deck) == 0):
            return "None"

        # Choose a random index from the deck
        random_int = randint(0, len(self.card_deck) - 1)


        # Take new card from deck
        card = self.card_deck[random_int]


        # Remove new card from deck
        del self.card_deck[random_int]


        # Assign surety to card if it was provided
        card["surety"] = surety

        return card

    def is_game_over(is_user_passed, is_computer_passed):
        """
        Determines if the game is over or not.
        """
        if(is_user_passed == True and is_computer_passed == True):
            return True
        else:
            return False

    # Set the target score to a random values from 21 to 27
    def set_target_score(self):
        self.target_score = randint(21,27)
        return self.target_score
        

    def print_winner(self, username, user_total_points, computer_name, computer_total_points):
        """
        Determines who won the game and prints the game results
        """

        # Get user's score
        user_score = self.calculate_score(user_total_points)

        # Get computer's score
        computer_score = self.calculate_score(computer_total_points)

        # Set results text
        text = [f"{username}, has {user_score} and {computer_name} has {computer_score}"]

        winner = ''

        # If the user has more points(not over target score)
        if(user_score  > computer_score and user_score  <= self.target_score ):
            text.append(f"{username} won by {int(user_score - computer_score)}")
            winner = 'user'

        # If the computer has more points(not over target score)
        elif(computer_score > user_score  and computer_score <= self.target_score ):
            text.append(f"{computer_name} won by {int(computer_score - user_score)}")
            winner = 'computer'

        # If the computer overshot target score)
        elif(computer_score > self.target_score  and user_score <= self.target_score ):
            text.append(f"{username} won, {computer_name} went bust")
            winner = 'user'

        # If the user overshot target score)
        elif(user_score  > self.target_score  and computer_score <= self.target_score ):
            text.append(f"{computer_name} won {username} went bust")
            winner = 'computer'

        # If the computer and user have the same number of point or both overshot 21
        else:
            text.append("It's a tie")
            winner = 'tie'
        return { 'winner_text': text, 'winner': winner }

    def run(self):
        """
        This function controls the overall game and logic for the given user and computer.
        """

        # Over or not will be set to true when game is over
        global over_or_not
        over_or_not = False


        self.card_deck = [
        {'card_value': 2, 'card_suite': 'hearts'}, {'card_value': 2, 'card_suite': 'diamonds'}, {'card_value': 2, 'card_suite': 'clubs'}, {'card_value': 2, 'card_suite': 'spades'},
        {'card_value': 3, 'card_suite': 'hearts'}, {'card_value': 3, 'card_suite': 'diamonds'}, {'card_value': 3, 'card_suite': 'clubs'}, {'card_value': 3, 'card_suite': 'spades'},
        {'card_value': 4, 'card_suite': 'hearts'}, {'card_value': 4, 'card_suite': 'diamonds'}, {'card_value': 4, 'card_suite': 'clubs'}, {'card_value': 4, 'card_suite': 'spades'},
        {'card_value': 5, 'card_suite': 'hearts'}, {'card_value': 5, 'card_suite': 'diamonds'}, {'card_value': 5, 'card_suite': 'clubs'}, {'card_value': 5, 'card_suite': 'spades'},
        {'card_value': 6, 'card_suite': 'hearts'}, {'card_value': 6, 'card_suite': 'diamonds'}, {'card_value': 6, 'card_suite': 'clubs'}, {'card_value': 6, 'card_suite': 'spades'},
        {'card_value': 7, 'card_suite': 'hearts'}, {'card_value': 7, 'card_suite': 'diamonds'}, {'card_value': 7, 'card_suite': 'clubs'}, {'card_value': 7, 'card_suite': 'spades'},
        {'card_value': 8, 'card_suite': 'hearts'}, {'card_value': 8, 'card_suite': 'diamonds'}, {'card_value': 8, 'card_suite': 'clubs'}, {'card_value': 8, 'card_suite': 'spades'},
        {'card_value': 9, 'card_suite': 'hearts'}, {'card_value': 9, 'card_suite': 'diamonds'}, {'card_value': 9, 'card_suite': 'clubs'}, {'card_value': 9, 'card_suite': 'spades'},
        {'card_value': 10, 'card_suite': 'hearts'}, {'card_value': 10, 'card_suite': 'diamonds'}, {'card_value': 10, 'card_suite': 'clubs'}, {'card_value': 10, 'card_suite': 'spades'},
        {'card_value': 'J', 'card_suite': 'hearts'}, {'card_value': 'J', 'card_suite': 'diamonds'}, {'card_value': 'J', 'card_suite': 'clubs'}, {'card_value': 'J', 'card_suite': 'spades'},
        {'card_value': 'Q', 'card_suite': 'hearts'}, {'card_value': 'Q', 'card_suite': 'diamonds'}, {'card_value': 'Q', 'card_suite': 'clubs'}, {'card_value': 'Q', 'card_suite': 'spades'},
        {'card_value': 'K', 'card_suite': 'hearts'}, {'card_value': 'K', 'card_suite': 'diamonds'}, {'card_value': 'K', 'card_suite': 'clubs'}, {'card_value': 'K', 'card_suite': 'spades'},
        {'card_value': 'A', 'card_suite': 'hearts'}, {'card_value': 'A', 'card_suite': 'diamonds'}, {'card_value': 'A', 'card_suite': 'clubs'}, {'card_value': 'A', 'card_suite': 'spades'}
    ]

        self.is_computer_passed = False
        self.user_visible_card_total_values = []
        self.user_hidden_card_value = []
        self.computer_visible_card_total_values = []
        self.computer_hidden_card_value = []

        # Set up user's cards
        self.user_hidden_card_value = [self.next_card()]
        self.user_visible_card_total_values = [self.next_card()]

        self.cards['user_hidden_card_value'] = self.user_hidden_card_value
        self.cards['user_visible_card_total_values'] = self.user_visible_card_total_values


        # Set up computer's cards
        self.computer_hidden_card_value = [self.next_card()]
        self.computer_visible_card_total_values = [self.next_card()]

        self.cards['computer_hidden_card_value'] = self.computer_hidden_card_value
        self.cards['computer_visible_card_total_values'] = self.computer_visible_card_total_values

        return self.cards

    def computer_turn(self):

        # Determine is computer takes another card
        [computer_takes_card, surety] = self.AI_take_another_card()

        # If the computer takes a new card
        if(computer_takes_card == True):

            # Get next card
            next_card_value = self.next_card(surety)

            if next_card_value != None:
                self.cards['computer_visible_card_total_values'].append(next_card_value)

            self.is_computer_passed = False

        # If the computer does not take another card
        else:
            self.is_computer_passed = True

    def play_turn(self):

        next_card_value = self.next_card()

        if next_card_value != None:
            self.cards['user_visible_card_total_values'].append(next_card_value)


        # If the computer has not yet passed determine if is takes a new card
        if(self.is_computer_passed == False):
            self.computer_turn()

        return self.cards


    def player_passes(self):

        # Determine if the game is over
        while  self.is_computer_passed == False:
            self.computer_turn()

        winner_dict= self.print_winner(username, self.user_hidden_card_value + self.user_visible_card_total_values, self.computer_name,
            self.computer_visible_card_total_values + self.computer_hidden_card_value)
        
        response = {'cards': self.cards, 'winner_dict': winner_dict}

        return response

    def set_user_name(self, name):
        global username
        username = name
        response = "Welcome " + username + "!"
        return response
