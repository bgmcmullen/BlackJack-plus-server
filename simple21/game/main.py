"""
author - Brendan McMullen
"""



# import the random module
# use "random_int = randint(1, 13)" to generate a random int from 1 - 13 and store in a variable "random_int"
from random import randint

from django.http import HttpResponse
from django.http import JsonResponse
from django.views.decorators.csrf import ensure_csrf_cookie

import json

import torch
import random
import numpy as np
import torch.nn as nn

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




@ensure_csrf_cookie
def hello_world(request):

    return HttpResponse("Hello World!")


class Game:

    # Global Variables

    def __init__(self):

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


        # Test the model on specific examples
    def AI_take_another_card(self):


        self.computer_total_points = self.calculate_score(self.computer_visible_card_total_values + self.computer_hidden_card_value, True)
        self.user_visible_points = self.calculate_score(self.user_visible_card_total_values, True)

        state = self.computer_total_points + self.user_visible_points + [self.target_score]

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        # Get the Q-values predicted by the model
        with torch.no_grad():
            q_values = model(state_tensor).squeeze().numpy()

        # Choose the best action based on Q-values
        best_action = np.argmax(q_values)


        surety = float(abs(q_values[0] - q_values[1]))

        if best_action == 1:
            return [True, surety]
        if best_action == 0:
            return [False, surety]


    def print_instructions(self):
        """
        Prints out instructions for the game.
        """
        intructions = "Hello and welcome to Simple21!\r\nThe object of the game it to get as close to 21 as you can, but DON'T go over!"
        return intructions

    def calculate_score(self, stack, includeAs=False):
        score = 0
        num_of_As = 0

        print(stack)
        for card in stack:
            if isinstance(card['card_value'], int):
                score += card['card_value']
            elif card['card_value'] == 'A':
                score += 11
                num_of_As += 1
            else:
                score += 10
            
        if score > self.target_score:
            for i in range(0, num_of_As):
                score -= 10
                if score <= 21:
                    break

        if includeAs == True:
            return [score, num_of_As]
        else:
            return score

    def ask_yes_or_no(prompt):
        """
        Displays the given prompt and asks the user for input.  If the user's input starts with 'y', returns True.
        If the user's input starts with 'n', returns False.
        For example, calling ask_yes_or_no("Do you want to play again? (y/n)")
        would display "Do you want to play again? (y/n)", wait for user input that starts with 'y' or 'n',
        and return True or False accordingly.
        """

        #cast input to a list
        y_or_n = list(input(prompt))

        #determine if list contains Y or y and return true or false accordingly
        for i in range(0, len(y_or_n)):
            if(y_or_n[i] == "y" or y_or_n[i] == "Y"):
                return True
            else:
                continue

    def next_card(self, surety=None):
        """
        Returns a random "card", represented by an int between 1 and 10, inclusive.
        The "cards" are the numbers 1 through 10 and they are randomly generated, not drawn from a deck of
        limited size.  The odds of returning a 10 are four times as likely as any other value (because in an
        actual deck of cards, 10, Jack, Queen, and King all count as 10).
        """


        if(len(self.card_deck) == 0):
            return "None"

        random_int = randint(0, len(self.card_deck) - 1)
        
        card = self.card_deck[random_int]

        del self.card_deck[random_int]

        card["surety"] = surety

        return card


    def take_another_card(self):
        """
        Strategy for computer to take another card or not.  According to the computer’s own given
        total points (sum of visible cards + hidden card) and the user's sum of visible cards, you
        need to design a game strategy for the computer to win the game.
        Returns True if the strategy decides to take another card, False if the computer decides not
        to take another card.
        """

        #The computer will take a new card is 15 of less or if it has 18 or less and
        #the computers total point(s) are less and four more the the users visible point(s)

        print('computer_total_card', self.computer_visible_card_total_values + self.computer_hidden_card_value)

        self.computer_total_points = self.calculate_score(self.computer_visible_card_total_values + self.computer_hidden_card_value)
        self.user_visible_points = self.calculate_score(self.user_visible_card_total_values)

        
        if(self.computer_total_points < 15 or (( self.computer_total_points < (self.user_visible_points + 4)) and self.computer_total_points < 18)):
            return True
        else:
            return False
        

    def is_game_over(is_user_passed, is_computer_passed):
        """
        Determines if the game is over or not.
        If the given is_user_passed is set to True, the user has passed.
        If the given is_computer_passed is set to True, the computer has passed.
        This function returns True if both the user and the computer have passed,
        and False if either of them has not yet passed.
        """
        if(is_user_passed == True and is_computer_passed == True):
            return True
        else:
            return False

    # def print_status(is_user, name, hidden_card, visible_card, total_points):
    #     """
    #     In each turn, prints out the current status of the game.
    #     If the given player (name) is the user, is_user will be set to True.  In this case, print out
    #     the user's given name, his/her hidden card points, visible card points, and total points.
    #     If the given player (name) is the computer, is_user will be set to False.  In this case, print out
    #     the computer's given name, and his/her visible card points.
    #     """

    #     text = ''
    #     #print all users points
    #     if(is_user == True):    
    #         text = (
    #             f"{name} has:\r\n   {hidden_card} hidden point(s),\r\n  {visible_card} "
    #             f"visible point(s),\r\n   {total_points} total point(s)"
    #         )

    #     #print computers visible point(s)
    #     if(is_user == False):
    #         text = f"{name} has:\r\n   {visible_card}  visible point(s)"

    #     return text


    def set_target_score(self):
        self.target_score = randint(21,27)
        return self.target_score
        

    def print_winner(self, username, user_total_points, computer_name, computer_total_points):
        """
        Determines who won the game and prints the game results in the following format:
        - User's given name and the given user's total points
        - Computer's given name and the given computer's total points
        - The player who won the game and the total number of points he/she won by, or if it's a tie, nobody won.
        """

        user_score = self.calculate_score(user_total_points)
        computer_score = self.calculate_score(computer_total_points)


        text = [f"{username}, has {user_score} and {computer_name} has {computer_score}"]

        winner = ''

            #If the user has more points(not over target score)
        if(user_score  > computer_score and user_score  <= self.target_score ):
            text.append(f"{username} won by {int(user_score - computer_score)}")
            winner = 'user'

            #If the computer has more points(not over target score)
        elif(computer_score > user_score  and computer_score <= self.target_score ):
            text.append(f"{computer_name} won by {int(computer_score - user_score)}")
            winner = 'computer'

            #If the computer overshot target score)
        elif(computer_score > self.target_score  and user_score <= self.target_score ):
            text.append(f"{username} won, {computer_name} went bust")
            winner = 'user'

            #If the user overshot target score)
        elif(user_score  > self.target_score  and computer_score <= self.target_score ):
            text.append(f"{computer_name} won {username} went bust")
            winner = 'computer'

            #If the computer and user have the same number of point or both overshot 21
        else:
            text.append("It's a tie")
            winner = 'tie'
        return { 'winner_text': text, 'winner': winner }

    def run(self):
        """
        This function controls the overall game and logic for the given user and computer.
        """

        #Over or not will be set to true when game is over
        global over_or_not
        over_or_not = False

        # global user_visible_card_total_values
        # global user_hidden_card_value
        # global computer_visible_card_total_values
        # global computer_hidden_card_value
        # global is_computer_passed
        # global cards
        # global card_deck

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

        #determine and print starting point values for user
        self.user_hidden_card_value = [self.next_card()]
        self.user_visible_card_total_values = [self.next_card()]

        self.cards['user_hidden_card_value'] = self.user_hidden_card_value
        self.cards['user_visible_card_total_values'] = self.user_visible_card_total_values


        # text.append(print_status(True, username, user_hidden_card_value, user_visible_card_total_values,
        #              int(user_hidden_card_value + user_visible_card_total_values)))

        #determine and print starting visible points for computer
        self.computer_hidden_card_value = [self.next_card()]
        self.computer_visible_card_total_values = [self.next_card()]

        self.cards['computer_hidden_card_value'] = self.computer_hidden_card_value
        self.cards['computer_visible_card_total_values'] = self.computer_visible_card_total_values

        # text.append(print_status(False, computer_name, computer_hidden_card_value, computer_visible_card_total_values,
        #              int(computer_hidden_card_value + computer_visible_card_total_values)))

        return self.cards


        #is_computer_passed will be set true when the computer declines a new card

        #This while loop will continue untill the game is over



    def computer_turn(self):

        # global computer_visible_card_total_values
        # global computer_hidden_card_value
        # global user_visible_card_total_values
        # global is_computer_passed

        [computer_takes_card, surety] = self.AI_take_another_card()


        # computer_takes_card = self.take_another_card()

        #if the computer takes a new card
        if(computer_takes_card == True):
            next_card_value = self.next_card(surety)
            # computer_visible_card_total_values.append(next_card_value)
            # text.append(f"{computer_name} gets {next_card_value}")
            if next_card_value != None:
                self.cards['computer_visible_card_total_values'].append(next_card_value)

            self.is_computer_passed = False
            # text.append(print_status(False, computer_name, computer_hidden_card_value, computer_visible_card_total_values,
            #                 [*computer_hidden_card_value, *computer_visible_card_total_values]))

        else:
            self.is_computer_passed = True

        # if the computer passes


    def play_turn(self):


        # global user_visible_card_total_values
        # global user_hidden_card_value
        # global is_computer_passed
        # global cards
    

        next_card_value = self.next_card()
        # user_visible_card_total_values.append(next_card_value)

        
        # text.append(f"{username} gets {next_card_value}")
        if next_card_value != None:
            self.cards['user_visible_card_total_values'].append(next_card_value)

        # text.append(print_status(True, username, user_hidden_card_value, user_visible_card_total_values,
        #                 [*user_hidden_card_value, *user_visible_card_total_values]))

        #if the computer has not yet passed determine if is takes a new card
        if(self.is_computer_passed == False):
            self.computer_turn()
            
                # text.append(f"{computer_name} passed")

        return self.cards


    def player_passes(self):

        # global is_computer_passed

        #determine if the game is over
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
