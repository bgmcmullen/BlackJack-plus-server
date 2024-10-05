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


# Global Variables
username = ''
computer_name = "Computer"


@ensure_csrf_cookie
def hello_world(request):
    body = json.loads(request.body)
    if request.method == 'POST':
        return JsonResponse({'method': 'post', 'body': body})
    elif request.method == 'GET':
        return JsonResponse({'method': 'get'})
    else:
        return JsonResponse({'method': request.method})

def print_instructions():
    """
    Prints out instructions for the game.
    """
    intructions = "Hello and welcome to Simple21!\r\nThe object of the game it to get as close to 21 as you can, but DON'T go over!"
    return intructions

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

def next_card():
    """
    Returns a random "card", represented by an int between 1 and 10, inclusive.
    The "cards" are the numbers 1 through 10 and they are randomly generated, not drawn from a deck of
    limited size.  The odds of returning a 10 are four times as likely as any other value (because in an
    actual deck of cards, 10, Jack, Queen, and King all count as 10).
    """

    #generate random number from 1 to 13 and reduce all numbers about 10 to 10
    random_int = randint(1, 13)
    if (random_int > 10):
        random_int = 10

    #return the int btween 1 and 10
    return random_int


def take_another_card(computer_total_points, user_visible_card):
    """
    Strategy for computer to take another card or not.  According to the computer’s own given
    total points (sum of visible cards + hidden card) and the user's sum of visible cards, you
    need to design a game strategy for the computer to win the game.
    Returns True if the strategy decides to take another card, False if the computer decides not
    to take another card.
    """

    #The computer will take a new card is 15 of less or if it has 18 or less and
    #the computers total point(s) are less and four more the the users visible point(s)
    if(computer_total_points < 15 or ((computer_total_points < (user_visible_card + 4)) and computer_total_points < 18)):
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

def print_status(is_user, name, hidden_card, visible_card, total_points):
    """
    In each turn, prints out the current status of the game.
    If the given player (name) is the user, is_user will be set to True.  In this case, print out
    the user's given name, his/her hidden card points, visible card points, and total points.
    If the given player (name) is the computer, is_user will be set to False.  In this case, print out
    the computer's given name, and his/her visible card points.
    """

    text = ''
    #print all users points
    if(is_user == True):    
        text = (
            f"{name} has:\r\n   {hidden_card} hidden point(s),\r\n  {visible_card} "
            f"visible point(s),\r\n   {total_points} total point(s)"
        )

    #print computers visible point(s)
    if(is_user == False):
        text = f"{name} has:\r\n   {visible_card}  visible point(s)"

    return text
    

def print_winner(username, user_total_points, computer_name, computer_total_points):
    """
    Determines who won the game and prints the game results in the following format:
    - User's given name and the given user's total points
    - Computer's given name and the given computer's total points
    - The player who won the game and the total number of points he/she won by, or if it's a tie, nobody won.
    """


    print(username, "has",user_total_points, "and", computer_name, "has", computer_total_points)

        #If the user has more points(not over 21)
    if(user_total_points > computer_total_points and user_total_points <= 21):
        print(username, "won by", int(user_total_points - computer_total_points))

        #If the computer has more points(not over 21)
    elif(computer_total_points > user_total_points and computer_total_points <= 21):
        print(computer_name, "won by", int(computer_total_points - user_total_points))

        #If the computer overshot 21)
    elif(computer_total_points > 21 and user_total_points <= 21):
        print(username, "won by", int(computer_total_points - user_total_points))

        #If the user overshot 21)
    elif(user_total_points > 21 and computer_total_points <= 21):
        print(computer_name, "won by ", int(user_total_points - computer_total_points))

        #If the computer and user have the same number of point or both overshot 21
    else:
        print("It's a tie")

#Over or not will be set to true when game is over
    over_or_not = False

def run():
    """
    This function controls the overall game and logic for the given user and computer.
    """

    #Over or not will be set to true when game is over
    over_or_not = False

    text = ''

    #determine and print starting point values for user
    user_hidden_card_value = next_card()
    user_visible_card_total_values = next_card()
    text += print_status(True, username, user_hidden_card_value, user_visible_card_total_values,
                 int(user_hidden_card_value + user_visible_card_total_values))

    #determine and print starting visible points for computer
    computer_hidden_card_value = next_card()
    computer_visible_card_total_values = next_card()
    text += print_status(False, computer_name, computer_hidden_card_value, computer_visible_card_total_values,
                 int(computer_hidden_card_value + computer_visible_card_total_values))

    return text

    #is_user_passed will be set true when the user declines a new card
    is_user_passed = False

    #is_computer_passed will be set true when the computer declines a new card
    is_computer_passed = False

    #This while loop will continue untill the game is over
    # while(over_or_not == False):

    #     #if the user has not yet passed ask the user if they want to take a new card
    #     if(is_user_passed == False):
    #         is_taking_new_card = ask_yes_or_no("Take another card? (y/n)")

    #         #if the user take a new card determine the value, add it to the total and print the status
    #         if(is_taking_new_card == True):
    #             next_card_value = next_card()
    #             user_visible_card_total_values += next_card_value
    #             print(username, "get", next_card_value)
    #             is_user_passed = False
    #             print_status(True, username, user_hidden_card_value, user_visible_card_total_values,
    #                          int(user_hidden_card_value + user_visible_card_total_values))

    #         # if the user passes
    #         else:
    #             is_user_passed = True
    #             print(username, "passed")

    #     #if the computer has not yet passed determine if is takes a new card
    #     if(is_computer_passed == False):
    #         computer_takes_card = take_another_card(int(computer_visible_card_total_values +
    #                     computer_hidden_card_value), user_visible_card_total_values)

    #         #if the computer takes a new card
    #         if(computer_takes_card == True):
    #             next_card_value = next_card()
    #             computer_visible_card_total_values += next_card_value
    #             print(computer_name, "gets", next_card_value)
    #             is_computer_passed = False
    #             print_status(False, computer_name, computer_hidden_card_value, computer_visible_card_total_values,
    #                          int(computer_hidden_card_value + computer_visible_card_total_values))

    #         # if the computer passes
    #         else:
    #             is_computer_passed = True
    #             print(computer_name, "passed")

    #     #determine if the game is over
    #     over_or_not = is_game_over(is_user_passed, is_computer_passed)

    # #print game over message and determine winner
    # print("----The Game is Over!----")
    # print_winner(username, int(user_hidden_card_value + user_visible_card_total_values), computer_name,
    #              int(computer_visible_card_total_values + computer_hidden_card_value))

    # #ask if the user wants to play again
    # wants_to_play_again = ask_yes_or_no("Play again? (y/n)")
    # if(wants_to_play_again == True):

    #     #If the player wants to play again ask if they want to change the user name
    #     changes_name = ask_yes_or_no("Change name? (y/n)")
    #     if(changes_name == True):
    #         username = input("New name?\r\n")

    #     #run the game again!
    #     run(username, computer_name)


def set_user_name(name):
    global username
    username = name
    response = "Welcome " + username + "!"
    return response

def main():
    """
    Main Function.
    """

    # print the game instructions
    print_instructions()

    # get and set user's name
    username = input("What's your name?\r\n")

    # set computer's name
    computer_name = "Computer"

    # insert the rest of the code in the main function here
    run(username, computer_name)

    #print ending message and exit program
    print("Have a nice day!")
    exit()



if __name__ == '__main__':
    main()