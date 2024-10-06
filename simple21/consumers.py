import json
from channels.generic.websocket import WebsocketConsumer

from  simple21.game.main import print_instructions, set_user_name, run, play_turn, player_passes


class GameConsumer(WebsocketConsumer):
  def connect(self):
    self.accept()

  def disconnect(self, close_code):
    pass

  def receive(self, text_data):
    data = json.loads(text_data)
    type = data["type"]
    switch = {
      "get_instructions": self.set_instructions,
      "set_name": self.handle_set_name,
      "run": self.handle_run,
      "take_a_card": self.take_a_card,
      "stand": self.stand
    }

    handler = switch[type]
    handler(data['payload'])

  def take_a_card(self, payload):
    text = play_turn()
    self.send_status(text)

  def handle_run(self, payload):
    cards = run()
    self.send_status(cards)

  def handle_set_name(self, name):
    response = set_user_name(name)
    self.send(text_data=json.dumps({
      'payload': response,
      'type': "welcome_user",
    }))

  def set_instructions(self, payload):
    instructions = print_instructions()
    self.send(text_data=json.dumps({
      'payload': instructions,
      'type': "set_instructions",
    }))

  def send_status(self, text):
    self.send(text_data=json.dumps({
      'payload': text,
      'type': "print_status",
    }))

  def stand(self, playload):
    reponse = player_passes()
    self.send_status(reponse['cards'])
    self.send(text_data=json.dumps({
      'payload': reponse['winner_text'],
      'type': "game_end",
    }))