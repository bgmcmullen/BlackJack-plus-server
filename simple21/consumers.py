import json
from channels.generic.websocket import WebsocketConsumer

from  simple21.game.main import Game


class GameConsumer(WebsocketConsumer):
  def connect(self):
    self.game = Game()
    self.accept()

  def disconnect(self, close_code):
    self.close()

  def receive(self, text_data):
    data = json.loads(text_data)
    type = data["type"]
    switch = {
      "get_target_score": self.send_target_score,
      "set_name": self.handle_set_name,
      "run": self.handle_run,
      "take_a_card": self.take_a_card,
      "stand": self.stand
    }

    handler = switch[type]
    if handler:
      handler(data['payload'])

  def take_a_card(self, payload):
    text = self.game.play_turn()
    self.send_status(text)

  def handle_run(self, payload):
    cards = self.game.run()
    self.send_status(cards)

  def handle_set_name(self, name):
    response = self.game.set_user_name(name)
    self.send(text_data=json.dumps({
      'payload': response,
      'type': "welcome_user",
    }))

  def send_target_score(self, payload):
    target_score = self.game.set_target_score()
    self.send(text_data=json.dumps({
      'payload': target_score,
      'type': "set_target_score",
    }))

  def send_status(self, text):
    self.send(text_data=json.dumps({
      'payload': text,
      'type': "set_card_status",
    }))

  def stand(self, playload):
    reponse = self.game.player_passes()
    self.send_status(reponse['cards'])
    self.send(text_data=json.dumps({
      'payload': reponse['winner_dict'],
      'type': "game_end",
    }))
