import json
from channels.generic.websocket import WebsocketConsumer

from  simple21.game.main import print_instructions, set_user_name, run


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
      "run": self.handle_run
    }

    handler = switch[type]
    handler(data['payload'])


  def handle_run(self, payload):
    text = run()
    self.send_status(text)

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

