import telegram

def send(msg, chat_id = 190649040, token='792681420:AAH_wiAsCO5Bk1kan_Iy3LTaJjDl3gWOZBU'):
  bot = telegram.Bot(token=token)
  bot.sendMessage(chat_id=chat_id, text=msg)


# send('IM BACK BITCHES')