from telethon.sync import TelegramClient  # классс клиента

from telethon.tl.functions.messages import GetDialogsRequest, GetHistoryRequest  # запрос к сообщениям чата
from telethon.tl.types import InputPeerEmpty, Channel, PeerChannel
from typing import Union, List
import emoji
from tqdm import tqdm


class TelegramParser:
    def __init__(self, api_id: int, api_hash: str, phone: str):
        self.api_id = api_id
        self.api_hash = api_hash
        self.client = TelegramClient(phone, api_id, api_hash)
        self.channels = []

    def start(self):
        self.client.start()

    def find_channels(self, limit: int = 500, last_date: str = None) -> List[str]:
        result = self.client(GetDialogsRequest(
            offset_date=last_date,
            offset_id=0,
            hash=0,
            offset_peer=InputPeerEmpty(),
            limit=limit
        ))
        self.channels = []
        for chat in result.chats:
            if isinstance(chat, Channel):
                self.channels.append(chat)
        return [f"{i+1}. {channel.title}" for i, channel in enumerate(self.channels)]

    def parse_channel(self, requested_channel: Union[str, int], limit = 1000) -> List[str]:
        channel = self.__get_channel(requested_channel)
        iteartion_limit = 100

        total_parsed = 0
        offset_id = 0
        messages_list = []

        while total_parsed < limit:
            history = self.client(GetHistoryRequest(
                peer=channel,
                offset_id=offset_id,  # С какого сообщения начинать парсинг
                offset_date=None,  # С какой даты начинать парсинг
                add_offset=0,
                limit=iteartion_limit,
                max_id=0,
                min_id=0,
                hash=0
            ))
            if not history.messages:  #  Кончились сообщения
                break
            filtered_parsed_list = [post.message for post in history.messages if post.message is not None and post.message != ""]
            offset_id = history.messages[-1].id
            total_parsed += len(filtered_parsed_list)
            messages_list.extend(filtered_parsed_list)

        return messages_list[:limit]

    def __get_channel(self, requested_channel: Union[str, int]):
        if isinstance(requested_channel, int):
            return self.channels[requested_channel - 1]
        elif isinstance(requested_channel, str):
            for current_channel in self.channels:
                if current_channel.title == requested_channel:
                    return current_channel


def filter_joke(joke: str) -> bool:
    for el in joke:
        if emoji.is_emoji(el):
            return False
    return joke.find("http") == -1


if __name__ == "__main__":
    api_id = input("Api id: ")
    api_hash = input("Api hash: ")
    phone = input("Phone: ")
    channel_limit = 500000

    parser = TelegramParser(api_id, api_hash, phone)
    parser.start()
    print(parser.find_channels())

    parsed_jokes = []
    channels = ["Анекдоты категории Б", "Анекдоты", "ачё)", "Платиновые анекдоты", "ачё))", "ничё)"]
    for channel in tqdm(channels):
        parsed_channel = [
            joke.replace("\n", " ") + "\n"
            for joke in parser.parse_channel(channel, limit=channel_limit)
            if filter_joke(joke)
        ]
        parsed_jokes.extend(parsed_channel)

    with open("parsed_jokes.txt", "w") as parsed_jokes_file:
        parsed_jokes_file.writelines(parsed_jokes)

