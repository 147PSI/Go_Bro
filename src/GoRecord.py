from sgfmill import sgf
from datetime import datetime
import os


def set_new_node(sgf_game, color, row, col):
    new_node = sgf_game.extend_main_sequence()
    new_node.set(color, (row, col))

    return sgf_game.serialise().decode("UTF-8")


class GoRecord():

    def __init__(self, sessionId=datetime.now().strftime("%Y%m%dT%H%M%S"), record_storage_path="./"):
        self.sessionId = sessionId
        self.record_storage_path = record_storage_path

    def _check_sgf_file_existence(self):
        for file in os.listdir(self.record_storage_path):
            if file.endswith('sfg') and self.sessionId in file:
                return True
        return False

    def write_new_step_to_sgf_file(self, color, row, col):
        if self._check_sgf_file_existence():
            with open(self.record_storage_path + self.sessionId + ".sfg", "r") as f:
                data = f.read()

                g = sgf.Sgf_game.from_string(data)
                sgf_string = set_new_node(g, color, row, col)

        else:
            g = sgf.Sgf_game(13)
            sgf_string = set_new_node(g, color, row, col)

        with open(self.record_storage_path + self.sessionId + ".sfg", "w") as f:
            f.write(sgf_string)

        return self.get_game_sequence()

    def get_game_sequence(self):
        if self._check_sgf_file_existence():
            with open(self.record_storage_path + self.sessionId + ".sfg", "r") as f:
                data = f.read()
                g = sgf.Sgf_game.from_string(data)
                sgf_string = g.serialise().decode("UTF-8").rstrip("\n")[1:-1]

                sgf_list = sgf_string.split(";")
                # properties = sgf_list[1]
                game_sequence = str(sgf_list[2:])
        else:
            g = sgf.Sgf_game(13)
            sgf_string = g.serialise().decode("UTF-8")
            with open(self.record_storage_path + self.sessionId + ".sfg", "w") as f:
                f.write(sgf_string)
            game_sequence = "New game session has been created!"
        return game_sequence


if __name__ == "__main__":
    go_game = GoRecord()
    go_game.write_new_step_to_sgf_file("W", 1, 4)
    go_game.write_new_step_to_sgf_file("B", 2, 6)
    go_game.write_new_step_to_sgf_file("W", 3, 6)
    go_game.write_new_step_to_sgf_file("B", 2, 8)

    sessionId = go_game.sessionId

    print(go_game.get_game_sequence())

    go_game2 = GoRecord()
    go_game2.write_new_step_to_sgf_file("W", 9, 3)
    go_game2.write_new_step_to_sgf_file("B", 6, 5)
    sessionId2 = go_game2.sessionId

    print(go_game2.get_game_sequence())

    go_game3 = GoRecord(sessionId=sessionId)
    go_game3.write_new_step_to_sgf_file("W", 7, 2)
    go_game3.write_new_step_to_sgf_file("B", 6, 12)

    print(go_game3.get_game_sequence())
