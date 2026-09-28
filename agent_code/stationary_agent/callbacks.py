"""Diagnostic opponent that always waits: never moves and never bombs."""


def setup(self):
    pass


def act(self, game_state: dict) -> str:
    return "WAIT"
