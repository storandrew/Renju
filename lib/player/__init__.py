from ..board import InvalidMoveError

# base class
class Player():
    def __init__(self, color):
        self.color = color

    def make_move(self, gui):
        gui.renew_board()
        if hasattr(gui.board, 'lastmove'):
            gui.highlight_lastmove()
        gui.update()
        gui.color_in_turn = self.color
        moves_left = gui.board.moves_left

        self._make_move(gui)

        if not gui.in_game:
            return

        if not moves_left - 1 == gui.board.moves_left:
            raise InvalidMoveError('Player "%s" did not place a stone.' % self.name)

    def _make_move(self, gui):
        "Override this function for specific players"
        raise NotImplementedError


# Human player
class Human(Player):
    name = 'Human'
    def _make_move(self, gui):
        # wait for user input
        gui.need_user_input = True
        moves_left = gui.board.moves_left
        while gui.board.moves_left == moves_left and gui.in_game:
            gui.update()
        gui.need_user_input = False



# search for player types in all files of this folder
available_player_types = [Human]
from os import listdir, path
player_directory = path.split(__file__)[0]
print('Searching for players in', player_directory)
filenames = listdir(player_directory)
filenames.sort() # search in alphabetical order
for filename in filenames:
    if filename[-3:] != '.py' or 'test' in filename or \
       filename == '__init__.py' or filename == 'lib.py':
           continue
    print('Processing', filename)
    exec('from . import ' + filename[:-3] + ' as playerlib')
    # search for classes derived from the base class ``Player``
    for objname in dir(playerlib):
        obj = playerlib.__dict__[objname]
        if type(obj) is not type or obj in available_player_types:
            continue
        if issubclass(obj, Player) and obj is not Player:
            print('    found', obj.name)
            available_player_types.append(obj)

# player management
available_player_names = [player.name for player in available_player_types]
def get_player_index(name, hint=None):
    for i,n in enumerate(available_player_names):
        if n == name:
            return i
    # the following is executed if the name is not found
    raise ValueError('"%s" is not a registered player type' % name)
