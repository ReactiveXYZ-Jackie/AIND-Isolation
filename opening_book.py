class OpeningBook(object):

	def __init__(self):
		self.load_rules()

	def set_game(self, game):
		self.game = game

	def set_current_player(self, player):
		self.player = player

	def move_for(self, game):
		self.set_game(game)
		for rule in self.rules:
			possible_move = getattr(self, rule)()
			if possible_move is not None:
				return possible_move
		return None

	def load_rules(self):
		self.rules = ["occupy_center_square"]

	def occupy_center_square(self):
		# area of board
		area = self.game.width * self.game.height
		# available blank spaces indicating game state
		blank_spaces = self.game.get_blank_spaces()
		# check if the game is just starting
		if (blank_spaces == area and self.game.is_player_one(self.player) or (blank_spaces == area - 1 and self.game.is_player_two(self.player))):
			# center move
			center_move = (int(self.game.width / 2), int(self.game.height / 2))
			# check if is legal
			if self.game.move_is_legal(center_move):
				return center_move
		
		return None

	# TODO: Buggy
	def find_reflection_move(self):
		# for now, only odd length and odd width board works
		if self.game.width % 2 == 1 and self.game.height % 2 == 1 and self.game.is_player_one(self.player):
			# fetch opponent's move
			opponent_move = self.game.get_player_location(self.game.get_opponent(self.player))
			# sanity check whether opponent move is valid
			if not opponent_move:
				return None
			# find its reflection
			reflected_move = (self.game.height - 1 - opponent_move[0], self.game.width - 1 - opponent_move[0])
			# check if is legal
			if reflected_move in self.game.get_legal_moves(self.player):
				return reflected_move
			
		return None
		
