"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # my moves - my opponents's moves
    return float(len(game.get_legal_moves(player)) - len(game.get_legal_moves(game.get_opponent(player))))


# some utility constants
POSITIVE_INFINITY = float("inf")
NEGATIVE_INFINITY = float("-inf")

class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        # initialize current best move
        best_move = (-1, -1)

        # when no legal moves available
        if legal_moves is None:
            return (-1, -1)

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            

            # with iterative deepening
            if self.iterative:
                # initialize depth to 1 
                depth = 1
                # go deeper
                while True:
                    # use chosen method to find best move
                    current_best, current_move = getattr(self, self.method)(game, depth)
                    # save current move as best move
                    best_move = current_move
                    # increment depth 
                    depth += 1
            # without iterative deepening
            else:
                # just keep using the search depth
                # use chosen method to find best move
                current_best, current_move = getattr(self, self.method)(game, self.search_depth)
                # save current move as best move
                best_move = current_move

        except Timeout:
            # Handle any actions required at timeout, if necessary
            return best_move

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # when no legal moves available
        if game.get_legal_moves() is None:
            return self.score(game, self), (-1, -1)

        # when depth is zero, reaching the end of tree
        if depth == 0:
            return self.score(game, self), game.get_player_location(self)

        # at the max layer
        if maximizing_player:
            # preassign the current value so that anything could be greater than
            current_max = NEGATIVE_INFINITY
            # loop through the game's subsequent valid moves
            for move in game.get_legal_moves():
                # apply the move and create the next minimum layer
                # and find the either min/max value of that layer
                next_max, next_move = self.minimax(game.forecast_move(move), depth - 1, False)
                # compare it with the current maximum value
                if next_max > current_max:
                    current_max = next_max
                    current_max_move = move
            # propagate the current maximum value up
            return current_max, current_max_move
        # at the min layer
        else:
            # preassign the current value possible so that anything could be less than
            current_min = POSITIVE_INFINITY
            # loop through the game's subsequent valid moves
            for move in game.get_legal_moves():
                # apply the move and create the next maximum layer
                # and find the either min/max value of that layer
                next_min, next_move = self.minimax(game.forecast_move(move), depth - 1, True)
                # compare it with the current minimum value
                if next_min < current_min:
                    current_min = next_min
                    current_min_move = move
            # propagate the current minimum value up
            return current_min, current_min_move



    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # when no legal moves available
        if game.get_legal_moves() is None:
            return self.score(game, self), (-1, -1)

        # when depth is zero, reaching the end of tree
        if depth == 0:
            return self.score(game, self), game.get_player_location(self)

        # at the max layer
        if maximizing_player:
            # preassign the current value so that any thing could be greater than
            current_max = NEGATIVE_INFINITY
            # loop through game's subsequent moves
            for move in game.get_legal_moves():
                # apply the move and create the next minimum layer
                next_max, next_move = self.alphabeta(game.forecast_move(move), depth - 1, alpha, beta, False)
                # compare the next value with current maximum value
                if next_max > current_max:
                    current_max = next_max
                    current_max_move = move
                # compare current alpha with current maximum value
                # so that a lower bound can be updated
                if current_max > alpha:
                    alpha = current_max
                # Sanity check: compare alpha with beta to check if the 
                # upper bound is already smaller than or equal to the  
                # lower bound, if so, ignore the rest of this branch
                if alpha >= beta:
                    break
            return current_max, current_max_move
        # at the min layer
        else:
            # preassign the current value so that anything could be less than
            current_min = POSITIVE_INFINITY
            # loop through the game's subsequent moves
            for move in game.get_legal_moves():
                # apply the move and create the next maximum layer
                next_min, next_move = self.alphabeta(game.forecast_move(move), depth - 1, alpha, beta, True)
                # compare the next value with current mimum value
                if next_min < current_min:
                    current_min = next_min
                    current_min_move = move
                # compare current beta with the current minimum value
                # so that an upper bound can be created
                if current_min < beta:
                    beta = current_min
                # Sanity check: compare alpha with beta to check if the 
                # upper bound is already smaller than or equal to the  
                # lower bound, if so, ignore the rest of this branch
                if alpha >= beta:
                    break
            return current_min, current_min_move

