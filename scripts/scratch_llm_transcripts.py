"""One-off diagnostic: play a handful of real two-team games against
LLMGuesser and print full turn-by-turn transcripts, so a human can read
exactly what happened -- not part of the permanent test/eval surface."""

from codenames.board import Board, OpponentBoardView, Role
from codenames.codemasters import LearnedCodemaster
from codenames.game import DEFAULT_MAX_TURNS, play_two_team_game
from codenames.guessers.llm import LLMGuesser
from codenames.similarity import SimilarityTensor

ROLE_LABEL = {Role.OWN: "OWN", Role.OPPONENT: "OPP", Role.NEUTRAL: "NEU", Role.ASSASSIN: "ASSASSIN"}


def main():
    sims = SimilarityTensor.load()
    codemaster = LearnedCodemaster("cache/m9/checkpoints/noise_0_08/scorer_best.pt")
    guesser = LLMGuesser()

    for seed in range(5):
        board = Board.generate(seed=seed)

        # Capture the full board layout before any word gets revealed --
        # play_two_team_game mutates this same Board object in place.
        by_role = {role: board.words_by_role(role) for role in (Role.OWN, Role.OPPONENT, Role.NEUTRAL, Role.ASSASSIN)}

        result = play_two_team_game(board, (codemaster, guesser), (codemaster, guesser), sims, max_turns=DEFAULT_MAX_TURNS)

        print(f"\n{'=' * 70}\nSEED {seed} -- outcome={result.outcome} winner={result.winner} total_reward={result.total_reward}")
        print(f"  Team A's own (9): {', '.join(by_role[Role.OWN])}")
        print(f"  Team B's own (8): {', '.join(by_role[Role.OPPONENT])}")
        print(f"  Neutral (7):      {', '.join(by_role[Role.NEUTRAL])}")
        print(f"  Assassin (1):     {', '.join(by_role[Role.ASSASSIN])}")
        print()
        for tt in result.turns:
            t = tt.turn
            guesses_str = ", ".join(f"{w}({ROLE_LABEL[r]})" for w, r in t.guesses)
            flag = "  <-- ASSASSIN" if t.ended_reason == "assassin" else ""
            print(f"  [{tt.team}] clue={t.clue!r} n={t.number} -> {guesses_str}  [{t.ended_reason}]{flag}")


if __name__ == "__main__":
    main()
