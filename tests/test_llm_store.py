from codenames.board import Role
from codenames.game import TwoTeamGameResult, TwoTeamTurnResult, TurnResult
from codenames.llm_store import GameRecordStore, LLMResponseCache, board_by_role


class _FakeBoard:
    def __init__(self, by_role: dict[Role, list[str]]):
        self._by_role = by_role

    def words_by_role(self, role: Role) -> list[str]:
        return self._by_role[role]


class TestLLMResponseCache:
    def test_miss_returns_none(self, tmp_path):
        cache = LLMResponseCache(tmp_path / "store.db")
        assert cache.get("claude-haiku-4-5", "fruit", ("Apple", "Car"), 1) is None

    def test_put_then_get_round_trips(self, tmp_path):
        cache = LLMResponseCache(tmp_path / "store.db")
        cache.put("claude-haiku-4-5", "fruit", ("Apple", "Car"), 1, ["Car", "Apple"])
        assert cache.get("claude-haiku-4-5", "fruit", ("Apple", "Car"), 1) == ["Car", "Apple"]

    def test_survives_a_fresh_connection_to_the_same_file(self, tmp_path):
        db_path = tmp_path / "store.db"
        LLMResponseCache(db_path).put("claude-haiku-4-5", "fruit", ("Apple", "Car"), 1, ["Car", "Apple"])
        assert LLMResponseCache(db_path).get("claude-haiku-4-5", "fruit", ("Apple", "Car"), 1) == ["Car", "Apple"]

    def test_different_model_is_a_separate_entry(self, tmp_path):
        cache = LLMResponseCache(tmp_path / "store.db")
        cache.put("claude-haiku-4-5", "fruit", ("Apple", "Car"), 1, ["Apple", "Car"])
        assert cache.get("claude-sonnet-5", "fruit", ("Apple", "Car"), 1) is None

    def test_different_candidate_order_is_a_separate_entry(self, tmp_path):
        cache = LLMResponseCache(tmp_path / "store.db")
        cache.put("claude-haiku-4-5", "fruit", ("Apple", "Car"), 1, ["Apple", "Car"])
        assert cache.get("claude-haiku-4-5", "fruit", ("Car", "Apple"), 1) is None


class TestBoardByRole:
    def test_snapshots_every_role(self):
        board = _FakeBoard(
            {Role.OWN: ["a"], Role.OPPONENT: ["b"], Role.NEUTRAL: ["c"], Role.ASSASSIN: ["d"]}
        )
        assert board_by_role(board) == {Role.OWN: ["a"], Role.OPPONENT: ["b"], Role.NEUTRAL: ["c"], Role.ASSASSIN: ["d"]}


class TestGameRecordStore:
    def _result(self) -> TwoTeamGameResult:
        turn = TurnResult(clue="shores", number=2, guesses=[("Port", Role.OWN), ("Seal", Role.NEUTRAL)], ended_reason="neutral")
        return TwoTeamGameResult(
            seed=3,
            turns=[TwoTeamTurnResult(team="A", turn=turn)],
            outcome="win",
            winner="A",
            total_reward={"A": 0.8, "B": 0.0},
        )

    def _by_role(self) -> dict[Role, list[str]]:
        return {Role.OWN: ["Port"], Role.OPPONENT: ["England"], Role.NEUTRAL: ["Seal"], Role.ASSASSIN: ["Loch ness"]}

    def test_add_then_all_games_round_trips(self, tmp_path):
        store = GameRecordStore(tmp_path / "store.db")
        store.add_game(self._by_role(), self._result(), label="haiku-trial")
        rows = store.all_games()
        assert len(rows) == 1
        assert rows[0]["seed"] == 3
        assert rows[0]["label"] == "haiku-trial"
        assert rows[0]["outcome"] == "win"
        assert rows[0]["winner"] == "A"

    def test_filter_by_label(self, tmp_path):
        store = GameRecordStore(tmp_path / "store.db")
        store.add_game(self._by_role(), self._result(), label="haiku-trial")
        store.add_game(self._by_role(), self._result(), label="sonnet-trial")
        assert len(store.all_games(label="sonnet-trial")) == 1

    def test_stored_board_and_turns_round_trip_through_json(self, tmp_path):
        import json

        store = GameRecordStore(tmp_path / "store.db")
        store.add_game(self._by_role(), self._result(), label="haiku-trial")
        row = store.all_games()[0]
        board = json.loads(row["board"])
        turns = json.loads(row["turns"])
        assert board == {"own": ["Port"], "opponent": ["England"], "neutral": ["Seal"], "assassin": ["Loch ness"]}
        assert turns == [
            {
                "team": "A",
                "clue": "shores",
                "number": 2,
                "guesses": [["Port", "own"], ["Seal", "neutral"]],
                "ended_reason": "neutral",
            }
        ]
