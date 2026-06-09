# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Werewolf game."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue
import random
import threading
from typing import Any, Iterable, List, Optional, Tuple
import copy

import tqdm

from werewolf.model import Round, RoundLog, State, VoteLog, Logger
from werewolf.config import  MAX_DEBATE_TURNS, RUN_SYNTHETIC_VOTES
from werewolf.logging import store,log_directory
from werewolf.lm import LmLog

def get_max_bids(d):
  """Gets all the keys with the highest value in the dictionary."""
  max_value = max(d.values())
  max_keys = [key for key, value in d.items() if value == max_value]
  return max_keys


class GameMaster:

  def __init__(
      self,
      state: State,
      num_threads: int = 1,
  ) -> None:
    """Initialize the Werewolf game.

    Args:
    """
    self.state = state
    self.current_round_num = len(self.state.rounds) if self.state.rounds else 0
    self.num_threads = num_threads
    self.logs: List[RoundLog] = []
    
  @property
  def this_round(self) -> Round:
    return self.state.rounds[self.current_round_num]

  @property
  def this_round_log(self) -> RoundLog:
    return self.logs[self.current_round_num]

  def eliminate(self):
    """Werewolves choose a player to eliminate."""
    werewolves_alive = [
        w for w in self.state.werewolves if w.name in self.this_round.players
    ]
    wolf = random.choice(werewolves_alive)
    eliminated, log = wolf.eliminate()
    self.this_round_log.eliminate = log
    if eliminated is not None:
      self.this_round.eliminated = eliminated
      tqdm.tqdm.write(f"{wolf.name} eliminated {eliminated}")
    else:
      eliminated = random.choice(self.this_round.players)
      self.this_round.eliminated = eliminated
      tqdm.tqdm.write(
          f"{wolf.name} did not eliminate anyone, randomly choosing {eliminated}.")
      # raise ValueError("Eliminate did not return a valid player.")
    for wolf in werewolves_alive:
        wolf._add_observation(
            "During the"
            f" night, {'we' if len(werewolves_alive) > 1 else 'I'} decided to"
            f" eliminate {eliminated}."
        )

  def protect(self):
    """Doctor chooses a player to protect."""
    if self.state.doctor.name not in self.this_round.players:
      return  # Doctor no longer in the game

    protect, log = self.state.doctor.save()
    self.this_round_log.protect = log

    if protect is not None:
      self.this_round.protected = protect
      tqdm.tqdm.write(f"{self.state.doctor.name} protected {protect}")
    else:
      protect = random.choice(self.this_round.players)
      self.this_round.protected = protect
      tqdm.tqdm.write(f"Did not protect, randomly choosing {protect}.")
      # raise ValueError("Protect did not return a valid player.")

  def unmask(self):
    """Seer chooses a player to unmask."""
    if self.state.seer.name not in self.this_round.players:
      return  # Seer no longer in the game

    unmask, log = self.state.seer.unmask()
    self.this_round_log.investigate = log

    if unmask is not None:
      self.this_round.unmasked = unmask
    else:
      unmask = random.choice(self.this_round.players)
      self.this_round.unmasked = unmask
      tqdm.tqdm.write(f"Did not unmask, randomly choosing {unmask}.")
      # raise ValueError("Unmask function did not return a valid player.")
    self.state.seer.reveal_and_update(unmask, self.state.players[unmask].role)

  def _get_bid(self, player_name):
    """Gets the bid for a specific player."""
    player = self.state.players[player_name]
    bid, log = player.bid()
    if bid is None:
      raise ValueError(
          f"{player_name} did not return a valid bid. Find the raw response"
          " in the `bid` field in the log"
      )
    if bid > 1:
      tqdm.tqdm.write(f"{player_name} bid: {bid}")
    return bid, log

  def get_next_speaker(self):
    return self.this_round.players[0]
    """Determine the next speaker based on bids."""
    previous_speaker, previous_dialogue = (
        self.this_round.debate[-1] if self.this_round.debate else (None, None)
    )

    with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
      player_bids = {
          player_name: executor.submit(self._get_bid, player_name)
          for player_name in self.this_round.players
          if player_name != previous_speaker
      }

      bid_log = []
      bids = {}
      try:
        for player_name, bid_task in player_bids.items():
          bid, log = bid_task.result()
          bids[player_name] = bid
          bid_log.append((player_name, log))
      except TypeError as e:
        print(e)
        raise e

    self.this_round.bids.append(bids)
    self.this_round_log.bid.append(bid_log)

    potential_speakers = get_max_bids(bids)
    # Prioritize mentioned speakers if there's previous dialogue
    if previous_dialogue:
      potential_speakers.extend(
          [name for name in potential_speakers if name in previous_dialogue]
      )

    random.shuffle(potential_speakers)
    return random.choice(potential_speakers)

  def run_summaries(self):
    """Collect summaries from players after the debate."""

    with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
      player_summaries = {
          name: executor.submit(self.state.players[name].summarize)
          for name in self.this_round.players
      }

      for player_name, summary_task in player_summaries.items():
        summary, log = summary_task.result()
        tqdm.tqdm.write(f"{player_name} summary: {summary}")
        self.this_round_log.summaries.append((player_name, log))

  def run_day_phase(self):
    """Run the day phase which consists of the debate and voting."""

    # for idx in range(MAX_DEBATE_TURNS):
      # next_speaker = self.get_next_speaker()
    random.shuffle(self.this_round.players)
    for next_speaker in self.this_round.players:
      if not next_speaker:
        raise ValueError("get_next_speaker did not return a valid player.")

      player = self.state.players[next_speaker]
      dialogue, log = player.debate()
      if dialogue is None:
        raise ValueError(
            f"{next_speaker} did not return a valid dialouge from debate()."
        )

      self.this_round_log.debate.append((next_speaker, log))
      self.this_round.debate.append([next_speaker, dialogue])
      tqdm.tqdm.write(f"{next_speaker} ({player.role}): {dialogue}")

      for name in self.this_round.players:
        player = self.state.players[name]
        if player.gamestate:
          player.gamestate.update_debate(next_speaker, dialogue)
        else:
          raise ValueError(f"{name}.gamestate needs to be initialized.")

      # if idx == MAX_DEBATE_TURNS - 1 or RUN_SYNTHETIC_VOTES:
      #   votes, vote_logs = self.run_voting()
      #   self.this_round.votes.append(votes)
      #   self.this_round_log.votes.append(vote_logs)
    votes, vote_logs = self.run_voting()
    self.this_round.votes.append(votes)
    self.this_round_log.votes.append(vote_logs)

    for player, vote in self.this_round.votes[-1].items():
      tqdm.tqdm.write(f"{player} voted to remove {vote}")

  def run_voting(self):
    """Conduct a vote among players to exile someone."""
    vote_log = []
    votes = {}

    with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
      player_votes = {
          name: executor.submit(self.state.players[name].vote)
          for name in self.this_round.players
      }

      for player_name, vote_task in player_votes.items():
        vote, log = vote_task.result()
        vote_log.append(VoteLog(player_name, vote, log))

        if vote is  None:
          vote = random.choice(self.this_round.players)
          tqdm.tqdm.write(f"{player_name} did not vote, randomly choosing {vote}.")
        votes[player_name] = vote
        # else:
        #   self.this_round.votes.append(votes)
        #   self.this_round_log.votes.append(vote_log)
        #   raise ValueError(f"{player_name} vote did not return a valid player.")

    return votes, vote_log

  def exile(self):
    """Exile the player who received the most votes."""

    most_voted, vote_count = Counter(
        self.this_round.votes[-1].values()
    ).most_common(1)[0]

    if vote_count > len(self.this_round.players) / 2:
      self.this_round.exiled = most_voted

    if self.this_round.exiled is not None:
      exiled_player = self.this_round.exiled
      self.this_round.players.remove(exiled_player)
      announcement = (
          f"The majority voted to remove {exiled_player} from the game."
      )
    else:
      announcement = (
          "A majority vote was not reached, so no one was removed from the"
          " game."
      )

    for name in self.this_round.players:
      player = self.state.players[name]
      if player.gamestate and self.this_round.exiled is not None:
        player.gamestate.remove_player(self.this_round.exiled)
      player.add_announcement(announcement)

    tqdm.tqdm.write(announcement)

  def resolve_night_phase(self):
    """Resolve elimination and protection during the night phase."""
    if self.this_round.eliminated != self.this_round.protected:
      eliminated_player = self.this_round.eliminated
      self.this_round.players.remove(eliminated_player)
      announcement = (
          f"The Werewolves removed {eliminated_player} from the game during the"
          " night."
      )
    else:
      announcement = "No one was removed from the game during the night."
    tqdm.tqdm.write(announcement)

    for name in self.this_round.players:
      player = self.state.players[name]
      if player.gamestate:
        player.gamestate.remove_player(self.this_round.eliminated)
      player.add_announcement(announcement)

  def run_round(self):
    """Run a single round of the game."""
    self.state.rounds.append(Round())
    self.logs.append(RoundLog())

    self.this_round.players = (
        list(self.state.players.keys())
        if self.current_round_num == 0
        else self.state.rounds[self.current_round_num - 1].players.copy()
    )

    for action, message in [
        (
            self.eliminate,
            "The Werewolves are picking someone to remove from the game.",
        ),
        (self.protect, "The Doctor is protecting someone."),
        (self.unmask, "The Seer is investigating someone."),
        (self.resolve_night_phase, ""),
        (self.check_for_winner, "Checking for a winner after Night Phase."),
        (self.run_day_phase, "The Players are debating and voting."),
        (self.exile, ""),
        (self.check_for_winner, "Checking for a winner after Day Phase."),
        (self.run_summaries, "The Players are summarizing the debate."),
    ]:
      tqdm.tqdm.write(message)
      action()

      if self.state.winner:
        tqdm.tqdm.write(f"Round {self.current_round_num} is complete.")
        self.this_round.success = True
        return

    tqdm.tqdm.write(f"Round {self.current_round_num} is complete.")
    self.this_round.success = True

  def get_winner(self) -> str:
    """Determine the winner of the game."""
    active_wolves = set(self.this_round.players) & set(
        w.name for w in self.state.werewolves
    )
    active_villagers = set(self.this_round.players) - active_wolves
    if len(active_wolves) >= len(active_villagers):
      return "Werewolves"
    return "Villagers" if not active_wolves else ""

  def check_for_winner(self):
    """Check if there is a winner and update the state accordingly."""
    self.state.winner = self.get_winner()
    if self.state.winner:
      tqdm.tqdm.write(f"The winner is {self.state.winner}!")

  def run_game(self) -> str:
    """Run the entire Werewolf game and return the winner."""
    while not self.state.winner:
      tqdm.tqdm.write(f"STARTING ROUND: {self.current_round_num}")
      self.run_round()
      for name in self.this_round.players:
        if self.state.players[name].gamestate:
          self.state.players[name].gamestate.round_number = (
              self.current_round_num + 1
          )
          self.state.players[name].gamestate.clear_debate()
      self.current_round_num += 1

    tqdm.tqdm.write("Game is complete!")
    return self.state.winner


class RecursiveGameMaster(GameMaster):
  def __init__(self, state: State, num_threads: int = 1):
    super().__init__(state, num_threads)
    self.logger = Logger(round=0)
    self.log_directory = log_directory()
    
  
  def run_day_phase(self):
    random.shuffle(self.this_round.players)
    for name in self.this_round.players:
      player = self.state.players[name]
      tqdm.tqdm.write(f"{name} ({player.role}) is in the debate phase.")
    # exit(0)
      
    self.logger = Logger(round=self.current_round_num)
    self.player_debate(0, [])
    store(self.logger,self.log_directory)
    tqdm.tqdm.write("Debate and voting complete.\n\n\n\n")

  def player_debate(self, player_id, muteplayers):
    """Run the day phase recursively which consists of the debate and voting."""
    if(player_id>=len(self.this_round.players)):
      self.vote_and_eval(muteplayers)
      return

    next_speaker = self.this_round.players[player_id]
    player_entity = self.state.players[next_speaker]
    ## skipping this player
    if len(muteplayers) <= len(self.this_round.players)/2:
      new_master = copy.deepcopy(self)
      new_master.logger = self.logger
      say = "I choose to skip my turn and not make a statement this round."
      new_master.this_round.debate.append([player_entity,say])

      for name in new_master.this_round.players:
        player = new_master.state.players[name]
        if player.gamestate:
          player.gamestate.update_debate(next_speaker, say)
        else:
          raise ValueError(f"{name}.gamestate needs to be initialized.")
      
      new_master.player_debate(player_id+1, muteplayers+ [player_id])

    ## not skipping
    dialogue, log = player_entity.debate()
    if dialogue is None:
      raise ValueError(
          f"{next_speaker} did not return a valid dialouge from debate()."
          + log.to_json_minimal()
      )
    mutednames = tuple(self.this_round.players[i] for i in muteplayers)
    self.logger.add_log(log.to_json_minimal(), next_speaker, mutednames)

    self.this_round_log.debate.append((next_speaker, log))
    self.this_round.debate.append([next_speaker, dialogue])
    tqdm.tqdm.write(f"{next_speaker} ({player_entity.role}): {dialogue}")

    for name in self.this_round.players:
      player = self.state.players[name]
      if player.gamestate:
        player.gamestate.update_debate(next_speaker, dialogue)
      else:
        raise ValueError(f"{name}.gamestate needs to be initialized.")

    self.player_debate(player_id+1, muteplayers)

  def vote_and_eval(self, muteplayers):
    # Find the index of the last muted player. If no one is muted, start
    # from -1 so that the next index becomes 0 (the first player).
    if muteplayers:
      last_muted = max(muteplayers)
    else:
      last_muted = -1

    votes, vote_logs = self.run_voting()
    # for item in vote_logs:
    #     tqdm.tqdm.write("vote_logs:"+item.log.to_json_minimal()+"\n\n")
    # if len(muteplayers) == len(self.this_round.players)-1:
    #   raise ValueError("All players are muted, cannot proceed with voting evaluation.")
    self.this_round.votes.append(votes)
    self.this_round_log.votes.append(vote_logs)

    # If next_idx is within the player list, get the player name and then
    # search the debate log in reverse for their most recent utterance.
    mutednames = tuple(self.this_round.players[i] for i in muteplayers)
    for idx in range(last_muted+1,len(self.this_round.players)):
      speaker = self.this_round.players[idx]
      self.logger.add_pvote_log(votes,speaker, mutednames)
    
    if last_muted >= 0:
      speaker = self.this_round.players[last_muted]
      mutednames = tuple(self.this_round.players[i] for i in muteplayers[:-1])
      self.logger.add_nvote_log(votes,speaker, mutednames)

  def exile_player(self):
    most_voted, vote_count = Counter(
        self.this_round.votes[-1].values()
    ).most_common(1)[0]

    if vote_count > len(self.this_round.players) / 2:
      return most_voted
    
    return None
  

class InterruptGameMaster(GameMaster):
  def __init__(self, state: State, num_threads: int = 1):
    super().__init__(state, num_threads)
    self.interrupted = False

  def _coerce_interrupt_result(self, result: Any) -> bool:
    """Normalize different interrupt return styles to a boolean."""
    if isinstance(result, tuple):
      result = result[0]
    if isinstance(result, str):
      lowered = result.strip().lower()
      if lowered in {"true", "1", "yes"}:
        return True
      if lowered in {"false", "0", "no", ""}:
        return False
    return bool(result)

  def _broadcast_debate_chunk(
      self,
      speaker_name: str,
      chunk: str,
  ) -> None:
    """Print a streamed chunk as it arrives."""
    tqdm.tqdm.write(f"{speaker_name}: {chunk}")

  def _run_interrupt_check(
      self,
      speaker_name: str,
      next_speaker_name: str,
      speech_queue: Queue,
      speech_finished_event: threading.Event,
      stop_event: threading.Event,
      result_box: dict[str, Any],
  ):
    """Let the next speaker watch the stream and decide whether to interrupt."""
    transcript = ""
    buffer = ""
    player = self.state.players[next_speaker_name]

    try:
      while not stop_event.is_set():
        try:
          item = speech_queue.get(timeout=0.05)
        except Empty:
          if result_box.get("done"):
            break
          continue

        if item is None:
          break
        if isinstance(item, Exception):
          result_box["error"] = item
          stop_event.set()
          break
        
        buffer += item
        if len(buffer) < 100:
          continue
        else:
          transcript += buffer
          buffer = ""

        try:
          should_interrupt = player.interrupt(
              current_speaker=speaker_name,
              current_speech=transcript,
          )
        except NotImplementedError:
          should_interrupt = False

        result_box["value"] = self._coerce_interrupt_result(should_interrupt)


        ## Random Intterupt for testing
        # result_box["value"] = (random.randint(0,9) == 0) 

        # print(f"\n [Testing] Next player choice:{result_box["value"]}\n")

        if result_box["value"] and not speech_finished_event.is_set():
          stop_event.set()
          break
        if result_box["value"] and speech_finished_event.is_set():
          result_box["value"] = False
    except Exception as exc:  # noqa: BLE001
      result_box["error"] = exc
      result_box["value"] = False
      stop_event.set()
    finally:
      result_box["done"] = True

  def _run_debate_turn(self, speaker_name: str, next_speaker: Optional[str]) -> bool:
    """Run one speaker's turn and return whether it was interrupted."""
    player = self.state.players[speaker_name]
    reasoning, reasoning_log = player.reason()
    if reasoning is None:
      # raise ValueError(f"{speaker_name} did not return a valid reasoning.")
      print(f"{speaker_name} did not return a valid reasoning.")
      reasoning = ""

    if player.gamestate:
      player.gamestate.set_last_thought(reasoning)

    tqdm.tqdm.write(f"{speaker_name} reasoning: {reasoning}")

    speech_stream = player.say()
    stop_event = threading.Event()
    speech_finished_event = threading.Event()
    display_queue: Queue = Queue()
    interrupt_queue: Queue = Queue()
    interrupt_state: dict[str, Any] = {"done": False, "value": False}
    interrupt_thread: Optional[threading.Thread] = None
    speech_chunks: list[str] = []
    speech_error: Optional[Exception] = None

    if next_speaker:
      interrupt_thread = threading.Thread(
          target=self._run_interrupt_check,
          args=(
          speaker_name,
          next_speaker,
          interrupt_queue,
          speech_finished_event,
          stop_event,
          interrupt_state,
          ),
          daemon=True,
      )
      interrupt_thread.start()

    def speech_worker():
      try:
        for chunk in speech_stream:
          if stop_event.is_set():
            break

          display_queue.put(chunk)
          interrupt_queue.put(chunk)
        display_queue.put(None)
        interrupt_queue.put(None)
        speech_finished_event.set()
      except Exception as exc:  # noqa: BLE001
        display_queue.put(exc)
        interrupt_queue.put(exc)

    debate_thread = threading.Thread(target=speech_worker, daemon=True)
    debate_thread.start()

    tqdm.tqdm.write(f"{speaker_name}: ")
    while True:
      if stop_event.is_set() and display_queue.empty():
        break

      try:
        item = display_queue.get(timeout=0.05)
      except Empty:
        if not debate_thread.is_alive() and display_queue.empty():
          break
        continue

      if item is None:
        speech_finished_event.set()
        break
      if isinstance(item, Exception):
        speech_error = item
        break

      speech_chunk = item
      if not speech_chunk:
        continue

      speech_chunks.append(speech_chunk)
      # self._broadcast_debate_chunk(speaker_name, speech_chunk)
      print(speech_chunk, end="", flush=True)
      if stop_event.is_set():
        self.interrupted = True
        break

    debate_thread.join()
    if interrupt_thread is not None:
      interrupt_thread.join()

    if "error" in interrupt_state:
      raise interrupt_state["error"]
    if speech_error is not None:
      raise speech_error

    speech = "".join(speech_chunks)
    self.this_round.debate.append((speaker_name, speech))
    for name in self.this_round.players:
      current_player = self.state.players[name]
      if current_player.gamestate:
        current_player.gamestate.update_debate(speaker_name, speech)
      else:
        raise ValueError(f"{name}.gamestate needs to be initialized.")

    debate_log = speech_stream.log or LmLog(
        prompt=speech_stream.prompt,
        raw_resp=speech,
        result={"reasoning": reasoning, "say": speech},
    )
    debate_log.result = {
        "reasoning": reasoning,
        "say": speech,
        "interrupted": bool(interrupt_state.get("value", False)),
    }
    self.this_round_log.debate.append((speaker_name, debate_log))

    interrupted = bool(interrupt_state.get("value", False))
    if interrupted and next_speaker:
      tqdm.tqdm.write(
          f"\n 打断成功: {next_speaker} 打断了 {speaker_name} 的发言。\n"
      )

    return interrupted

  def run_day_phase(self):
    random.shuffle(self.this_round.players)

    speaker_idx = 0
    while speaker_idx < len(self.this_round.players):
      speaker_name = self.this_round.players[speaker_idx]
      if not speaker_name:
        raise ValueError("run_day_phase did not return a valid player.")

      next_speaker = (
          self.this_round.players[speaker_idx + 1]
          if speaker_idx + 1 < len(self.this_round.players)
          else None
      )

      self.interrupted = False
      self._run_debate_turn(speaker_name, next_speaker)
      speaker_idx += 1

    votes, vote_logs = self.run_voting()
    self.this_round.votes.append(votes)
    self.this_round_log.votes.append(vote_logs)

    for player, vote in self.this_round.votes[-1].items():
      tqdm.tqdm.write(f"{player} voted to remove {vote}")
