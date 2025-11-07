import argparse
import itertools
import math
import os
import re
import copy
from itertools import zip_longest
import numpy as np
import scipy.stats
import scipy.special
from abc import abstractmethod
from dataclasses import dataclass, field
from sgfmill import sgf
from colorama import init, Style
from typing import List, Dict, Tuple, Sequence

Player = str
PlayerIdx = int

# ========HOW TO USE========================================================
# py summarize_elo.py "path" -standard {number} -prior {number} -create {v/p/t} -color -first-advantage -recursive
# py summarize_elo.py -h

# ========SETTINGS TO COVER THE COMMAND=====================================

# Customize the standard deviation
# Standard_Deviation = 2

# number of games for bayesian prior around Elo 0
# Prior_Games = 0

# ========================================================

ELO_PER_STRENGTH = 400.0 * math.log10(math.exp(1.0))
P1_ADVANTAGE_NAME = "P1Advantage"

# Color module
init()
def color_256(code): 
    return f"\033[38;5;{code}m" if Color else ""

class EloInfo:
    """Summarizes maximum likelihood Elos and uncertainties for a group of players."""

    def __init__(
        self,
        players: List[Player],
        elo: Dict[Player,float],
        elo_stderr: Dict[Player,float],
        elo_covariance: Dict[Tuple[Player,Player],float],
        effective_game_count: Dict[Player,float],
    ):
        self.players = players
        self.elo = elo
        self.elo_stderr = elo_stderr
        self.elo_covariance = elo_covariance
        self.effective_game_count = effective_game_count

        self.players = sorted(self.players, key=(lambda player: -elo[player] if player != P1_ADVANTAGE_NAME else 1e50))

    def get_players(self) -> List[Player]:
        return self.players

    def get_elo(self, p1: Player) -> float:
        """Returns the maximum likelihood Elo of p1"""
        return self.elo[p1]

    def get_approx_elo_stderr(self, p1: Player) -> float:
        """Returns an approximation of the standard error on the Elo of p1, ASSUMING all other players Elos are equal to their maximum likelihood value.
        This approximation may underestimate if the amount of data is very small."""
        return self.elo_stderr[p1]

    def get_elo_difference(self, p1: Player, p2: Player) -> float:
        """Returns the maximum likelhood difference in Elo between p1 and p2"""
        return self.elo[p1] - self.elo[p2]

    def get_approx_elo_difference_stderr(self, p1: Player, p2: Player) -> float:
        """Returns an approximation of the standard error on difference in Elo between p1 and p2.
        This approximation may underestimate if the amount of data is very small."""
        return standard_deviation * math.sqrt(
          self.elo_covariance[(p1,p1)] - self.elo_covariance[(p1,p2)] - self.elo_covariance[(p2,p1)] + self.elo_covariance[(p2,p2)]
        )

    def get_approx_likelihood_of_superiority(self, p1: Player, p2: Player) -> float:
        """Returns an approximation of the likelihood that elo(p1) - elo(p2) > 0, given the data.
        This approximation may be overconfident or inaccurate if the amount of data is very small."""
        if p1 == p2:
            return 0.5
        mean = self.get_elo_difference(p1,p2)
        stderr = self.get_approx_elo_difference_stderr(p1,p2)
        return scipy.stats.t.cdf(mean/stderr,df=self.effective_game_count[p1]-1)

    def get_log10_odds_surprise_max_likelihood(self, p1: Player, p2: Player, g1: float, total: int) -> float:
        """Returns an indication of how surprised we would be for p1 to win g1 games out of total versus p2, given the maximum likeihood Elos.
        Doesn't explicitly model draw prob, if g1 is fractional then assumes the fractional part randomly rounds to a full 1 or 0
        and averages the probabilities.
        A value of 3.0 means that g1 is larger than we would expect given the players' estimated Elos, such that a result
        that extreme or more has less than 1 : 10^3.0 odds of happening under the posterior.
        A value of -3.0 means that g1 is smaller larger than we would expect given the players' estimated Elos, such that a result
        that extreme or more has less than 1 : 10^3.0 odds of happening under the posterior.
        """
        if total <= 0:
            return 1
        mean = self.get_elo_difference(p1,p2) / ELO_PER_STRENGTH

        max_likelihood_winprob = 1.0 / (1.0 + math.exp(-mean))
        max_likelihood_g1 = max_likelihood_winprob * total
        if max_likelihood_g1 < g1:
            signflip = -1
            mean = -mean
            g1 = total - g1
        else:
            signflip = 1

        winprob = 1.0 / (1.0 + math.exp(-mean))

        g1_floor = int(math.floor(g1))
        g1_frac = g1 - g1_floor
        if g1_frac != 0.0:
            # Average endpoints for draws
            logxf = scipy.stats.binom.logcdf(g1_floor,total,winprob)
            logxp1 = scipy.stats.binom.logcdf(g1_floor+1,total,winprob)
            log_prob = scipy.special.logsumexp([logxf+math.log(1.0-g1_frac),logxp1+math.log(g1_frac)])
        else:
            log_prob = scipy.stats.binom.logcdf(g1,total,winprob)

        log_odds = log_prob - (math.log(-log_prob) if log_prob > -1e-10 else math.log(1.0 - math.exp(log_prob)))
        return -signflip * max(0.0, -log_odds / math.log(10.0))

    def get_approx_log10_odds_surprise_bayes(self, p1: Player, p2: Player, g1: float, total: int) -> float:
        """Returns an indication of how surprised we would be for p1 to win g1 games out of total versus p2, given the posterior.
        Computed via numeric integration over the pairwise posteriors.
        Doesn't explicitly model draw prob, if g1 is fractional then assumes the fractional part randomly rounds to a full 1 or 0
        and averages the probabilities.
        A value of 3.0 means that g1 is larger than we would expect given the players' estimated Elos, such that a result
        that extreme or more has less than 1 : 10^3.0 odds of happening under the posterior.
        A value of -3.0 means that g1 is smaller larger than we would expect given the players' estimated Elos, such that a result
        that extreme or more has less than 1 : 10^3.0 odds of happening under the posterior.
        """
        mean = self.get_elo_difference(p1,p2) / ELO_PER_STRENGTH
        stderr = self.get_approx_elo_difference_stderr(p1,p2) / ELO_PER_STRENGTH

        max_likelihood_winprob = 1.0 / (1.0 + math.exp(-mean))
        max_likelihood_g1 = max_likelihood_winprob * total
        if max_likelihood_g1 < g1:
            signflip = -1
            mean = -mean
            g1 = total - g1
        else:
            signflip = 1

        logw_list = []
        logwx_list = []
        for x in np.linspace(-15.0,15.0,num=1000,endpoint=False):
            logw = scipy.stats.norm.logpdf(x)
            winprob = 1.0 / (1.0 + math.exp(-(mean + x * stderr)))
            # print("X", x)
            # print("winprob", winprob)
            # print("g1 total", g1, total, g1/total)

            g1_floor = int(math.floor(g1))
            g1_frac = g1 - g1_floor
            if g1_frac != 0.0:
                # Average endpoints for draws
                logxf = scipy.stats.binom.logcdf(g1_floor,total,winprob)
                logxp1 = scipy.stats.binom.logcdf(g1_floor+1,total,winprob)
                log_prob = scipy.special.logsumexp([logxf+math.log(1.0-g1_frac),logxp1+math.log(g1_frac)])
            else:
                logx = scipy.stats.binom.logcdf(g1,total,winprob)

            # print("logw logx", logw, logx)
            logw_list.append(logw)
            logwx_list.append(logw+logx)

        log_wsum = scipy.special.logsumexp(logw_list)
        log_wxsum = scipy.special.logsumexp(logwx_list)
        # print("log_wsum",log_wsum)
        # print("log_wxsum",log_wxsum)

        log_prob = log_wxsum - log_wsum
        log_odds = log_prob - (math.log(-log_prob) if log_prob > -1e-10 else math.log(1.0 - math.exp(log_prob)))
        return -signflip * max(0.0, -log_odds / math.log(10.0))

    def __str__(self) -> str:
        pla_names = list(self.elo.keys())
        max_name_len = max(len(str(name)) for name in pla_names) if pla_names else 0
        lines = []
        for player in self.players:
            lines.append(f"{str(player):>{max_name_len}}:{self.elo[player]:9.1f} +/- {self.elo_stderr[player]:<.1f}")
        return "\n".join(lines)


class Likelihood:
    """Summarizes the information in an observed datapoint about player Elos, or a prior about them.

    Represents that sum_{p in playercombo} Strength(p)*playercombo[p] + offset has likelihood function f raised to the power of weight.
    If kind is SIGMOID_KIND, f is the sigmoid function.
    If kind is GAUSSIAN_KIND, f is the pdf of a unit gaussian

    where strength is such that 1 unit of strength is e:1 odds of winning in a head-to-head game.
    """

    SIGMOID_KIND = 1
    GAUSSIAN_KIND = 2

    def __init__(
        self,
        playercombo: Dict[Player,float],
        offset: float,
        weight: float,
        gamecount: float,
        kind: int,
    ):
        self.playercombo = playercombo
        self.offset = offset
        self.weight = weight
        self.gamecount = gamecount
        self.kind = kind
        assert kind == Likelihood.SIGMOID_KIND or kind == Likelihood.GAUSSIAN_KIND, "invalid kind"

    def add_idxs(self, player_to_idx: Dict[Player,PlayerIdx]):
        self.pidxcombo : List[Tuple[PlayerIdx,float]] = [(player_to_idx[player],coeff) for (player,coeff) in self.playercombo.items()]

    LOG_ONE_OVER_SQRT_TWO_PI = math.log(1.0 / math.sqrt(2.0 * math.pi))

    def get_loglikelihood(self, strengths: np.array) -> float:
        strength_total = self.offset + sum(strengths[pidx] * coeff for (pidx,coeff) in self.pidxcombo)
        if self.kind == Likelihood.SIGMOID_KIND:
            if strength_total < -40:
                return strength_total
            return -self.weight * math.log(1.0 + math.exp(-strength_total))
        else:
            return self.weight * (Likelihood.LOG_ONE_OVER_SQRT_TWO_PI - 0.5 * strength_total * strength_total)

    def accum_dloglikelihood_dstrength(self, strengths: np.array, accum: np.array):
        strength_total = self.offset + sum(strengths[pidx] * coeff for (pidx,coeff) in self.pidxcombo)
        if self.kind == Likelihood.SIGMOID_KIND:
            dloglikelihood_dstrength_total = self.weight / (1.0 + math.exp(strength_total))
        else:
            dloglikelihood_dstrength_total = -self.weight * strength_total
        for (pidx,coeff) in self.pidxcombo:
            accum[pidx] += coeff * dloglikelihood_dstrength_total

    def accum_d2loglikelihood_dstrength2(self, strengths: np.array, accum: np.array):
        strength_total = self.offset + sum(strengths[pidx] * coeff for (pidx,coeff) in self.pidxcombo)
        if self.kind == Likelihood.SIGMOID_KIND:
            denom = math.exp(-0.5 * strength_total) + math.exp(0.5 * strength_total)
            d2loglikelihood_dstrength_total2 = -self.weight / (denom * denom)
        else:
            d2loglikelihood_dstrength_total2 = -self.weight

        for (pidx1,coeff1) in self.pidxcombo:
            for (pidx2,coeff2) in self.pidxcombo:
                accum[pidx1,pidx2] += coeff1 * coeff2 * d2loglikelihood_dstrength_total2

    def accum_d2loglikelihood_dstrength2_scalepow(self, strengths: np.array, accum: np.array, scale: float, power: float):
        strength_total = self.offset + sum(strengths[pidx] * coeff for (pidx,coeff) in self.pidxcombo)
        if self.kind == Likelihood.SIGMOID_KIND:
            denom = math.exp(-0.5 * strength_total) + math.exp(0.5 * strength_total)
            d2loglikelihood_dstrength_total2 = -self.weight / (denom * denom)
        else:
            d2loglikelihood_dstrength_total2 = -self.weight

        for (pidx1,coeff1) in self.pidxcombo:
            for (pidx2,coeff2) in self.pidxcombo:
                x = coeff1 * coeff2 * d2loglikelihood_dstrength_total2
                accum[pidx1,pidx2] += scale * (x ** power)


def likelihood_of_games(
    p1: Player,
    p2: Player,
    num_games: float,
    p1_won_proportion: float,
    include_first_player_advantage: bool,
) -> List[Likelihood]:
    """Return a list of Likelihood objects representing the result of set of games between p1 and p2

    These Likelihoods can accumulated with any other Likelihoods, and then all passed to compute_elos
    to compute maximum likelihood Elos for all the players.

    NOTE: For performance reasons, you should try to minimize the number of these you create. If p1 and p2 played
    a large number of games, don't call this function once per game. Instead, call it once for all the games
    together (or twice, if you are using include_first_player_advantage=True, separately reporting the stats
    that occured when each side was the first player).

    NOTE: If specifying include_first_player_advantage=True, make sure to add a make_single_player_prior on
    P1_ADVANTAGE_NAME.

    Arguments:
    p1: Name of the first player
    p2: Name of the second player
    num_games: The number of games played
    p1_won_proportion: The proportion of games that p1 won among those games played, counting draws as 0.5.
    include_first_player_advantage: If true, will also make the computation take into account that the first player
      might have an advantage (or a disadvantage!) and it will try to estimate the amount of that advantage.

    Returns:
    List of likelihood objects summarizing the information.
    """

    ret = []
    assert p1_won_proportion >= 0.0 and p1_won_proportion <= 1.0
    assert num_games >= 0.0
    assert p1 != p2

    if num_games > 0.0:
        if not include_first_player_advantage:
            if p1_won_proportion > 0.0:
                ret.append(Likelihood(
                    playercombo={p1: 1.0, p2: -1.0},
                    offset=0.0,
                    weight=p1_won_proportion*num_games,
                    gamecount=p1_won_proportion*num_games,
                    kind=Likelihood.SIGMOID_KIND
                ))
            if p1_won_proportion < 1.0:
                ret.append(Likelihood(
                    playercombo={p2: 1.0, p1: -1.0},
                    offset=0.0,
                    weight=(1.0-p1_won_proportion)*num_games,
                    gamecount=(1.0-p1_won_proportion)*num_games,
                    kind=Likelihood.SIGMOID_KIND
                ))
        else:
            if p1_won_proportion > 0.0:
                ret.append(Likelihood(
                    playercombo={p1: 1.0, p2: -1.0, P1_ADVANTAGE_NAME: 1.0},
                    offset=0.0,
                    weight=p1_won_proportion*num_games,
                    gamecount=p1_won_proportion*num_games,
                    kind=Likelihood.SIGMOID_KIND
                ))
            if p1_won_proportion < 1.0:
                ret.append(Likelihood(
                    playercombo={p2: 1.0, p1: -1.0, P1_ADVANTAGE_NAME: -1.0},
                    offset=0.0,
                    weight=(1.0-p1_won_proportion)*num_games,
                    gamecount=(1.0-p1_won_proportion)*num_games,
                    kind=Likelihood.SIGMOID_KIND
                ))

    return ret

def make_single_player_prior(
    p1: Player,
    num_games: float,
    elo: float,
) -> List[Likelihood]:
    """Return a list of Likelihood objects representing a Bayesian prior that p1 is the specified Elo.

    The strength of the prior that p1 is the specified Elo is as if p1 were observed to have played
    num_games many games against a known player of that Elo and won half and lost half.

    Returns:
    List of likelihood objects summarizing the information.
    """
    ret = []
    assert num_games >= 0.0
    assert np.isfinite(elo)
    if num_games > 0.0:
        ret.append(Likelihood(
            playercombo={p1: 1.0},
            offset=(-elo / ELO_PER_STRENGTH),
            weight=0.5*num_games,
            gamecount=0.5*num_games,
            kind=Likelihood.SIGMOID_KIND
        ))
        ret.append(Likelihood(
            playercombo={p1: -1.0},
            offset=(elo / ELO_PER_STRENGTH),
            weight=0.5*num_games,
            gamecount=0.5*num_games,
            kind=Likelihood.SIGMOID_KIND
        ))
    return ret


def make_sequential_prior(
    players: List[Player],
    num_games: float,
) -> List[Likelihood]:
    """Return a list of Likelihood objects representing a Bayesian prior that each player in the sequence is similar in strength to the previous.

    This can be used, for example, if there were a sequence of changes between different versions, such that each version on average
    is expected to be more similar to its neighbors.

    The strength of the prior between each sequential pair of players is as if they were observed to have played
    num_games many games against each other and won half and lost half.

    Returns:
    List of likelihood objects summarizing the information.
    """
    ret = []
    assert num_games >= 0.0
    assert len(set(players)) == len(players), "players must not contain any duplicates"

    if len(players) < 1:
        return ret

    for i in range(len(players)-1):
        ret.extend(likelihood_of_games(
            p1=players[i],
            p2=players[i+1],
            num_games=num_games,
            p1_won_proportion=0.5,
            include_first_player_advantage=False,
        ))
    return ret


def make_center_elos_prior(
    players: Sequence[Player],
    elo: float,
) -> List[Likelihood]:
    """Return a list of Likelihood objects representing a Bayesian prior that the mean of all player Elos is the specified Elo.

    This prior will have no effect on the relative Elos of the players, unless it fights with another that sets players to
    specific Elos, such as make_single_player_prior. It can simply be used to center all the Elos of the players.

    Returns:
    List of likelihood objects summarizing the information.
    """
    ret = []
    assert np.isfinite(elo)
    assert len(set(players)) == len(players), "players must not contain any duplicates"
    playercombo = { player: 1.0 for player in players }
    ret.append(Likelihood(
        playercombo=playercombo,
        offset=-len(players) * elo / ELO_PER_STRENGTH,
        weight=0.001,
        gamecount=0.0,
        kind=Likelihood.GAUSSIAN_KIND
    ))
    return ret


def compute_elos(
    data: List[Likelihood],
    tolerance: float = 0.001,
    max_iters: int = 1000,
    verbose: bool = False,
) -> EloInfo:
    """Compute maximum-likelihood Elo ratings given the data and any priors.

    NOTE: It is recommend that you specify some sort of prior, even if relatively weak, for numerical stability.
    When you call this function, it is up to you to make sure that the data and priors you have provided result in all
    all Elos of all players being "anchored". For example, this function may crash or fail or return bad values if:

    * There is a player with no data at all, and no prior for that player.
      (since that would mean an Elo for that player cannot be defined).

    * There is a player who has only won and never lost, and there is no prior that restricts that player's rating.
      (since then nothing could stop player's Elo from going to infinity).

    * All players *have* won and lost against other players, but there is nothing that anchors the Elo of the
      population as a whole, such no player having a make_single_player_prior AND there being no make_center_elos_prior.
      (since even if all players Elos are known relative to one another, the Elos to report are undefined - there is nothing
       to say where the Elos should be centered or what value they should start from).

    Examples of things that are normally sufficient to anchor everyone:
    If all players have won and lost against other players, adding a make_center_elos_prior is sufficient.
    If all players have won and lost against other players, AND every player has transitively beat every other player and
    transitively lost to every other player, then adding a make_single_player_prior to one player is sufficient.
    Regardless of whether the players have won or lost, adding a make_single_player_prior to every player is sufficient.
    Regardless of whether the players have won or lost, adding both a make_sequential_prior and a make_center_elos_prior is sufficient.

    NOTE: Even aside from ensuring numeric stability and anchoring as above, it is probably good practice to add some mild prior
    beyond that anyways. If your players are sequential (e.g. a series of different nets), a weak make_sequential_prior could be good.
    If you want to just put all players on equal footing unbiasedly, adding a weak make_single_player_prior to each player that its Elo
    is 0 (or 1000, or whatever) is also good.

    And if you have actual prior beliefs about the players Elos, feel free to add those.

    Arguments:
    data: A single list of all the likelihoods from all your games and priors.
    tolerance: Stop soon after the Elos stop changing by more than this.
    max_iters: Bail out if the optimization takes more than this many iterations.
    verbose: Print out the iteration as it proceeds.

    Returns:
    Elos. Yay!
    """

    players = []
    for d in data:
        players.extend(d.playercombo.keys())
    players = list(set(players))
    players.sort()
    player_to_idx = { player: i for (i,player) in enumerate(players) }

    data = [copy.copy(d) for d in data]
    for d in data:
        d.add_idxs(player_to_idx)

    num_players = len(players)

    def compute_loglikelihood(strengths: np.array) -> float:
        total = 0.0
        for d in data:
            total += d.get_loglikelihood(strengths)
        return total

    # Gauss newton
    def find_ascent_vector(strengths: np.array) -> np.array:
        dloglikelihood_dstrength = np.zeros(num_players,dtype=np.float64)
        d2loglikelihood_dstrength2 = np.zeros((num_players,num_players),dtype=np.float64)

        for d in data:
            d.accum_dloglikelihood_dstrength(strengths, dloglikelihood_dstrength)
            d.accum_d2loglikelihood_dstrength2(strengths, d2loglikelihood_dstrength2)
        ascent = -np.linalg.solve(d2loglikelihood_dstrength2,dloglikelihood_dstrength)
        return ascent

    def line_search_ascend(strengths: np.array, cur_loglikelihood: float) -> Tuple[np.array,float]:
        ascent = find_ascent_vector(strengths)
        # Try up to this many times to find an improvement
        for i in range(30):
            new_strengths = strengths + ascent
            new_loglikelihood = compute_loglikelihood(new_strengths)
            if new_loglikelihood > cur_loglikelihood:
                return (new_strengths, new_loglikelihood)
            # Shrink ascent step and try again
            ascent *= 0.6
        return (strengths,cur_loglikelihood)

    strengths = np.zeros(num_players,dtype=np.float64)
    loglikelihood = compute_loglikelihood(strengths)
    iters_since_big_change = 0
    last_elo_change = None
    for i in range(max_iters):
        #if verbose:
            #print(f"Beginning iteration {i}, cur log likelihood {loglikelihood:4.12f}, last elo change {last_elo_change}")

        (new_strengths, new_loglikelihood) = line_search_ascend(strengths, loglikelihood)
        elodiff = (new_strengths - strengths) * ELO_PER_STRENGTH
        last_elo_change = 0 if len(elodiff) <= 0 else np.max(np.abs(elodiff))

        strengths = new_strengths
        loglikelihood = new_loglikelihood

        iters_since_big_change += 1
        if np.any(elodiff > tolerance):
            iters_since_big_change = 0

        if iters_since_big_change > 3:
            break


    d2loglikelihood_dstrength2 = np.zeros((num_players,num_players),dtype=np.float64)
    for d in data:
        d.accum_d2loglikelihood_dstrength2(strengths, d2loglikelihood_dstrength2)
    strength_precision = -d2loglikelihood_dstrength2
    elo_precision = strength_precision / (ELO_PER_STRENGTH * ELO_PER_STRENGTH)
    elo_covariance = np.linalg.inv(elo_precision)

    sqrt_ess_numerator = np.zeros((num_players,num_players),dtype=np.float64)
    ess_denominator = np.zeros((num_players,num_players),dtype=np.float64)
    for d in data:
        if d.gamecount > 0.0:
            d.accum_d2loglikelihood_dstrength2_scalepow(strengths, sqrt_ess_numerator, scale = 1.0, power=1.0)
            d.accum_d2loglikelihood_dstrength2_scalepow(strengths, ess_denominator, scale = 1.0 / d.gamecount, power=2.0)

    info = EloInfo(
      players = players,
      elo = { player: ELO_PER_STRENGTH * strengths[player_to_idx[player]] for player in players },
      elo_stderr = { player: standard_deviation * math.sqrt(1.0 / elo_precision[player_to_idx[player],player_to_idx[player]]) for player in players },
      elo_covariance = { (p1,p2): elo_covariance[player_to_idx[p1],player_to_idx[p2]] for p1 in players for p2 in players },
      effective_game_count = {
          player: (np.square(sqrt_ess_numerator[player_to_idx[player],player_to_idx[player]]) /
                   ess_denominator[player_to_idx[player],player_to_idx[player]])
          for player in players
      },
    )
    return info

def has_only_factors_of_2_and_3(n: int) -> bool:
    while n > 1:
        if n % 2 == 0:
            n //= 2
        elif n % 3 == 0:
            n //= 3
        else:
            return False
    return True

@dataclass
class GameRecord:
    player1: str
    player2: str
    win: int = 0
    loss: int = 0
    draw: int = 0
    black_moves: int = 0
    white_moves: int = 0
    white_wins: int = 0
    black_visits: int = 0
    white_visits: int = 0
    black_time : float = 0
    white_time : float = 0

    black_moves_each: list = field(default_factory=list)
    white_moves_each: list = field(default_factory=list)
    black_visits_percentage: list = field(default_factory=list)
    white_visits_percentage: list = field(default_factory=list)

class GameResultSummary:

    def __init__(
        self,
        elo_prior_games: int,
        estimate_first_player_advantage: bool,
    ):
        self.results = {}  # dict of { (player1_name, player2_name) : GameRecord }

        self._all_game_files = set()
        self._elo_prior_games = prior_games # number of games for bayesian prior around Elo 0
        self._estimate_first_player_advantage = estimate_first_player_advantage
        self._elo_info = None
        self._game_count = 0

    def add_games_from_file_or_dir(self, input_file_or_dir: str, recursive=False):
        """Add games found in input_file_or_dir into the results. Repeated paths to the same file across multiple calls will be ignored."""
        new_files = self._add_files(input_file_or_dir, recursive)

    def add_game_record(self, record: GameRecord):
        """Add game record to results."""
        if (record.player1, record.player2) not in self.results:
            self.results[(record.player1, record.player2)] = GameRecord(player1=record.player1,player2=record.player2)
        self.results[(record.player1, record.player2)].win += record.win
        self.results[(record.player1, record.player2)].loss += record.loss
        self.results[(record.player1, record.player2)].draw += record.draw
        self.results[(record.player1, record.player2)].black_moves += record.black_moves
        self.results[(record.player1, record.player2)].white_moves += record.white_moves
        self.results[(record.player1, record.player2)].white_wins += record.white_wins
        self.results[(record.player1, record.player2)].black_visits += record.black_visits
        self.results[(record.player1, record.player2)].white_visits += record.white_visits
        self.results[(record.player1, record.player2)].black_time += record.black_time
        self.results[(record.player1, record.player2)].white_time += record.white_time
        self.results[(record.player1, record.player2)].black_moves_each += record.black_moves_each
        self.results[(record.player1, record.player2)].white_moves_each += record.white_moves_each
        self.results[(record.player1, record.player2)].black_visits_percentage += record.black_visits_percentage
        self.results[(record.player1, record.player2)].white_visits_percentage += record.white_visits_percentage
        self._game_count += record.win + record.loss + record.draw

    def clear(self):
        """Clear all data added."""
        self.results = {}
        self._all_game_files = set()
        self._elo_info = None

    def print_game_results(self):
        """Print tables of wins and win percentage."""
        pla_names = set(itertools.chain(*(name_pair for name_pair in self.results.keys())))
        self._print_result_matrix(pla_names)

    def print_elos(self):
        """Print game results and maximum likelihood posterior Elos."""
        elo_info = self._compute_elos_if_needed()
        real_players = [player for player in elo_info.players if player != P1_ADVANTAGE_NAME]
        self._print_result_matrix(real_players)
        print(color_256(226) + "【Elos with +/-",standard_deviation,"approx standard error】" + Style.RESET_ALL)
        print(elo_info)

        try:
            print(color_256(213) + "【Likelihood of superiority between each networks】" + Style.RESET_ALL)
            los_matrix = []
            for player in real_players:
                los_row = []
                for player2 in real_players:
                    los = elo_info.get_approx_likelihood_of_superiority(player,player2)
                    los_row.append(f"{los*100:.1f}")
                los_matrix.append(los_row)
            self._print_matrix(real_players,los_matrix)

            players_count = 0
            surprise_matrix = []
            for pla1 in real_players:
                row = []
                for pla2 in real_players:
                    if (pla1 == pla2):
                        row.append("-")
                        continue
                    else:
                        pla1_pla2 = self.results[(pla1, pla2)] if (pla1, pla2) in self.results else GameRecord(pla1,pla2)
                        pla2_pla1 = self.results[(pla2, pla1)] if (pla2, pla1) in self.results else GameRecord(pla2,pla1)
                        win = pla1_pla2.win + pla2_pla1.loss + 0.5 * (pla1_pla2.draw + pla2_pla1.draw)
                        total = pla1_pla2.win + pla2_pla1.win + pla1_pla2.loss + pla2_pla1.loss + pla1_pla2.draw + pla2_pla1.draw
                        surprise = elo_info.get_log10_odds_surprise_max_likelihood(pla1, pla2, win, total)
                        if total <= 0:
                            row.append("-")
                        else:
                            row.append(f"{surprise:.1f}")
                surprise_matrix.append(row)
        # print("Log10odds surprise matrix given the maximum-likelihood Elos:")
        # print("E.g. +3.0 means a 1:1000 unexpected good performance by row vs column.")
        # print("E.g. -4.0 means a 1:10000 unexpected bad performance by row vs column.")
            for player_number in real_players:
                players_count += 1
            if (players_count > 2):
                print(color_256(112) + "【Rock/paper/scissors or Elo is not a good model for the data】" + Style.RESET_ALL)
                self._print_matrix(real_players,surprise_matrix)
        except:
            pass


        print(color_256(35) + "【Average number/dispersion of move】" + Style.RESET_ALL)
        result_matrix = []
        for pla1 in real_players:
            row = []
            for pla2 in real_players:
                if (pla1 == pla2):
                    row.append("-")
                    continue
                else:
                    pla1_pla2 = self.results[(pla1, pla2)] if (pla1, pla2) in self.results else GameRecord(pla1,pla2)
                    pla2_pla1 = self.results[(pla2, pla1)] if (pla2, pla1) in self.results else GameRecord(pla2,pla1)
                    moves = 2 * (pla1_pla2.black_moves + pla2_pla1.white_moves)
                    moves_each = (pla1_pla2.black_moves_each + pla2_pla1.white_moves_each)
                    total = pla1_pla2.win + pla2_pla1.win + pla1_pla2.loss + pla2_pla1.loss + pla1_pla2.draw + pla2_pla1.draw
                    if total <= 0:
                        row.append("-")
                    else:
                        avg_moves = moves/total
                        row.append(f"{avg_moves:.0f}/{np.var(moves_each):.0f}")
            result_matrix.append(row)
        self._print_matrix(real_players,result_matrix)

        
        Average_WHITE_winrate = 0
        Total_White_wins = 0
        Total_Win = 0
        result_matrix = []
        for pla1 in real_players:
            row = []
            for pla2 in real_players:
                if (pla1 == pla2):
                    row.append("-")
                    continue
                else:
                    pla1_pla2 = self.results[(pla1, pla2)] if (pla1, pla2) in self.results else GameRecord(pla1,pla2)
                    pla2_pla1 = self.results[(pla2, pla1)] if (pla2, pla1) in self.results else GameRecord(pla2,pla1)
                    white_wins = pla2_pla1.white_wins
                    win = pla1_pla2.win + pla2_pla1.loss + 0.5 * (pla1_pla2.draw + pla2_pla1.draw)
                    total = pla1_pla2.win + pla2_pla1.win + pla1_pla2.loss + pla2_pla1.loss + pla1_pla2.draw + pla2_pla1.draw
                    if total <= 0:
                        row.append("-")
                    else:
                        Total_White_wins += white_wins
                        Total_Win += win
                        if ( win == 0 ):
                            row.append("0.0%")
                        else:
                            row.append(f"{white_wins/win*100:.1f}%")
                if (Total_Win != 0):
                    Average_WHITE_winrate = (Total_White_wins/Total_Win*100)
            result_matrix.append(row)
        print(color_256(39) + "【Percentage of White in winning games | Average:" + f"{Average_WHITE_winrate:.2f}" + "】" + Style.RESET_ALL)
        self._print_matrix(real_players,result_matrix)


        result_matrix = []
        have_draw = []
        for pla1 in real_players:
            row = []
            for pla2 in real_players:
                if (pla1 == pla2):
                    row.append("-")
                    continue
                else:
                    pla1_pla2 = self.results[(pla1, pla2)] if (pla1, pla2) in self.results else GameRecord(pla1,pla2)
                    pla2_pla1 = self.results[(pla2, pla1)] if (pla2, pla1) in self.results else GameRecord(pla2,pla1)
                    draw = pla1_pla2.draw + pla2_pla1.draw
                    total = pla1_pla2.win + pla2_pla1.win + pla1_pla2.loss + pla2_pla1.loss + pla1_pla2.draw + pla2_pla1.draw
                    if total <= 0:
                        row.append("-")
                    else:
                        row.append(f"{draw}/{draw/total*100:.1f}%")
                    have_draw.append(draw)
            result_matrix.append(row)
        if any(DRAW != 0 for DRAW in have_draw):
            print(color_256(105) + "【Total of draws and invalid games with other players】" + Style.RESET_ALL)
            self._print_matrix(real_players,result_matrix)
            # print(have_draw)
        else:
            pass

        try:
            result_matrix = []
            for pla1 in real_players:
                row = []
                for pla2 in real_players:
                    if (pla1 == pla2):
                        row.append("-")
                        continue
                    else:
                        pla1_pla2 = self.results[(pla1, pla2)] if (pla1, pla2) in self.results else GameRecord(pla1,pla2)
                        pla2_pla1 = self.results[(pla2, pla1)] if (pla2, pla1) in self.results else GameRecord(pla2,pla1)
                        visits = pla1_pla2.black_visits + pla2_pla1.white_visits
                        visits_percentage = pla1_pla2.black_visits_percentage + pla2_pla1.white_visits_percentage
                        moves = pla1_pla2.black_moves + pla2_pla1.white_moves
                        total = pla1_pla2.win + pla2_pla1.win + pla1_pla2.loss + pla2_pla1.loss + pla1_pla2.draw + pla2_pla1.draw
                        if total <= 0:
                            row.append("-")
                        else:
                            if (visits_percentage):
                                row.append(f"{visits/moves:.0f}/{np.mean(visits_percentage):.2f}")
                            else:
                                row.append(f"{visits/moves:.0f}")
                result_matrix.append(row)
            if (visits_percentage):
                print(color_256(183) + "【Average visits and reuse rate per move】" + Style.RESET_ALL)
                self._print_matrix(real_players,result_matrix)
            #else:
                #print(color_256(183) + "【Visits per move】" + Style.RESET_ALL)
            
            
        except:
            pass

        try:
            results = []
            for pla1 in real_players:
                Total_time = 0
                Total_moves = 0
                for pla2 in real_players:
                    if pla1 != pla2:
                        pla1_pla2 = self.results.get((pla1, pla2), GameRecord(pla1, pla2))
                        pla2_pla1 = self.results.get((pla2, pla1), GameRecord(pla2, pla1))
                        Total_time += pla1_pla2.black_time + pla2_pla1.white_time
                        Total_moves += pla1_pla2.black_moves + pla2_pla1.white_moves
                if Total_moves > 0:
                    avg_time = Total_time / Total_moves
                    results.append((pla1, avg_time))
            max_name_len = max(len(str(name)) for name in real_players) if real_players else 0
            indent = max_name_len // 2
            sorted_results = sorted(results, key=lambda x: x[1])
            print(color_256(224) + "【Average time per move but may not be accurate due to devices】" + Style.RESET_ALL)
            print(f"{"Sort_by_strength":>{indent+10}}{" "*(max_name_len-5)}{"Sort_by_time"}")
            for orig, sorted_ in zip_longest(results, sorted_results, fillvalue=("", 0)):
                left_str = f"{str(orig[0]):>{max_name_len}}: {orig[1]:.2f}" if orig[0] else ""
                right_str = f"{str(sorted_[0])}: {sorted_[1]:.2f}" if sorted_[0] else ""
                orig_len = f"{orig[1]:.2f}"
                # print(len(str(orig_len)))
                print(f"{left_str}{" "*(8 - len(str(orig_len)))}{right_str:>{max_name_len+6}}")
        except:
            pass

        # print(color_256(39) + "\nAverage_WHITE_winrate:" + f"{Average_WHITE_winrate/(Bot):0.2f}" + Style.RESET_ALL)
        # print(f"Used a prior of {self._elo_prior_games} games worth that each player is near Elo 0.")
        print("")

    def get_elos(self) -> EloInfo:
        return self._compute_elos_if_needed()

    def get_game_results(self) -> Dict:
        """Return a dictionary of game results as { (player1_name, player2_name) : GameRecord }

          You can retrieve results by player's name like:
          results[(player1_name, player2_name)].win
          results[(player1_name, player2_name)].loss
          results[(player1_name, player2_name)].draw
        """
        return self.results

    # Functions that can be implemented by subclasses -----------------------------------------------------
    # You can override these methods if you want add_games_from_file_or_dir to work.
    # Otherwise, you can just add game records yourself via add_game_record.
    @abstractmethod
    def is_game_file(self, input_file: str) -> bool:
        """Returns true if this file or directory is one that should have game results in it somewhere"""
        raise NotImplementedError()

    @abstractmethod
    def get_game_records(self, input_file: str) -> List[GameRecord]:
        """Return all game records contained in this file"""
        raise NotImplementedError()

    # Private functions ------------------------------------------------------------------------------------

    def _compute_elos_if_needed(self):
        if self._elo_info is None:
            self._elo_info = self._estimate_elo()
        return self._elo_info

    def _add_files(self, input_file_or_dir, recursive):
        print("Searching and adding files, please wait...")

        if not os.path.exists(input_file_or_dir):
            raise Exception(f"There is no file or directory with name: {input_file_or_dir}")

        files = []
        if os.path.isdir(input_file_or_dir):
            if recursive:
                for (dirpath, dirnames, filenames) in os.walk(input_file_or_dir):
                    files += [os.path.join(dirpath, file) for file in filenames if self.is_game_file(os.path.join(dirpath, file))]
            else:
                files = [os.path.join(input_file_or_dir, file) for file in os.listdir(input_file_or_dir) if self.is_game_file(os.path.join(input_file_or_dir, file))]
        else:
            if self.is_game_file(input_file_or_dir):
                files.append(input_file_or_dir)

        # Remove duplicates
        new_game_files = set(files)
        new_game_files = new_game_files.difference(self._all_game_files)
        self._all_game_files = self._all_game_files.union(new_game_files)
        self._add_new_games_to_result_dict(new_game_files, input_file_or_dir)

        # print(f"Added {len(new_game_files)} new game files from {input_file_or_dir}")
        print(f"Running with following config: standard_deviation = {standard_deviation}, prior_games = {prior_games:.0f}")
        # Let user to input the match config
        if (create == "V" or create == "v"):
            Tester = input("Tested by ")
            Visits = input("Visits: ")
            rule = input("Rule: ")
        elif (create == "P" or create == "p"):
            Tester = input("Tested by ")
            playouts = input("Playouts: ")
            rule = input("Rule: ")
        elif (create == "T" or create == "t"):
            Tester = input("Tested by ")
            Times = input("Times: ")
            rule = input("Rule: ")

    def _add_new_games_to_result_dict(self, new_game_files, source):
        idx = 0
        for game_file in new_game_files:
            records = self.get_game_records(game_file)
            idx += 1
            #if idx % 10 == 0 and has_only_factors_of_2_and_3(idx // 10):
                #print(f"Added {idx}/{len(new_game_files)} game files for {source}")

            for record in records:
                self.add_game_record(record)

    def _estimate_elo(self) -> EloInfo:
        """Estimate and print elo values. This function must be called after adding all games"""
        pla_names = set(itertools.chain(*(name_pair for name_pair in self.results.keys())))
        data = []
        for pla_first in pla_names:
            for pla_second in pla_names:
                if (pla_first == pla_second):
                    continue
                else:
                    if (pla_first, pla_second) not in self.results:
                        continue
                    record = self.results[(pla_first, pla_second)]
                    total = record.win + record.loss + record.draw
                    assert total >= 0
                    if total == 0:
                        continue

                    win = record.win + 0.5 * record.draw
                    winrate = win / total
                    data.extend(likelihood_of_games(
                        pla_first,
                        pla_second,
                        total,
                        winrate,
                        include_first_player_advantage=self._estimate_first_player_advantage
                    ))

        for pla in pla_names:
            data.extend(make_single_player_prior(pla, self._elo_prior_games,0))
        data.extend(make_center_elos_prior(list(pla_names),0)) # Add this in case user put elo_prior_games = 0
        if self._estimate_first_player_advantage:
            data.extend(make_single_player_prior(P1_ADVANTAGE_NAME, (1.0 + self._elo_prior_games) * 2.0, 0))

        info = compute_elos(data, verbose=True)
        return info

    def _print_matrix(self,pla_names,results_matrix):
        per_elt_space = 2
        for sublist in results_matrix:
            for elt in sublist:
                per_elt_space = max(per_elt_space, len(str(elt)))
        per_elt_space += 2

        per_name_space = 1 if len(pla_names) == 0 else max(len(name) for name in pla_names)
        # per_name_space += 1
        if per_name_space > per_elt_space:
            per_elt_space += 1

        row_format = f"{{:>{per_name_space}}}" +  f"{{:>{per_elt_space}}}" * len(results_matrix)
        print(row_format.format("", *[name[:per_elt_space-2] for name in pla_names]))
        for name, row in zip(pla_names, results_matrix):
            print(row_format.format(name, *row))

    def _print_result_matrix(self, pla_names):
        print(f"Total games: {self._game_count}")
        print("Games by player:")
        max_name_len = max(len(str(name)) for name in pla_names) if pla_names else 0
        for pla1 in pla_names:
            total = 0
            for pla2 in pla_names:
                if (pla1 == pla2):
                    continue
                else:
                    pla1_pla2 = self.results[(pla1, pla2)] if (pla1, pla2) in self.results else GameRecord(pla1,pla2)
                    pla2_pla1 = self.results[(pla2, pla1)] if (pla2, pla1) in self.results else GameRecord(pla2,pla1)
                    total += pla1_pla2.win + pla2_pla1.win + pla1_pla2.loss + pla2_pla1.loss + pla1_pla2.draw + pla2_pla1.draw
            print(f"{str(pla1):>{max_name_len}}: {total}")

        print(color_256(117) + "【Wins/Total games by row player against column player】" + Style.RESET_ALL)
        result_matrix = []
        all_total = []
        all_win = []
        for pla1 in pla_names:
            row = []
            for pla2 in pla_names:
                if (pla1 != pla2):
                    pla1_pla2 = self.results[(pla1, pla2)] if (pla1, pla2) in self.results else GameRecord(pla1,pla2)
                    pla2_pla1 = self.results[(pla2, pla1)] if (pla2, pla1) in self.results else GameRecord(pla2,pla1)
                    win = pla1_pla2.win + pla2_pla1.loss + 0.5 * (pla1_pla2.draw + pla2_pla1.draw)
                    total = pla1_pla2.win + pla2_pla1.win + pla1_pla2.loss + pla2_pla1.loss + pla1_pla2.draw + pla2_pla1.draw
                    all_win.append(win)
                    all_total.append(total)
                else:
                    pass
        win_len = 0
        for win_each in all_win:
            if float(win_each) % 1 != 0:
                win_len = 1
                break
        total_len = max(len(str(long)) for long in all_total) if all_total else 0
        # print(total_len)
        
        for pla1 in pla_names:
            row = []
            for pla2 in pla_names:
                if (pla1 == pla2):
                    row.append("-"+" "*total_len)
                    continue
                else:
                    all_total = []
                    pla1_pla2 = self.results[(pla1, pla2)] if (pla1, pla2) in self.results else GameRecord(pla1,pla2)
                    pla2_pla1 = self.results[(pla2, pla1)] if (pla2, pla1) in self.results else GameRecord(pla2,pla1)
                    win = pla1_pla2.win + pla2_pla1.loss + 0.5 * (pla1_pla2.draw + pla2_pla1.draw)
                    total = pla1_pla2.win + pla2_pla1.win + pla1_pla2.loss + pla2_pla1.loss + pla1_pla2.draw + pla2_pla1.draw
                    if (total == 0):
                        row.append("-"+" "*(total_len))
                    else:
                        row.append(f"{win:.{win_len}f}/{total:<{total_len}}")
                    
            result_matrix.append(row)
        self._print_matrix(pla_names,result_matrix)

        result_matrix = []
        all_ELO = []
        for pla1 in pla_names:
            row = []
            for pla2 in pla_names:
                if (pla1 != pla2):
                    pla1_pla2 = self.results[(pla1, pla2)] if (pla1, pla2) in self.results else GameRecord(pla1,pla2)
                    pla2_pla1 = self.results[(pla2, pla1)] if (pla2, pla1) in self.results else GameRecord(pla2,pla1)
                    win = pla1_pla2.win + pla2_pla1.loss + 0.5 * (pla1_pla2.draw + pla2_pla1.draw)
                    total = pla1_pla2.win + pla2_pla1.win + pla1_pla2.loss + pla2_pla1.loss + pla1_pla2.draw + pla2_pla1.draw
                    if (total != 0):
                        winrate = win/total*100
                        if ( winrate <= 1) | (winrate >= 99):
                            ELO = f".INF"
                            pass
                        else:
                            ELO = f"{400*math.log10((winrate)/(100-winrate)):.0f}"
                        all_ELO.append(ELO)
                    # print(win,total,winrate,ELO)
        try:
            total_len_ELO = max(len(str(long_ELO)) for long_ELO in all_ELO)
        except:
            total_len_ELO = 3
        # print(all_ELO,total_len_ELO)

        for pla1 in pla_names:
            row = []
            for pla2 in pla_names:
                if (pla1 == pla2):
                    row.append("-"+" "*(total_len_ELO))
                    continue
                else:
                    pla1_pla2 = self.results[(pla1, pla2)] if (pla1, pla2) in self.results else GameRecord(pla1,pla2)
                    pla2_pla1 = self.results[(pla2, pla1)] if (pla2, pla1) in self.results else GameRecord(pla2,pla1)
                    win = pla1_pla2.win + pla2_pla1.loss + 0.5 * (pla1_pla2.draw + pla2_pla1.draw)
                    total = pla1_pla2.win + pla2_pla1.win + pla1_pla2.loss + pla2_pla1.loss + pla1_pla2.draw + pla2_pla1.draw
                    if (total > 2):
                        winrate = win/total*100
                        if ( winrate <= 0.2 ):
                            row.append(f"{winrate:.1f}%/-INF"+" "*(total_len_ELO-4))
                        elif ( winrate >= 99.8 ):
                            row.append(f"{winrate:.1f}%/+INF"+" "*(total_len_ELO-4))
                        elif ( winrate == 50 ):
                            row.append(f"{winrate:.1f}%/"+" "*(total_len_ELO-1)+"0")
                        else:
                            sign = "+" if winrate >= 50 else "-"
                            fiex = 1 if winrate >= 50 else -1
                            elo = f"{fiex*400*math.log10((winrate)/(100-winrate)):.0f}"
                            row.append(f"{winrate:.1f}%/{sign}{elo:>{total_len_ELO-1}}")
                    else:
                        row.append("-"+" "*(total_len_ELO))
            result_matrix.append(row)
        print(color_256(214) + "【Winrate/Elo by row player against column player】" + Style.RESET_ALL)
        self._print_matrix(pla_names,result_matrix)

    def new_method(self):
        pla_names = list(self.elo.keys())
        return pla_names

    def new_method(self):
        return list(self.elo.keys())

# Testing code
# if __name__ == "__main__":
    # Example 1
    # data = []
    # data.extend(likelihood_of_games("Alice","Bob", 18, 12/18, False))
    # data.extend(likelihood_of_games("Bob","Carol", 18, 12/18, False))
    # data.extend(likelihood_of_games("Carol","Dan", 36, 12/18, False))
    # data.extend(likelihood_of_games("Dan","Eve", 48, 40/48, False))
    # data.extend(make_center_elos_prior(["Alice","Bob","Carol","Dan","Eve"],0))
    # info = compute_elos(data,verbose=True)

    # for player in info.players:
    #     for player2 in info.players:
    #         print(info.get_approx_likelihood_of_superiority(player,player2),end=" ")
    #     print()

    # Example 2
    # summary = GameResultSummary(
        # elo_prior_games = 5.0,
        # estimate_first_player_advantage = False,
    # )
    # summary.add_game_record(GameRecord("Alice","Bob",win=12,loss=6,draw=0))
    # summary.add_game_record(GameRecord("Bob","Carol",win=12,loss=6,draw=0))
    # summary.add_game_record(GameRecord("Carol","Dan",win=36,loss=18,draw=0))
    # summary.add_game_record(GameRecord("Dan","Eve",win=48,loss=24,draw=0))
    # summary.print_elos()

class GoGameResultSummary(GameResultSummary):

    def __init__(
        self,
        elo_prior_games: int,
        estimate_first_player_advantage: bool,
    ):
        super().__init__(elo_prior_games, estimate_first_player_advantage)
        self._should_warn_handicap_komi = False

    # @override
    def print_elos(self):
        super().print_elos()
        if self._should_warn_handicap_komi:
            print('\033[91m'+"WARNING: There are handicap games or games with komi < 5.5 or komi > 7.5, these games may not be fair?\n"+ '\x1b[0m')

    # @override
    def is_game_file(self, input_file: str) -> bool:
        lower = input_file.lower()
        return input_file.endswith(".sgf") or input_file.endswith(".sgfs")

    # @override
    def get_game_records(self, input_file: str) -> List[GameRecord]:
        if input_file.lower().endswith(".sgfs"):
            with open(input_file, "rb") as f:
                sgfs_strings = f.readlines()

            records = []
            for sgf in sgfs_strings:
                record = self.sgf_string_to_game_record(sgf, input_file)
                if record is not None:
                    records.append(record)
            return records
        else:
            with open(input_file, "rb") as f:
                sgf = f.read()

            records = []
            record = self.sgf_string_to_game_record(sgf, input_file)
            if record is not None:
                records.append(record)
            return records

    def sgf_string_to_game_record(self, sgf_string, debug_source = None) -> GameRecord:
        try:
            # sgfmill for some reason can't handle rectangular boards, even though it's part of the SGF spec.
            # So lie and say that they're square, so that we can load them.
            sgf_string = re.sub(r'SZ\[(\d+):\d+\]', r'SZ[\1]', sgf_string.decode("utf-8"))
            sgf_string = sgf_string.encode("utf-8")

            game = sgf.Sgf_game.from_bytes(sgf_string)
            winner = game.get_winner()
            nodes = game.get_main_sequence()
        except ValueError:
            print('\033[91m'+f"A sgf string is damaged in {debug_source}, and its record has been skipped!"+ '\x1b[0m')
        pla_black = game.get_player_name('b')
        pla_white = game.get_player_name('w')
        if (game.get_handicap() is not None) or game.get_komi() < 5.5 or game.get_komi() > 7.5:
            self._should_warn_handicap_komi = True

        game_record = GameRecord(player1=pla_black,player2=pla_white)

        # Calculate how many move in a game
        black_visits_each = []
        white_visits_each = []
        total_moves = 0
        breakpoint = 0
        visits = 0
        visit_pattern = re.compile(r'v=(\d+)')
        bTimeUsed_pattern = re.compile(r'bTimeUsed=(\d+\.?\d*)')
        wTimeUsed_pattern = re.compile(r'wTimeUsed=(\d+\.?\d*)')
        for node in nodes:
            move_result = node.get_move()
            color, move = move_result
            if ((move is None) | (color is None)) :
                breakpoint += 1
            if (total_moves !=0 & breakpoint < 2):
                if (total_moves %2 == 1):
                    game_record.black_moves += 1
                elif (total_moves %2 == 0):
                    game_record.white_moves += 1

            # ====== For debug ======
            # print(total_moves)
            # print(color, move)
            # print(game_record.black_moves, game_record.white_moves)

            
            try:
                comment = node.get("C")
                Visit = visit_pattern.search(comment)
                if Visit:
                    visits = int(Visit.group(1))
                    if (total_moves %2 == 1):
                        game_record.black_visits += visits
                        black_visits_each.append(visits)
                    elif (total_moves %2 == 0):
                        game_record.white_visits += visits
                        white_visits_each.append(visits)
            except:
                pass
            total_moves = total_moves + 1
            # ====== For debug ======
            # print(comment)
            # print(visits)
            # print(game_record.black_visits, game_record.white_visits)

            BTime = bTimeUsed_pattern.search(comment)
            WTime = wTimeUsed_pattern.search(comment)
            if BTime:
                black_time = float(BTime.group(1))
                game_record.black_time = black_time
                # print(game_record.black_time)
            if WTime:
                white_time = float(WTime.group(1))
                game_record.white_time = white_time
                # print(game_record.white_time)

        game_record.black_moves_each.append(game_record.black_moves)
        game_record.white_moves_each.append(game_record.white_moves)
        # print(game_record.black_moves_each,game_record.white_moves_each)
        # print(black_visits_each)
        # print(white_visits_each)
        
        if (black_visits_each[1]>=1.06*black_visits_each[0]):
            loop = 1
            all_gap =[]
            all_percentage = []
            for b in black_visits_each:
                try:
                    loop_each = black_visits_each[loop]
                    gap_each = (black_visits_each[loop] - black_visits_each[0])
                    # percentage_each = round(100 * gap_each / black_visits_each[loop - 1], 3)
                    # print(f"{loop_each} {gap_each} {percentage_each}") 
                    # all_percentage.append(percentage_each)
                    all_gap.append(gap_each)
                    loop += 1
                except:
                    pass
            # median_all_percentage = float(np.median(all_percentage))
            game_record.black_visits_percentage.append(100 * sum(all_gap)/sum(black_visits_each))

            # print(all_percentage)
            # print(round(np.mean(all_percentage), 3))
            # print(np.median(all_percentage))
            # print(sum(black_visits_each),sum(all_gap),sum(all_gap)/sum(black_visits_each))
            # print(game_record.black_visits_percentage)
        

        if (white_visits_each[1]>=1.06*white_visits_each[0]):
            loop = 1
            all_gap =[]
            all_percentage = []
            for w in white_visits_each:
                try:
                    loop_each = white_visits_each[loop]
                    gap_each = (white_visits_each[loop] - white_visits_each[0])
                    # percentage_each = round(100 * gap_each / white_visits_each[loop - 1], 3)
                    # print(f"{loop_each} {gap_each} {percentage_each}") 
                    # all_percentage.append(percentage_each)
                    all_gap.append(gap_each)
                    loop += 1
                except:
                    pass
            # median_all_percentage = float(np.median(all_percentage))
            game_record.white_visits_percentage.append(100 * sum(all_gap)/sum(white_visits_each))
            
            # print(all_percentage)
            # print(round(np.mean(all_percentage), 3))
            # print(np.median(all_percentage))
            # print(sum(white_visits_each),sum(all_gap),sum(all_gap)/sum(white_visits_each))
            # print(game_record.white_visits_percentage"\n")      

        # print(game_record.black_moves)

        if (winner == 'b'):
            game_record.win += 1
        elif (winner == 'w'):
            game_record.loss += 1
            game_record.white_wins += 1
        else:
            game_record.draw += 1
            game_record.white_wins += 0.5

        # print(f"B: {game_record.black_moves}, W: {game_record.white_moves}")
        # print(game_record)

        return game_record
    

if __name__ == "__main__":
    description = """
    Summarize SGF/SGFs files and estimate Bayes Elo score for each of the player.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "input-files-or-dirs",
        help="sgf/sgfs files or directories of them",
        nargs="+",
    )
    parser.add_argument(
        "-recursive",
        help="Recursively search subdirectories of input directories",
        required=False,
        action="store_true",
    )
    parser.add_argument(
        "-prior",
        help="Prior for Bayes Elo calculation, using input as the prior number of games to stabilize the results",
        required=False,
        type=int,
        default=0,
    )
    parser.add_argument(
        "-standard",
        help="The numbers of standard deviation",
        required=False,
        type=float,
        default=2,
    )
    parser.add_argument(
        "-first-advantage",
        help="Attempt to estimate first player advantage instead of assuming fair game",
        required=False,
        action="store_true",
    )
    parser.add_argument(
        "-create",
        help="Output match configs (v for visits, p for playouts, t for times)",
        required=False,
        type=str,
        choices=["V", "P", "T","v", "p", "t"]
    )
    parser.add_argument(
        "-color",
        help="Show color in each caption",
        required=False,
        action="store_true",
    )
    args = vars(parser.parse_args())

    #print(args) 

    input_files_or_dirs = args["input-files-or-dirs"]

    recursive = args["recursive"]

    if 'Standard_Deviation' in globals():
        standard_deviation = globals()['Standard_Deviation']
    else:
        standard_deviation = args["standard"]

    if 'Prior_Games' in globals():
        prior_games = globals()['Prior_Games']
    else:
        prior_games = args["prior"]

    elo_prior_games = prior_games

    estimate_first_player_advantage = args["first_advantage"]

    create = args["create"]

    Color = args["color"]

    game_result_summary = GoGameResultSummary(
        elo_prior_games=elo_prior_games,
        estimate_first_player_advantage=estimate_first_player_advantage,
    )
    for input_file_or_dir in input_files_or_dirs:
        game_result_summary.add_games_from_file_or_dir(input_file_or_dir, recursive=recursive)

    game_result_summary.print_elos()
