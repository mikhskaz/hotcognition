from datetime import timedelta
import math
import logging

from pyClarion import (Event, Agent, Priority, Input, Pool, Choice,
                       ChunkStore, BaseLevel, Family, NumDict, Atoms, Atom, Chunk, ks_crawl)


import random
from itertools import product

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import statistics as stats

# Constants
NUM_CHUNKS = 30
CONSONANTS = "BCDFGHJKLMNPQRSTVWXYZ"
VOWELS = "AEIOU"
NUMBERS = "0123456789"
delta_t = 3  # Time between each event


class Character(Atoms):
    A: Atom
    B: Atom
    C: Atom
    D: Atom
    E: Atom
    F: Atom
    G: Atom
    H: Atom
    I: Atom
    J: Atom
    K: Atom
    L: Atom
    M: Atom
    N: Atom
    O: Atom
    P: Atom
    Q: Atom
    R: Atom
    S: Atom
    T: Atom
    U: Atom
    V: Atom
    W: Atom
    X: Atom
    Y: Atom
    Z: Atom
    _0: Atom
    _1: Atom
    _2: Atom
    _3: Atom
    _4: Atom
    _5: Atom
    _6: Atom
    _7: Atom
    _8: Atom
    _9: Atom


class IO(Atoms):
    CVC1: Atom
    CVC2: Atom
    CVC3: Atom
    NUM1: Atom
    NUM2: Atom


class PairedAssoc(Family):
    char: Character
    io: IO

# TODO: MCMC for parameter estimation


logger = logging.getLogger("pyClarion.system")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.FileHandler("log.log"))


class Participant(Agent):
    d: PairedAssoc
    input: Input
    store: ChunkStore
    blas: BaseLevel
    pool: Pool
    choice: Choice

    def __init__(self, name: str, scale, decay, sd, blw) -> None:
        p = Family()
        e = Family()
        d = PairedAssoc()
        super().__init__(name, p=p, e=e, d=d)
        self.d = d
        with self:
            self.input = Input("input", (d, d))
            self.store = ChunkStore("store", d, d, d)
            self.blas = BaseLevel(
                "blas", p, e, self.store.chunks, sc=scale, de=decay, unit=timedelta(milliseconds=1))
            self.pool = Pool("pool", p, self.store.chunks, func=NumDict.sum)
            self.choice = Choice("choice", p, self.store.chunks, sd=sd)
        self.store.bu.input = self.input.main
        self.blas.input = self.choice.main
        self.pool["store.bu"] = (
            self.store.bu.main,
            lambda d: d.shift(x=1).scale(x=0.5).logit())
        self.pool["blas"] = (
            self.blas.main,
            lambda d: d.bound_min(x=1e-8).log().with_default(c=0.0))
        self.choice.input = self.pool.main
        with self.pool.params[0].mutable():
            self.pool.params[0][~self.pool.p["blas"]] = blw
        self.blas.ignore.add(~self.store.chunks.nil)

    def resolve(self, event: Event) -> None:
        if event.source == self.store.bu.update:
            self.blas.update()
        if event.source == self.blas.update:
            self.choice.trigger()

    def start_trial(self,
                    dt: timedelta,
                    priority: Priority = Priority.PROPAGATION
                    ) -> None:
        self.system.schedule(self.start_trial, dt=dt, priority=priority)

    def finish_trial(self,
                     dt: timedelta,
                     priority: Priority = Priority.PROPAGATION
                     ) -> None:
        self.system.schedule(self.finish_trial, dt=dt, priority=priority)


# def simulate(*params: tuple[float, float, float, float]):
#     results = {
#         "pid": [],  # Which participant
#         "trial": [],
#         "cvc": [],
#         "num": [],
#         "scale": [],
#         "decay": [],
#         "sd": [],
#         "blw": [],
#         "interval": [],
#         "correct": [],
#         "rt": [],
#     }

#     for j, (scale, decay, sd, blw) in enumerate(params):
#         p = Family()
#         e = Family()
#         d = PairedAssoc()
#         char = d.char
#         io = d.io

#         # Create Chunks, Random Consonant-Vowel-Consonant and Number pairs
#         items = []
#         visited = set()
#         for _ in range(NUM_CHUNKS):
#             c1 = random.choice(CONSONANTS)
#             c2 = random.choice(VOWELS)
#             c3 = random.choice(CONSONANTS)
#             num1 = random.choice(NUMBERS)
#             num2 = random.choice(NUMBERS)

#             while c1 + c2 + c3 in visited or num1 + num2 in visited:
#                 c1 = random.choice(CONSONANTS)
#                 c2 = random.choice(VOWELS)
#                 c3 = random.choice(CONSONANTS)
#                 num1 = random.choice(NUMBERS)
#                 num2 = random.choice(NUMBERS)

#             visited.add(c1 + c2 + c3)
#             visited.add(num1 + num2)
#             items.append((
#                 # Study Chunk
#                 + io.CVC1 ** char[c1]
#                 + io.CVC2 ** char[c2]
#                 + io.CVC3 ** char[c3]
#                 + io.NUM1 ** char["_" + num1]
#                 + io.NUM2 ** char["_" + num2],

#                 # Test Chunk
#                 + io.CVC1 ** char[c1]
#                 + io.CVC2 ** char[c2]
#                 + io.CVC3 ** char[c3],

#                 # CVC and Number in string form
#                 f"{c1}{c2}{c3}",
#                 f"{num1}{num2}"
#             ))

#         experiment_sequence = list(range(NUM_CHUNKS)) * 2
#         random.shuffle(experiment_sequence)
#         accessed = {}

#         logger.info(experiment_sequence)

#         with Agent("agent", p=p, d=d, e=e) as agent:
#             chunks = ChunkStore("chunks", c=d, d=d, v=d)
#             base_levels = BaseLevel(
#                 "bla",
#                 p,
#                 e,
#                 chunks.chunks,
#                 sc=scale,  # Set the scale
#                 de=decay  # Set the decay
#             )
#             input_ = Input("input", (d, d))
#             choice = Choice("choice", p, chunks.chunks, sd=sd)

#             pool = Pool("pool", p, chunks.chunks, func=Pool.Heckerman)
#             chunks.bu.input = input_.main
#             pool["bu"] = chunks.bu.main
#             pool["bla"] = base_levels.main

#             with pool.params[0].mutable():
#                 pool.params[0][~pool.p["bu"]] = blw
#             choice.input = pool.main

#             # TODO: Define response time per agent based on real life data

#         logger.info(f"Successfully created Agent {j}:")

#         correct_recall = 0
#         for trial, i in enumerate(experiment_sequence):
#             time = agent.system.clock.time
#             if accessed.setdefault(i, time) == time:
#                 # Study
#                 chunks.compile(items[i][0])  # * if unpacking list
#                 agent.breakpoint(dt=timedelta(seconds=delta_t))
#                 while agent.system.queue:  # Processing of adding new chunk
#                     event = agent.system.advance()
#                 continue

#             # Recall
#             interval = (time - accessed[i]) / timedelta(seconds=1)
#             input_.send(items[i][1], dt=timedelta(seconds=delta_t))

#             # TODO: Account for reaction time

#             while agent.system.queue:
#                 event = agent.system.advance()
#                 if event.source == chunks.bu.update:
#                     base_levels.update()

#             choice.select()
#             while agent.system.queue:  # Clear the queue
#                 agent.system.advance()

#             selected = choice.poll()[~chunks.chunks]

#             # Check if selected chunk is correct
#             correct = ~items[i][0] == selected
#             correct_recall += 1 if correct else 0

#             # TODO: Average time for two key strokes

#             rt = math.exp(-choice.sample[0][selected])
#             message = "\n    ".join([
#                 f"Retrieved {selected} in {rt} s; correct={correct}",
#                 str(base_levels.main[0]).replace("\n", "\n    "),
#                 str(choice.input[0]).replace("\n", "\n    "),
#                 str(choice.sample[0]).replace("\n", "\n    ")])
#             agent.system.logger.info(message)

#             results["pid"].append(j)
#             results["trial"].append(trial)
#             results["cvc"].append(items[i][2])
#             results["num"].append(items[i][3])
#             results["scale"].append(scale)
#             results["decay"].append(decay)
#             results["sd"].append(sd)
#             results["blw"].append(blw)
#             results["interval"].append(interval)
#             results["correct"].append(correct)
#             results["rt"].append(rt)

#         print(
#             f"Agent {j} recalled {correct_recall}/{NUM_CHUNKS} \
#                 times successfuly.")

#     return results

def init_stimuli(d: PairedAssoc, l: list[str]) -> list[tuple[Chunk, Chunk]]:
    io, char = d.io, d.char
    return [
        (s ^
         (query :=
          s[:3] ^
          + io.CVC1 ** char[s[0]]
          + io.CVC2 ** char[s[1]]
          + io.CVC3 ** char[s[2]])
         + io.NUM1 ** char[f"_{s[4]}"]
         + io.NUM2 ** char[f"_{s[5]}"],
         query)
        for s in l]


def generate_cvc_list(n):
    items = []
    visited = set()
    for _ in range(NUM_CHUNKS):
        c1 = random.choice(CONSONANTS)
        c2 = random.choice(VOWELS)
        c3 = random.choice(CONSONANTS)
        num1 = random.choice(NUMBERS)
        num2 = random.choice(NUMBERS)

        while c1 + c2 + c3 in visited or num1 + num2 in visited:
            c1 = random.choice(CONSONANTS)
            c2 = random.choice(VOWELS)
            c3 = random.choice(CONSONANTS)
            num1 = random.choice(NUMBERS)
            num2 = random.choice(NUMBERS)

        visited.add(c1 + c2 + c3)
        visited.add(num1 + num2)
        items.append(f"{c1}{c2}{c3}_{num1}{num2}")
    return items


def simulate(scale, decay, sd, blw):
    participant = Participant("participant", scale, decay, sd, blw)
    stimuli = init_stimuli(
        participant.d, generate_cvc_list(18))
    indices = list(range(len(stimuli))) * 2
    random.shuffle(indices)
    presentations = {}
    trial = 0
    results = {
        "trial": [],
        "stim": [],
        "time": [],
        "delta": [],
        "response": [],
        "correct": [],
        "strength": [],
        "rt": [],
        "decay": [],
        "scale": [],
    }
    participant.start_trial(timedelta())
    while participant.system.queue:
        event = participant.system.advance()
        # print(event.describe())
        if event.source == participant.start_trial:
            i = indices[trial]
            study, test = stimuli[i]
            if i not in presentations:
                participant.store.compile(study)
            else:
                participant.input.send(test)
            participant.finish_trial(timedelta(seconds=3))
        if event.source == participant.finish_trial:
            i = indices[trial]
            if i in presentations:
                target, _ = stimuli[i]
                time = participant.system.clock.time / timedelta(seconds=1)
                response_key = participant.choice.poll()[
                    ~participant.store.chunks]
                response_chunk = ks_crawl(
                    participant.system.root, response_key)
                results["trial"].append(trial)
                results["stim"].append(target._name_)
                results["time"].append(time)
                results["delta"].append(trial - presentations[i])
                results["response"].append(
                    response_chunk._name_)  # type: ignore
                results["correct"].append(response_chunk == target)
                results["strength"].append(
                    participant.choice.sample[0][response_key])
                results["rt"].append(
                    math.exp(-participant.choice.sample[0][response_key]))
                results["decay"].append(decay)
                results["scale"].append(scale)
            else:
                presentations[i] = trial
            if trial < 2 * len(stimuli) - 1:
                participant.start_trial(timedelta(seconds=2))
                trial += 1
    print(f"Completed simulation with {np.mean(results["correct"])}.")
    return results


# ======= PLOTTING THE DATA =======

def heatmap(df):
    heatmap_data = df.copy()
    heatmap_data['arousal_bin'] = pd.cut(heatmap_data['arousal'], bins=10)
    heatmap_data['valence_bin'] = pd.cut(heatmap_data['valence'], bins=10)

    heatmap_pivot = heatmap_data.pivot_table(
        index='arousal_bin', columns='valence_bin', values='correct', aggfunc='mean'
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_pivot, cmap='viridis', annot=True, fmt=".2f")
    plt.title('Retention Probability by Arousal and Valence')
    plt.xlabel('Valence')
    plt.ylabel('Arousal')
    plt.tight_layout()
    plt.show()


def scatter_plot(df):
    plt.figure(figsize=(8, 6))

    # Aggregate retention by decay
    scatter_data = df.groupby('decay')['correct'].mean().reset_index()

    plt.scatter(scatter_data['decay'], scatter_data['correct'], alpha=0.7)
    m, b = np.polyfit(scatter_data['decay'], scatter_data['correct'], 1)
    plt.plot(scatter_data['decay'], m*scatter_data['decay'] +
             b, color='red', linewidth=2, label='Trendline')

    plt.xlabel('Decay (1 - Arousal)')
    plt.ylabel('Mean Observed Retention')
    plt.title('Decay Parameter vs. Observed Retention')
    plt.legend()
    plt.tight_layout()
    plt.show()


def mean_delta_plot(df):
    plt.figure(figsize=(8, 6))

    delta_data = df.groupby(['delta', 'decay'])['correct'].agg(
        ['mean', 'std', 'sem']).reset_index()
    print(delta_data.head(5))
    cmap = {v: i for i, v in enumerate(delta_data['decay'].unique())}
    grp = delta_data.groupby('decay')
    for decay, grpd in grp:
        m, b = np.polyfit(grpd['delta'], grpd['mean'], 1)
        plt.errorbar(grpd['delta'], grpd['mean'],
                     yerr=grpd['sem'], fmt='o', color=f"C{cmap[decay]}",
                     capsize=5, alpha=.3)
        plt.plot(grpd['delta'], [m*x + b for x in grpd["delta"]],
                 color=f"C{cmap[decay]}", linewidth=2, label=f'decay={decay}')
    plt.xlabel('Delta')
    plt.ylabel('Mean Observed Retention')
    plt.title('Delta vs. Observed Retention')
    plt.legend()
    plt.tight_layout()
    plt.show()


def two_way_plot(df, key='scale'):
    plt.figure(figsize=(8, 6))

    # Aggregate retention by decay
    two_way_data = df.groupby(['scale', 'decay'])[
        'correct'].mean().reset_index()

    for i in two_way_data[key].unique():
        i_data = two_way_data[two_way_data[key] == i]
        plt.plot(i_data[key], df['rt'],
                 label=f'{key} = {i}')

    plt.xlabel(f'{key}')
    plt.ylabel('Mean Observed Retention')
    plt.title('Decay Parameter vs. Observed Retention')
    plt.legend()
    plt.tight_layout()
    plt.show()


def violin(df):
    conditions = []
    for _, row in df.iterrows():
        if row['arousal'] > 0.5:
            conditions.append('High Arousal')
        elif row['valence'] < 1:
            conditions.append('Neutral')
        elif row['valence'] >= 3:
            conditions.append('Extreme Valence')
        else:
            conditions.append('Moderate Valence')

    df['condition'] = conditions

    plt.figure(figsize=(10, 6))
    sns.violinplot(x='condition', y='rt', data=df, inner='quartile')
    plt.title('Retrieval Latency Distribution by Emotional Condition')
    plt.ylabel('Retrieval Latency (s)')
    plt.xlabel('Emotional Condition')
    plt.tight_layout()
    plt.show()


# ======= MAIN =======


if __name__ == "__main__":
    # scales = [1, 2, 4, 8, 16, 32, 64]
    # decays = [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9]
    decays = [0.45, 0.55] * 25
    scales = [1500]
    # decays = [0.5]
    sds = [0.5]
    blw = [1e-2]

    results = []
    for scale, decay, sd, blw in product(scales, decays, sds, blw):
        results.append(simulate(scale, decay, sd, blw))

    results = pd.concat([pd.DataFrame(r) for r in results])

    df = pd.DataFrame(results)
    df['arousal'] = 1 - df['decay']
    df['valence'] = np.sqrt(df['scale'] - 1)  # FORGET negatives

    # heatmap(df)
    # scatter_plot(df)
    # two_way_plot(df)
    # violin(df)

    mean_delta_plot(df)

    '''
    TODO: Play with parameters
    TODO: Latency & Correct vs parameters
    TODO: 2-way plots
    TODO: Research Hierarchical Regression (Tradeoff complexity vs computation/accuracy)
    TODO: Include typing time / recall time in interval
    TODO: Correct ~ Interval X Condition (* Low (0.4) vs Neutral (0.5) \
        Decay, High (2+) vs Neutral (1) Scale)
    TODO: Scatter (Beta, Normal) 
    '''

# Can we reliably recover the scale parameter for each participant?
