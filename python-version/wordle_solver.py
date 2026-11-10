#!/usr/bin/env python3
"""
Wordle Solver
=============
Run:            python3 wordle_solver.py
Hard mode on:   python3 wordle_solver.py --hard        (or type `hard` at the prompt)

First-time setup (build the dictionary files from word lists you already have):

    python3 wordle_solver.py --build-dict --answers small_list.txt \
            --guesses big_list_1.txt big_list_2.txt [--extra huge_list.txt]

  * --answers  : your SMALLEST / most common-word list (ideally ~2,300 words).
                 Becomes answers.txt and is weighted heavily as possible solutions.
  * --guesses  : any other lists (they may overlap; duplicates are removed).
                 Combined with the answers into guesses.txt (the ~17k accepted words).

  * --extra    : optional huge/messy lists. Words here are accepted when you type
                 them but are never recommended or counted as candidates.

Optional: past_answers.txt (one word per line) - previous Wordle answers, which
Wordle doesn't repeat. They are demoted to guess-only words.

Colour codes when entering results:  g = green, y = yellow, x = grey.
Standard library only.
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
ANSWERS_FILE = os.path.join(HERE, "answers.txt")
GUESSES_FILE = os.path.join(HERE, "guesses.txt")
PAST_FILE = os.path.join(HERE, "past_answers.txt")
EXTRA_FILE = os.path.join(HERE, "extra_guesses.txt")  # accepted when typed, never recommended
CACHE_FILE = os.path.join(HERE, "opener_cache.json")
EXCLUDE_FILE = os.path.join(HERE, "excluded_words.txt")  # words banned with `ban WORD`

ANSWER_WEIGHT = 1.0    # words from the small (answers) dictionary
GUESS_WEIGHT = 0.02    # guess-only words: almost never the real answer
CANDIDATE_BONUS = 0.97  # test words that could also be the answer get a small edge


# --------------------------------------------------------------------------
# Dictionary handling
# --------------------------------------------------------------------------
def read_words(path):
    """Return the set of 5-letter alphabetic words found in a text file."""
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return {w.lower() for w in re.findall(r"(?<![A-Za-z])[A-Za-z]{5}(?![A-Za-z])", text)}


def build_dictionary(answer_files, guess_files, extra_files=()):
    """Merge source files into answers.txt, guesses.txt (and optional extra_guesses.txt)."""
    answers = set()
    for p in answer_files:
        answers |= read_words(p)
    guesses = set(answers)
    for p in guess_files:
        guesses |= read_words(p)
    with open(ANSWERS_FILE, "w") as f:
        f.write("\n".join(sorted(answers)) + "\n")
    with open(GUESSES_FILE, "w") as f:
        f.write("\n".join(sorted(guesses)) + "\n")
    if extra_files:
        extra = set()
        for p in extra_files:
            extra |= read_words(p)
        extra -= guesses
        with open(EXTRA_FILE, "w") as f:
            f.write("\n".join(sorted(extra)) + "\n")
        print(f"Wrote {len(extra):,} words to extra_guesses.txt")
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
    print(f"Wrote {len(answers):,} words to answers.txt")
    print(f"Wrote {len(guesses):,} words to guesses.txt")
    if len(answers) > 5000:
        print("Note: answers.txt is large. The weighting works best when it is your "
              "smallest, most common-word list (about 2,300 words).")


def load_dictionary():
    """Return (all_words, weight_map, answer_list, extra_words)."""
    if not os.path.exists(ANSWERS_FILE) or not os.path.exists(GUESSES_FILE):
        print("Dictionary files not found (answers.txt and guesses.txt).\n"
              "Build them from your word lists with:\n"
              "  python3 wordle_solver.py --build-dict --answers small_list.txt "
              "--guesses big_list.txt [more_lists.txt ...]")
        sys.exit(1)
    answers = read_words(ANSWERS_FILE)
    guesses = read_words(GUESSES_FILE)
    past = read_words(PAST_FILE) if os.path.exists(PAST_FILE) else set()
    banned = read_words(EXCLUDE_FILE) if os.path.exists(EXCLUDE_FILE) else set()
    answers -= banned
    live_answers = answers - past
    all_words = sorted((answers | guesses | past) - banned)
    extra = (read_words(EXTRA_FILE) if os.path.exists(EXTRA_FILE) else set()) - banned - set(all_words)
    weight = {w: (ANSWER_WEIGHT if w in live_answers else GUESS_WEIGHT) for w in all_words}
    return all_words, weight, sorted(live_answers), extra


# --------------------------------------------------------------------------
# Core logic
# --------------------------------------------------------------------------
def feedback(guess, answer):
    """Wordle's colour pattern (string of g/y/x) for guess scored against answer.
    Handles repeated letters exactly like Wordle does."""
    pat = ["x"] * 5
    remaining = {}
    for i in range(5):
        if guess[i] == answer[i]:
            pat[i] = "g"
        else:
            remaining[answer[i]] = remaining.get(answer[i], 0) + 1
    for i in range(5):
        if pat[i] == "x":
            c = guess[i]
            if remaining.get(c, 0) > 0:
                pat[i] = "y"
                remaining[c] -= 1
    return "".join(pat)


def filter_rounds(rounds, words):
    """Apply rounds in order. Returns (conflict_index or None, candidates).
    A word survives only if scoring each guess against it reproduces the entered pattern."""
    cands = words
    for k, (g, p) in enumerate(rounds):
        cands = [w for w in cands if feedback(g, w) == p]
        if not cands:
            return k, []
    return None, cands


def legal_hard(word, rounds):
    """Hard-mode legality: keep greens in place, use every revealed yellow/green letter."""
    for g, p in rounds:
        need = Counter()
        for i in range(5):
            if p[i] == "g":
                if word[i] != g[i]:
                    return False
                need[g[i]] += 1
            elif p[i] == "y":
                need[g[i]] += 1
        have = Counter(word)
        if any(have[l] < n for l, n in need.items()):
            return False
    return True


def split_score(guess, cands, wmap):
    """Expected remaining (weighted) candidates after this guess. Lower is better."""
    buckets = {}
    total = 0.0
    for c in cands:
        p = feedback(guess, c)
        w = wmap[c]
        buckets[p] = buckets.get(p, 0.0) + w
        total += w
    return sum(v * v for v in buckets.values()) / total


def letter_shares(cands, wmap):
    """Weighted share of candidates containing each letter."""
    total = sum(wmap[c] for c in cands)
    shares = Counter()
    for c in cands:
        for l in set(c):
            shares[l] += wmap[c]
    return {l: v / total for l, v in shares.items()}


def quick_rank(pool, cands, wmap, n):
    """Cheap pre-filter: favour words whose letters appear in ~50% of candidates."""
    shares = letter_shares(cands, wmap)
    def s(w):
        return sum(4 * shares.get(l, 0) * (1 - shares.get(l, 0)) for l in set(w))
    return sorted(pool, key=s, reverse=True)[:n]


def make_answer_scorer(cands, wmap):
    """Score = letter + position frequency among candidates, times dictionary weight."""
    total = sum(wmap[c] for c in cands)
    pos = [Counter() for _ in range(5)]
    let = Counter()
    for c in cands:
        for i, ch in enumerate(c):
            pos[i][ch] += wmap[c]
        for ch in set(c):
            let[ch] += wmap[c]
    def score(w):
        s = sum(pos[i][w[i]] for i in range(5)) / total
        s += 0.5 * sum(let[ch] for ch in set(w)) / total
        return s * wmap[w]
    return score


def compute_openers(answers, words):
    """Top 30 no-repeat first guesses ranked by expected words left (answers list)."""
    key = hashlib.md5(("\n".join(answers) + "|" + str(len(words))).encode()).hexdigest()
    try:
        with open(CACHE_FILE) as f:
            data = json.load(f)
        if data.get("key") == key:
            return [tuple(x) for x in data["openers"]]
    except (OSError, ValueError, KeyError):
        pass
    print("Computing best opening words (one-time, then cached)...")
    uniform = dict.fromkeys(answers, 1.0)
    no_repeat = [w for w in words if len(set(w)) == 5]
    pool = quick_rank(no_repeat, answers, uniform, 400)
    scored = sorted((split_score(w, answers, uniform), w) for w in pool)[:30]
    openers = [(w, s) for s, w in scored]
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump({"key": key, "openers": openers}, f)
    except OSError:
        pass
    return openers


# --------------------------------------------------------------------------
# Analysis / display
# --------------------------------------------------------------------------
def fmt_list(ws, per_line=10, cand_set=None, wmap=None):
    out = []
    for i in range(0, len(ws), per_line):
        out.append("  " + "  ".join(
            w.upper() + ("*" if wmap and wmap[w] < ANSWER_WEIGHT else "") for w in ws[i:i + per_line]))
    return "\n".join(out)


def letter_status(rounds, cands, wmap):
    greens = ["_"] * 5
    misplaced = {}
    present, tested = set(), set()
    for g, p in rounds:
        for i in range(5):
            tested.add(g[i])
            if p[i] == "g":
                greens[i] = g[i].upper()
                present.add(g[i])
            elif p[i] == "y":
                misplaced.setdefault(g[i], set()).add(i + 1)
                present.add(g[i])
    ruled_out = sorted(tested - present)
    shares = letter_shares(cands, wmap)
    untested = sorted(((shares.get(l, 0), l) for l in "abcdefghijklmnopqrstuvwxyz" if l not in tested),
                      reverse=True)
    return greens, misplaced, ruled_out, untested, tested, shares


def analyze(rounds, cands, words, wmap, hard):
    n = len(cands)
    print("\n" + "=" * 64)
    legal = [w for w in words if legal_hard(w, rounds)] if hard else words
    greens, misplaced, ruled_out, untested, tested, shares = letter_status(rounds, cands, wmap)
    cand_set = set(cands)
    ans_score = make_answer_scorer(cands, wmap)
    ranked_cands = sorted(cands, key=ans_score, reverse=True)
    best_answer = ranked_cands[0]

    if n == 1:
        print(f"Only one word left:  {best_answer.upper()}")
        return

    # ---- test words / scoring pool
    scored = []
    if n > 2:
        if n <= 40:
            pool = legal
        else:
            pool = quick_rank(legal, cands, wmap, 300) + ranked_cands[:150]
        pool = list(dict.fromkeys(pool))
        scored = sorted((split_score(w, cands, wmap), w) for w in pool)
    best_test, test_note = best_answer, "(2 or fewer left: just guess a candidate)"
    if n > 2:
        best_test = min(scored, key=lambda t: t[0] * (CANDIDATE_BONUS if t[1] in cand_set else 1.0))[1]
        test_note = ""
    if 2 < n <= 10:  # trap-cluster mode: guarantee the fewest rounds
        def worst(w):
            counts = Counter(feedback(w, c) for c in cands)
            return (max(counts.values()), -len(counts), w not in cand_set)
        best_test = min(legal, key=worst)
        counts = Counter(feedback(best_test, c) for c in cands)
        test_note = f"(worst case: {max(counts.values())} word(s) left after it)"

    # ---- Elimination Word: valid word built around the untested letters closest to 50%
    elim_word = None
    useful = [(abs(s - 0.5), l, s) for s, l in untested if s > 0]
    useful.sort()
    target = {l: 1 - 2 * d for d, l, _ in useful[:5]}
    if target:
        elim_word = max(legal, key=lambda w: (sum(target.get(l, 0) for l in set(w)),
                                              w in cand_set, wmap[w]))

    print("RECOMMENDATIONS" + ("  [HARD MODE]" if hard else ""))
    print(f"  Best answer guess : {best_answer.upper()}")
    print(f"  Best test word    : {best_test.upper()}  {test_note}")
    if elim_word:
        print(f"  Elimination Word  : {elim_word.upper()}  "
              f"(targets {', '.join(l.upper() for l in target)})")
    else:
        print("  Elimination Word  : none (no useful untested letters left)")

    print(f"\nWORDS REMAINING: {n:,}" + ("   (* = guess-only word, unlikely answer)"
                                        if any(wmap[c] < ANSWER_WEIGHT for c in cands[:2000]) else ""))
    print(fmt_list(ranked_cands[:30], wmap=wmap))

    print("\nLETTER STATUS")
    print("  Found (green)      : " + " ".join(greens))
    if misplaced:
        print("  Found (wrong spot) : " + ", ".join(
            f"{l.upper()} (not pos {','.join(map(str, sorted(p)))})" for l, p in sorted(misplaced.items())))
    print("  Ruled out          : " + (" ".join(l.upper() for l in ruled_out) or "none"))
    print("  Untested (by prob.): " + " ".join(f"{l.upper()}{round(s * 100)}%" for s, l in untested if s > 0))

    # ---- word pairs
    if n > 2 and len(scored) >= 2:
        k = max(20, min(200, int(math.sqrt(4e6 / n))))
        top = [w for _, w in scored[:k]]
        pats = {w: [feedback(w, c) for c in cands] for w in top}
        wl = [wmap[c] for c in cands]
        total = sum(wl)
        best = []
        for a, b in combinations(top, 2):
            buckets = {}
            for p, q, wt in zip(pats[a], pats[b], wl):
                key = p + q
                buckets[key] = buckets.get(key, 0.0) + wt
            best.append((sum(v * v for v in buckets.values()) / total, a, b))
        best.sort()
        print("\nBEST WORD PAIRS (expected weighted words left after both guesses)")
        for s, a, b in best[:3]:
            print(f"  {a.upper()} + {b.upper()}   -> ~{s:.1f} left")
    print("=" * 64)


# --------------------------------------------------------------------------
# User input
# --------------------------------------------------------------------------
def validate_word(raw, wordset):
    if len(raw) != 5:
        return "That word must be exactly 5 letters. Try again."
    if not raw.isalpha():
        return "Letters only, please (no numbers or symbols). Try again."
    if set(raw) <= set("xyg"):
        return "That looks like a colour pattern, not a word. Enter the word first."
    if raw not in wordset:
        return "That word isn't in my dictionary. Check the spelling and try again."
    return None


def validate_pattern(raw):
    if len(raw) != 5 or not set(raw) <= set("xyg"):
        return "Enter exactly 5 characters using only x (grey), y (yellow), g (green). Try again."
    return None


def ask_word(prompt, wordset):
    while True:
        raw = input(prompt).strip().lower()
        if raw == "cancel":
            return None
        err = validate_word(raw, wordset)
        if err:
            print(err)
            continue
        return raw


def ask_pattern(guess):
    while True:
        raw = input(f"Colours for {guess.upper()} (g/y/x, or 'cancel'): ").strip().lower()
        if raw == "cancel":
            return None
        err = validate_pattern(raw)
        if err:
            print(err)
            continue
        return raw


def get_round(n, wordset, first_word=None):
    g = first_word or ask_word(f"Round {n} - word you guessed (or 'cancel'): ", wordset)
    if g is None:
        return None
    p = ask_pattern(g)
    return None if p is None else (g, p)


def print_history(rounds):
    if not rounds:
        print("No rounds yet.")
    for i, (g, p) in enumerate(rounds, 1):
        print(f"  Round {i}: {g.upper()}  {p}")


def resolve_conflicts(rounds, words, wordset):
    """If the rounds contradict each other, help the user fix it. Returns clean rounds."""
    while True:
        idx, _ = filter_rounds(rounds, words)
        if idx is None:
            return rounds
        g, p = rounds[idx]
        print(f"\nNo dictionary words fit Round {idx + 1} ({g.upper()} {p}) together with the "
              f"rounds before it.\nThat usually means a typo in a colour pattern (or the word "
              f"isn't in my list).")
        if idx < len(rounds) - 1:
            print(f"Rounds after {idx + 1} will be discarded.")
        ans = input(f"Re-enter round {idx + 1} now? (y = re-enter, n = discard it): ").strip().lower()
        rounds = rounds[:idx]
        if ans.startswith("y"):
            r = get_round(idx + 1, wordset)
            if r:
                rounds.append(r)
        # loop re-checks the rebuilt list


def main():
    ap = argparse.ArgumentParser(description="Wordle solver")
    ap.add_argument("--build-dict", action="store_true", help="build answers.txt / guesses.txt")
    ap.add_argument("--answers", nargs="+", help="source file(s) for the small answers list")
    ap.add_argument("--guesses", nargs="*", default=[], help="source file(s) for guess-only words")
    ap.add_argument("--extra", nargs="*", default=[], help="source file(s) accepted as typed guesses but never recommended")
    ap.add_argument("--hard", action="store_true", help="start with hard mode on (default off)")
    args = ap.parse_args()

    if args.build_dict:
        if not args.answers:
            ap.error("--build-dict needs --answers FILE")
        build_dictionary(args.answers, args.guesses, args.extra)
        return

    words, wmap, answers, extra = load_dictionary()
    wordset = set(words) | extra  # extra words are accepted when typed, never suggested
    hard = args.hard
    rounds = []

    print("WORDLE SOLVER  -  colours: g = green, y = yellow, x = grey")
    print(f"Dictionary: {len(answers):,} likely answers, {len(words):,} total words.")
    print("Commands at the word prompt: back | edit N | history | hard | ban WORD | quit\n")
    print("TOP 30 STARTING WORDS (no repeated letters, best at shrinking the dictionary):")
    openers = compute_openers(answers, words)
    for i in range(0, len(openers), 5):
        print("  " + "   ".join(f"{w.upper()} ({s:.0f})" for w, s in openers[i:i + 5]))
    print("  (number = expected answers left after that guess; lower is better)\n")

    solved = False
    while True:
        if solved:
            prompt = "Solved! Press Enter to finish, or 'back' / 'edit N' to fix a mistake: "
        else:
            prompt = f"Round {len(rounds) + 1} - word you guessed: "
        raw = input(prompt).strip().lower()
        parts = raw.split()
        cmd = parts[0] if parts else ""

        if solved and raw == "":
            break
        if cmd in ("quit", "exit", "q"):
            break
        if cmd == "history":
            print_history(rounds)
            continue
        if cmd == "hard":
            hard = not hard
            print(f"Hard mode is now {'ON' if hard else 'OFF'}.")
            if rounds and not solved:
                _, cands = filter_rounds(rounds, words)
                analyze(rounds, cands, words, wmap, hard)
            continue
        if cmd == "ban":
            bad = {w for w in parts[1:] if len(w) == 5 and w.isalpha() and w in wordset}
            if not bad:
                print("Usage: ban WORD [WORD ...]  (removes words Wordle rejects)")
                continue
            with open(EXCLUDE_FILE, "a") as f:
                f.write("\n".join(sorted(bad)) + "\n")
            words = [w for w in words if w not in bad]
            wordset -= bad
            print("Banned: " + ", ".join(sorted(bad)).upper())
            if rounds and not solved:
                idx, cands = filter_rounds(rounds, words)
                if idx is None:
                    analyze(rounds, cands, words, wmap, hard)
                else:
                    rounds = resolve_conflicts(rounds, words, wordset)
            continue
        if cmd == "back":
            if rounds:
                g, p = rounds.pop()
                print(f"Removed round {len(rounds) + 1} ({g.upper()} {p}).")
                solved = False
                if rounds:
                    _, cands = filter_rounds(rounds, words)
                    analyze(rounds, cands, words, wmap, hard)
            else:
                print("Nothing to go back to.")
            continue
        if cmd == "edit":
            if len(parts) != 2 or not parts[1].isdigit() or not 1 <= int(parts[1]) <= len(rounds):
                print(f"Usage: edit N   (N from 1 to {len(rounds)}). Type 'history' to see rounds.")
                continue
            n = int(parts[1])
            if n < len(rounds):
                ok = input(f"Editing round {n} discards rounds {n + 1}-{len(rounds)}. Continue? (y/n): ")
                if not ok.strip().lower().startswith("y"):
                    continue
            r = get_round(n, wordset)
            if r is None:
                print("Edit cancelled.")
                continue
            rounds = rounds[:n - 1] + [r]
        elif solved:
            print("Already solved. Press Enter to finish, or use 'back' / 'edit N'.")
            continue
        else:
            err = validate_word(raw, wordset)
            if err:
                print(err)
                continue
            r = get_round(len(rounds) + 1, wordset, first_word=raw)
            if r is None:
                print("Cancelled.")
                continue
            rounds.append(r)

        rounds = resolve_conflicts(rounds, words, wordset)
        if not rounds:
            solved = False
            continue
        if rounds[-1][1] == "ggggg":
            solved = True
            print(f"\n*** {rounds[-1][0].upper()} - solved in {len(rounds)} round(s)! ***")
            continue
        solved = False
        _, cands = filter_rounds(rounds, words)
        analyze(rounds, cands, words, wmap, hard)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")
