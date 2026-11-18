"""Task generators and scorers: code, math, needle, json. 32 EXPLORE + 128 CONFIRM distinct items per kind.

  python -m tools.qualification.turbo_bar.tasks gen --out tasks.jsonl     # prints the summary sha
Items: {id, kind, split, messages, max_tokens, check, sha}; sha = sha256 of the canonical messages (prompt).
EXPLORE and CONFIRM draw from disjoint seed namespaces; generation asserts that no prompt repeats.
"""
import argparse
import hashlib
import json
import math
import random
import re
import sys

from tools.qualification.turbo_bar.common import canon, canon_sha, sha256_bytes, write_jsonl

KINDS = ("code", "math", "needle", "json")
SPLIT_SIZES = {"explore": 32, "confirm": 128}


def rng_for(kind, split, i, attempt=0):
    h = hashlib.sha256(f"turbo-bar/{kind}/{split}/{i}/{attempt}".encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


def _seed(*parts):
    return random.Random(int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big"))


def msgs(text):
    return [{"role": "user", "content": text}]


# ----------------------------------------------------------------------------------------------- math
def gen_math(rng):
    """Returns (question, answer:int, steps:list[str]); several families, all with a checkable integer."""
    t = rng.randrange(7)
    if t == 0:
        a, b, c, d, e = rng.randint(12, 90), rng.randint(6, 48), rng.randint(20, 400), rng.randint(3, 40), rng.randint(5, 60)
        item = rng.choice(["crates", "boxes", "cartons"])
        q = (f"A warehouse holds {a} {item} with {b} units each. {c} units are shipped out, then {d} new {item} "
             f"of {e} units each arrive. How many units are in the warehouse now?")
        s1, s2 = a * b, d * e
        return q, s1 - c + s2, [f"{a} * {b} = {s1} units at the start.", f"{s1} - {c} = {s1 - c}.",
                                f"{d} * {e} = {s2} units arrive.", f"{s1 - c} + {s2} = {s1 - c + s2}."]
    if t == 1:
        a, b, c, m = rng.randint(3, 40), rng.randint(5, 60), rng.randint(1, 99), rng.randint(7, 97)
        p = pow(a, b, m)
        return (f"What is the remainder when {a}^{b} + {c} is divided by {m}?", (p + c) % m,
                [f"{a}^{b} mod {m} = {p}.", f"({p} + {c}) mod {m} = {(p + c) % m}."])
    if t == 2:
        x = rng.randint(-30, 30)
        a, c = rng.randint(2, 15), rng.randint(-9, 1)
        while a == c:
            a += 1
        b = rng.randint(-40, 40)
        d = (a - c) * x + b
        return (f"Solve {a}x + {b} = {c}x + {d} for x. What is x?", x,
                [f"Move terms: ({a} - {c})x = {d} - {b}.", f"{a - c}x = {d - b}.", f"x = {x}."])
    if t == 3:
        n = rng.randint(8, 20)
        k = rng.randint(2, 5)
        if rng.random() < 0.5:
            return (f"In how many ways can a committee of {k} be chosen from {n} people?", math.comb(n, k),
                    [f"C({n},{k}) = {math.comb(n, k)}."])
        return (f"In how many ways can {k} of {n} runners take gold, silver and bronze-style ranked places "
                f"(ordered, {k} places)?", math.perm(n, k), [f"P({n},{k}) = {math.perm(n, k)}."])
    if t == 4:
        a, d, n = rng.randint(-20, 50), rng.randint(2, 13), rng.randint(8, 60)
        ans = n * (2 * a + (n - 1) * d) // 2
        return (f"An arithmetic sequence starts at {a} and increases by {d} each step. "
                f"What is the sum of its first {n} terms?", ans,
                [f"Last term = {a} + {n - 1} * {d} = {a + (n - 1) * d}.", f"Sum = {n} * ({a} + {a + (n - 1) * d}) / 2 = {ans}."])
    if t == 5:
        n, a, b = rng.randint(200, 2500), rng.randint(3, 12), rng.randint(5, 17)
        while a == b:
            b += 1
        ans = sum(1 for i in range(1, n + 1) if (i % a == 0) != (i % b == 0))
        return (f"How many integers from 1 to {n} are divisible by exactly one of {a} and {b}?", ans,
                [f"Divisible by {a}: {n // a}; by {b}: {n // b}; by both: {n // (a * b // math.gcd(a, b))}.",
                 f"Exactly one = {n // a} + {n // b} - 2 * {n // (a * b // math.gcd(a, b))} = {ans}."])
    a, b = rng.randint(12, 240), rng.randint(12, 240)
    g, l = math.gcd(a, b), a * b // math.gcd(a, b)
    return (f"Let g be the greatest common divisor and l the least common multiple of {a} and {b}. "
            f"What is l - g?", l - g, [f"g = {g}, l = {l}.", f"l - g = {l - g}."])


def math_item(rng):
    q, ans, _ = gen_math(rng)
    text = ("Solve the problem. Reason briefly, then finish with a final line of the form `ANSWER: <integer>`.\n\n"
            f"Problem: {q}")
    return msgs(text), 512, {"answer": ans}


def score_math(reply, check):
    # The final field must be a whole integer. Support balanced bold around
    # the value and conventional thousands separators, but no numeric suffix.
    if "ANSWER:" not in reply:
        return False
    field = reply.rsplit("ANSWER:", 1)[1].strip()
    if field.startswith("**") and field.endswith("**"):
        field = field[2:-2].strip()
    if not re.fullmatch(r"[+-]?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)", field):
        return False
    try:
        return int(field.replace(",", "")) == int(check["answer"])
    except ValueError:
        return False


# ----------------------------------------------------------------------------------------------- code
def _words(rng, n):
    syl = ["ka", "to", "mi", "re", "sun", "lo", "ba", "den", "pol", "vi", "ar", "xe", "qua", "ten", "zo"]
    return ["".join(rng.choice(syl) for _ in range(rng.randint(1, 3))) for _ in range(n)]


def _c_caesar(rng):
    k = rng.randint(1, 25)

    def ref(s):
        out = []
        for ch in s:
            if "a" <= ch <= "z":
                out.append(chr((ord(ch) - 97 + k) % 26 + 97))
            elif "A" <= ch <= "Z":
                out.append(chr((ord(ch) - 65 + k) % 26 + 65))
            else:
                out.append(ch)
        return "".join(out)

    def inp(r):
        return (" ".join(w.capitalize() if r.random() < .3 else w for w in _words(r, r.randint(1, 5))) + r.choice(["", "!", ", 7"]),)
    return (f"shifts every ASCII letter of the string s forward by {k} places in the alphabet (wrapping z to a), "
            "keeping case and leaving all other characters unchanged", "shift_text", "s", ref, inp)


def _c_longwords(rng):
    k = rng.randint(3, 8)

    def ref(text):
        return [w.lower() for w in text.split() if len(w) > k]

    def inp(r):
        return (" ".join(w.upper() if r.random() < .3 else w for w in _words(r, r.randint(0, 9))),)
    return (f"returns the list of words of text (split on whitespace) that have more than {k} characters, in order, "
            "converted to lowercase", "long_words", "text", ref, inp)


def _c_multiples(rng):
    a, b = rng.randint(2, 9), rng.randint(3, 13)

    def ref(n):
        return sum(i for i in range(1, n) if i % a == 0 or i % b == 0)

    def inp(r):
        return (r.choice([0, 1, 2, r.randint(3, 400), r.randint(100, 3000)]),)
    return (f"returns the sum of all positive integers below n that are divisible by {a} or by {b}", "sum_multiples", "n",
            ref, inp)


def _c_primes(rng):
    m, r0 = rng.choice([3, 4, 5, 6, 7, 10]), None
    r0 = rng.choice([x for x in range(m) if math.gcd(x, m) == 1])

    def ref(lo, hi):
        def pr(n):
            return n > 1 and all(n % d for d in range(2, int(n ** .5) + 1))
        return sum(1 for p in range(lo, hi + 1) if pr(p) and p % m == r0)

    def inp(r):
        lo = r.randint(0, 60)
        return (lo, lo + r.randint(0, 400))
    return (f"returns how many primes p with lo <= p <= hi (inclusive) satisfy p % {m} == {r0}", "count_primes_mod", "lo, hi",
            ref, inp)


def _c_parse(rng):
    sep = rng.choice([";", ",", "|"])
    pre = rng.choice(["a", "b", "k", "x", "ab"])

    def ref(s):
        tot = 0
        for part in s.split(sep):
            if "=" in part:
                key, val = part.split("=", 1)
                if key.startswith(pre):
                    tot += int(val)
        return tot

    def inp(r):
        keys = [r.choice(["a", "b", "k", "x", "ab", "ax", "bk"]) + str(r.randint(0, 9)) for _ in range(r.randint(0, 7))]
        return (sep.join(f"{k}={r.randint(-20, 99)}" for k in keys),)
    return (f"parses a string of key=value pairs separated by '{sep}' (values are integers, possibly negative; the string may "
            f"be empty) and returns the sum of the values whose key starts with '{pre}'", "prefix_total", "s", ref, inp)


def _c_rotstep(rng):
    k, m = rng.randint(1, 7), rng.randint(2, 4)

    def ref(xs):
        if not xs:
            return []
        j = k % len(xs)
        return (xs[j:] + xs[:j])[::m]

    def inp(r):
        return ([r.randint(-9, 99) for _ in range(r.randint(0, 12))],)
    return (f"rotates the list xs left by {k} positions (rotation count taken modulo its length) and then returns every "
            f"{m}-th element of the rotated list starting at index 0; an empty list gives []", "rot_step", "xs", ref, inp)


def _c_lcm(rng):
    m = rng.choice([97, 1009, 10007, 998, 360, 1000003])

    def ref(xs):
        l = 1
        for x in xs:
            l = l * x // math.gcd(l, x)
        return l % m

    def inp(r):
        return ([r.randint(1, 60) for _ in range(r.randint(1, 8))],)
    return (f"returns the least common multiple of a non-empty list xs of positive integers, taken modulo {m}",
            "lcm_mod", "xs", ref, inp)


def _c_rle(rng):
    count_first = rng.random() < 0.5

    def ref(s):
        out, i = [], 0
        while i < len(s):
            j = i
            while j < len(s) and s[j] == s[i]:
                j += 1
            n = j - i
            out.append(s[i] if n == 1 else (f"{n}{s[i]}" if count_first else f"{s[i]}{n}"))
            i = j
        return "".join(out)

    def inp(r):
        return ("".join(r.choice("abc") * r.randint(1, 5) for _ in range(r.randint(0, 6))),)
    fmt = "the count followed by the character" if count_first else "the character followed by the count"
    return (f"run-length encodes the string s: every run of identical characters becomes {fmt}, except that a run of length 1 "
            "is written as the bare character", "rle", "s", ref, inp)


def _c_bits(rng):
    k = rng.randint(1, 4)

    def ref(n):
        return sum(1 for i in range(1, n + 1) if bin(i).count("1") == k)

    def inp(r):
        return (r.choice([1, 2, 7, r.randint(8, 300), r.randint(300, 3000)]),)
    return (f"returns how many integers i with 1 <= i <= n have exactly {k} bits set in binary", "count_bits", "n", ref, inp)


def _c_depth(rng):
    pairs = rng.choice(["()[]", "()[]{}", "()<>", "[]{}"])
    opens, closes = pairs[0::2], pairs[1::2]

    def ref(s):
        st, best = [], 0
        for ch in s:
            if ch in opens:
                st.append(ch)
                best = max(best, len(st))
            elif ch in closes:
                if not st or opens.index(st.pop()) != closes.index(ch):
                    return -1
        return best if not st else -1

    def inp(r):
        s = "".join(r.choice(pairs) for _ in range(r.randint(0, 10)))
        if r.random() < .6:
            def g(d):
                if d == 0 or r.random() < .3:
                    return ""
                i = r.randrange(len(opens))
                return opens[i] + g(d - 1) + closes[i] + (g(d - 1) if r.random() < .4 else "")
            s = g(r.randint(1, 6))
        return (s,)
    return (f"returns the maximum nesting depth of the bracket string s that uses only the bracket pairs "
            f"{', '.join(pairs[i:i + 2] for i in range(0, len(pairs), 2))}; return -1 if the brackets are not balanced "
            "or are mismatched; an empty string has depth 0", "max_depth", "s", ref, inp)


def _c_recur(rng):
    A, B, C, D, M = rng.randint(0, 9), rng.randint(1, 9), rng.randint(1, 6), rng.randint(1, 6), rng.choice([7, 10, 97, 1000])

    def ref(n):
        a, b = A, B
        for _ in range(n):
            a, b = b, (C * b + D * a) % M
        return a % M

    def inp(r):
        return (r.choice([0, 1, 2, r.randint(3, 30), r.randint(30, 400)]),)
    return (f"returns term t(n) (n >= 0) of the sequence t(0) = {A}, t(1) = {B}, t(k) = ({C} * t(k-1) + {D} * t(k-2)) % {M} "
            f"for k >= 2 (t(0) and t(1) are returned as given reduced modulo {M} only if n >= 2 applies the rule; "
            f"for n = 0 return {A} % {M} and for n = 1 return {B} % {M})", "seq_term", "n",
            lambda n: ref(n) if n >= 2 else (A % M if n == 0 else B % M), inp)


def _c_window(rng):
    K = rng.randint(5, 40)

    def ref(xs):
        best = lo = tot = 0
        for hi, x in enumerate(xs):
            tot += x
            while tot > K:
                tot -= xs[lo]
                lo += 1
            best = max(best, hi - lo + 1)
        return best

    def inp(r):
        return ([r.randint(0, 15) for _ in range(r.randint(0, 14))],)
    return (f"returns the length of the longest contiguous sublist of xs (non-negative integers) whose sum is at most {K}; "
            "0 for an empty list or when no element fits", "longest_window", "xs", ref, inp)


def _c_digits(rng):
    K = rng.randint(2, 9)

    def ref(xs):
        return sum(x for x in xs if sum(map(int, str(x))) % K == 0)

    def inp(r):
        return ([r.randint(0, 9999) for _ in range(r.randint(0, 10))],)
    return (f"returns the sum of the elements of xs (non-negative integers) whose decimal digit sum is divisible by {K}",
            "digit_filter_sum", "xs", ref, inp)


CODE_TEMPLATES = [_c_caesar, _c_longwords, _c_multiples, _c_primes, _c_parse, _c_rotstep, _c_lcm, _c_rle, _c_bits,
                  _c_depth, _c_recur, _c_window, _c_digits]


def _call(fname, args):
    return f"{fname}({', '.join(repr(a) for a in args)})"


def code_item(rng):
    tpl = CODE_TEMPLATES[rng.randrange(len(CODE_TEMPLATES))]
    desc, fname, params, ref, inp = tpl(rng)
    cases, seen = [], set()
    for _ in range(200):
        a = inp(rng)
        key = repr(a)
        if key not in seen:
            seen.add(key)
            cases.append(a)
        if len(cases) >= 10:
            break
    examples, hidden = cases[:2], cases[2:10]
    ex = "\n".join(f"{_call(fname, a)} == {ref(*a)!r}" for a in examples)
    text = (f"Write a Python function `{fname}({params})` that {desc}.\n\nExamples:\n{ex}\n\n"
            "Reply with the complete function in a single ```python code block and nothing else.")
    tests = "\n".join(f"assert {_call(fname, a)} == {ref(*a)!r}" for a in hidden) + "\n"
    return msgs(text), 768, {"entry": fname, "tests": tests}


def extract_code(reply):
    m = re.search(r"```(?:python|py)?[ \t]*\r?\n(.*?)```", reply, re.S)
    return m.group(1) if m else reply


class ScoringUnavailable(RuntimeError):
    pass


def require_code_sandbox():
    # No qualified OS isolation backend ships with this kit. Never fall back to
    # python -I: it grants filesystem, network and process access to the reply.
    # A future backend must isolate secrets/mounts/network, bound resources, reap
    # the process tree, and use a trusted external controller for every test case.
    raise ScoringUnavailable("code scoring unavailable: qualified OS sandbox and external test controller required")


def score_code(reply, check, timeout=10):
    require_code_sandbox()


# ----------------------------------------------------------------------------------------------- needle
FIRST = ["Marta", "Marla", "Martin", "Elena", "Elias", "Helena", "Tomas", "Thomas", "Anika", "Annika", "Joran", "Jorun",
         "Selma", "Selene", "Viktor", "Victor", "Lucia", "Luciano", "Pavel", "Pavla", "Rune", "Runa", "Dario", "Daria",
         "Nils", "Nina", "Okan", "Oskar", "Ines", "Inge", "Bruno", "Brune", "Carla", "Carlo", "Hugo", "Hugh", "Mira", "Milo",
         "Greta", "Gregor"]
LAST = ["Velasco", "Velasquez", "Valdes", "Lindqvist", "Lindgren", "Okafor", "Okoro", "Petrov", "Petrova", "Hartmann",
        "Hartman", "Moreau", "Moreno", "Kowal", "Kowalski", "Sandoval", "Sandover", "Brandt", "Brandl", "Tanaka", "Tanabe",
        "Ferreira", "Ferrari", "Novak", "Novotny", "Haldane", "Halden", "Rinaldi", "Rinaldo", "Costa", "Costas", "Weber",
        "Weaver", "Ibarra", "Ibanez", "Duarte", "Duran", "Falk", "Falke", "Stroud"]
ADJ = ["quiet", "weathered", "narrow", "bright", "ancient", "crowded", "modest", "rusted", "gentle", "distant", "wooden",
       "cold", "restless", "tidy", "faded", "heavy", "polished", "windy"]
NOUN = ["ledger", "lantern", "bridge", "harbor", "orchard", "workshop", "market", "tower", "garden", "station", "library",
        "granary", "chapel", "mill", "courtyard", "canal", "cellar", "terrace"]
VERB = ["repaired", "inspected", "painted", "measured", "cleaned", "moved", "catalogued", "visited", "mapped", "surveyed"]
DAY = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
PLACE = ["the north gate", "the old quay", "the east wing", "the river bend", "the clock square", "the salt road",
         "the lower terrace", "the east cellar"]
FILLER_T = [
    "The {a1} {n1} was {v} by the {a2} {n2} near {p} on {d}.",
    "On {d}, a {a1} {n1} stood beside {p}, and nobody remarked on it.",
    "{who} {v} the {a1} {n1} at {p} before the {a2} {n2} opened.",
    "Records show that the {n1} near {p} was {v} twice, once on {d}.",
    "A {a1} {n1} and a {a2} {n2} shared the same wall by {p}.",
    "{who} wrote that the {a1} {n1} looked {a2} on {d}.",
    "Nothing about the {a1} {n1} at {p} changed until {d}.",
    "The {n1} at {p} is {a1}, while the {n2} is {a2}.",
]
HEADER = "Below are archive notes. Read them carefully and answer the question at the end.\n\n"


def _name_pool(rng, key):
    f0, l0 = key.split()
    similar = [f"{f0} {l}" for l in LAST if l != l0][:0]
    near = [f"{f0} {l}" for l in rng.sample(LAST, 5) if l != l0] + [f"{f} {l0}" for f in rng.sample(FIRST, 5) if f != f0]
    rest = [f"{f} {l}" for f in FIRST for l in LAST if f"{f} {l}" != key and f"{f} {l}" not in near]
    rng.shuffle(rest)
    return similar + list(dict.fromkeys(near)) + rest


def _filler(rng, names):
    t = rng.choice(FILLER_T)
    return t.format(a1=rng.choice(ADJ), a2=rng.choice(ADJ), n1=rng.choice(NOUN), n2=rng.choice(NOUN), v=rng.choice(VERB),
                    d=rng.choice(DAY), p=rng.choice(PLACE), who=rng.choice(names))


def record_sentence(name, code):
    return f"The archive code of {name} is {code}."


def render(parts):
    out = []
    for i, p in enumerate(parts):
        out.append(p)
        out.append("\n" if i % 7 == 6 else " ")
    return "".join(out).rstrip()


def wrap_prompt(text, name):
    return (HEADER + text + f"\n\nQuestion: What is the archive code of {name}? Answer with the 6 digits only.")


def gen_needle(rng, target_tokens, cpt=4.0, depth=0.5, n_records=None):
    """Haystack as sentence parts. The needle is the record whose position is nearest ``depth`` (0.02..0.95).
    Returns dict(parts, needle_idx, key, value, records: {idx: (name, value)}, cpt)."""
    target_chars = max(2000, int(target_tokens * cpt) - 330)
    est = target_chars // 105 + 1
    n_records = n_records or max(12, min(900, target_chars // 1200))
    n_records = min(n_records, est - 25, len(FIRST) * len(LAST) - 12)
    slots = sorted(rng.sample(range(est - 20), n_records))
    depth = min(0.95, max(0.02, depth))
    slot_pos = [s for s in slots]
    key_slot = min(slots, key=lambda s: abs(s / est - depth))
    key = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    pool = _name_pool(rng, key)
    codes = rng.sample(range(100000, 1000000), n_records)
    names = [pool[i] for i in range(n_records - 1)]
    filler_names = pool[n_records:n_records + 60] or pool[:60]
    parts, records, k, chars, idx = [], {}, 0, 0, 0
    slotset = set(slot_pos)
    while chars < target_chars or idx <= max(slots):
        if idx in slotset:
            if idx == key_slot:
                name, code = key, codes[-1]
            else:
                name, code = names[k], codes[k]
                k += 1
            s = record_sentence(name, code)
            records[idx] = (name, str(code))
        else:
            s = _filler(rng, filler_names)
        parts.append(s)
        chars += len(s) + 1
        idx += 1
    return {"parts": parts, "needle_idx": key_slot, "key": key, "value": str(codes[-1]), "records": records, "cpt": cpt}


def needle_item(rng):
    tokens = rng.randint(1200, 3200)
    g = gen_needle(rng, tokens, 4.0, rng.uniform(0.08, 0.92), n_records=rng.randint(12, 24))
    return msgs(wrap_prompt(render(g["parts"]), g["key"])), 16, {"answer": g["value"]}


def score_needle(reply, check):
    m = re.search(r"(?<!\d)\d{6}(?!\d)", reply)
    return bool(m) and m.group(0) == str(check["answer"])


# ----------------------------------------------------------------------------------------------- json
CITIES = ["Lisbon", "Oslo", "Nairobi", "Quito", "Hanoi", "Perth", "Tallinn", "Lima", "Cusco", "Bern", "Riga", "Kyoto"]
PEOPLE = ["alice", "bruno", "chen", "dalia", "emil", "farah", "goran", "hana"]
ROOMS = ["kitchen", "attic", "study", "garage", "nursery", "cellar"]
CUR = ["EUR", "USD", "JPY", "BRL", "CHF", "SEK"]
FILES = ["*.log", "report*.txt", "src/**/*.py", "*.csv", "backup_*", "notes-??.md"]


def _tool(name, desc, props, required):
    return {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required}}


def _json_tools():
    S, I, B = {"type": "string"}, {"type": "integer"}, {"type": "boolean"}
    return {
        "set_timer": (_tool("set_timer", "Start a countdown timer.",
                            {"minutes": dict(I, description="Duration in minutes"), "label": dict(S, description="Timer label")},
                            ["minutes", "label"]),
                      lambda r: {"minutes": r.randint(2, 180), "label": r.choice(["tea", "laundry", "oven", "stretch", "call"])},
                      lambda v: f"Please start a {v['minutes']} minute timer called '{v['label']}'."),
        "convert_currency": (_tool("convert_currency", "Convert an amount between currencies.",
                                   {"amount": I, "from_code": S, "to_code": S}, ["amount", "from_code", "to_code"]),
                             lambda r: {"amount": r.randint(5, 9000), "from_code": (c := r.sample(CUR, 2))[0], "to_code": c[1]},
                             lambda v: f"How much is {v['amount']} {v['from_code']} in {v['to_code']}? Convert it."),
        "search_files": (_tool("search_files", "Search files by glob pattern.",
                               {"pattern": S, "max_results": I, "include_hidden": B}, ["pattern", "max_results", "include_hidden"]),
                         lambda r: {"pattern": r.choice(FILES), "max_results": r.randint(1, 50), "include_hidden": r.random() < .5},
                         lambda v: (f"Find files matching {v['pattern']}, at most {v['max_results']} results, "
                                    f"{'including' if v['include_hidden'] else 'excluding'} hidden files.")),
        "get_weather": (_tool("get_weather", "Get the weather forecast.",
                              {"city": S, "units": {"type": "string", "enum": ["metric", "imperial"]}, "days": I},
                              ["city", "units", "days"]),
                        lambda r: {"city": r.choice(CITIES), "units": r.choice(["metric", "imperial"]), "days": r.randint(1, 7)},
                        lambda v: f"Give me the {v['days']}-day forecast for {v['city']} in {v['units']} units."),
        "send_message": (_tool("send_message", "Send a chat message.", {"recipient": S, "body": S, "urgent": B},
                               ["recipient", "body", "urgent"]),
                         lambda r: {"recipient": r.choice(PEOPLE), "body": r.choice(["running late", "call me back", "lunch at one",
                                                                                      "see attached", "thanks"]),
                                    "urgent": r.random() < .4},
                         lambda v: (f"Message {v['recipient']} saying \"{v['body']}\"; "
                                    f"{'mark it urgent' if v['urgent'] else 'it is not urgent'}.")),
        "add_to_cart": (_tool("add_to_cart", "Add a product to the shopping cart.", {"sku": S, "quantity": I},
                              ["sku", "quantity"]),
                        lambda r: {"sku": f"{r.choice('ABCDEFGH')}-{r.randint(1000, 9999)}", "quantity": r.randint(1, 12)},
                        lambda v: f"Add {v['quantity']} of item {v['sku']} to my cart."),
        "set_thermostat": (_tool("set_thermostat", "Set a room temperature.", {"room": S, "celsius": I}, ["room", "celsius"]),
                           lambda r: {"room": r.choice(ROOMS), "celsius": r.randint(12, 27)},
                           lambda v: f"Set the {v['room']} to {v['celsius']} degrees Celsius."),
        "create_event": (_tool("create_event", "Create a calendar event.",
                               {"title": S, "date": S, "duration_minutes": I, "attendees": {"type": "array", "items": S}},
                               ["title", "date", "duration_minutes", "attendees"]),
                         lambda r: {"title": r.choice(["review", "planning", "dentist", "demo"]),
                                    "date": f"2031-{r.randint(1, 12):02d}-{r.randint(1, 28):02d}",
                                    "duration_minutes": r.choice([15, 30, 45, 60, 90]),
                                    "attendees": r.sample(PEOPLE, r.randint(1, 3))},
                         lambda v: (f"Schedule '{v['title']}' on {v['date']} for {v['duration_minutes']} minutes with "
                                    f"{', '.join(v['attendees'])}.")),
        "resize_image": (_tool("resize_image", "Resize an image file.", {"path": S, "width": I, "height": I},
                               ["path", "width", "height"]),
                         lambda r: {"path": f"img/{r.choice(['cat', 'dog', 'map', 'logo'])}{r.randint(1, 99)}.png",
                                    "width": r.randint(32, 2048), "height": r.randint(32, 2048)},
                         lambda v: f"Resize {v['path']} to {v['width']} by {v['height']} pixels."),
    }


def json_item(rng):
    tools = _json_tools()
    names = sorted(tools)
    target = rng.choice(names)
    chosen = rng.sample([n for n in names if n != target], rng.randint(0, 2)) + [target]
    rng.shuffle(chosen)
    args = tools[target][1](rng)
    request = tools[target][2](args)
    schemas = "\n".join(json.dumps(tools[n][0]) for n in chosen)
    text = ("You can call these tools (JSON schema, one per line):\n" + schemas +
            "\n\nAnswer ONLY with a JSON object {\"name\": <tool name>, \"arguments\": {...}} using exactly the declared "
            f"parameters and types.\n\nUser request: {request}")
    return msgs(text), 256, {"name": target, "arguments": args}


def strict_eq(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(strict_eq(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(strict_eq(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def first_json_object(text):
    """Accept only a complete JSON object, with optional surrounding whitespace."""
    try:
        obj = json.loads(text.strip())
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


def score_json(reply, check):
    obj = first_json_object(reply)
    if obj is None or set(obj) != {"name", "arguments"}:
        return False
    return obj["name"] == check["name"] and strict_eq(obj["arguments"], check["arguments"])


# ----------------------------------------------------------------------------------------------- items
BUILDERS = {"code": code_item, "math": math_item, "needle": needle_item, "json": json_item}
SCORERS = {"code": score_code, "math": score_math, "needle": score_needle, "retrieval": score_needle, "json": score_json}


def score_item(item, reply):
    return bool(SCORERS[item["kind"]](reply, item["check"]))


def make_items(kind, split, seen=None):
    seen = set() if seen is None else seen
    out = []
    for i in range(SPLIT_SIZES[split]):
        for attempt in range(200):
            messages, max_tokens, check = BUILDERS[kind](rng_for(kind, split, i, attempt))
            sha = sha256_bytes(canon(messages).encode("utf-8"))
            if sha not in seen:
                seen.add(sha)
                break
        else:
            raise RuntimeError(f"cannot generate a distinct {kind} item {split}/{i}")
        out.append({"id": f"{kind}-{split}-{i:03d}", "kind": kind, "split": split, "messages": messages,
                    "max_tokens": max_tokens, "check": check, "sha": sha})
    return out


def generate(kinds=KINDS):
    items = []
    for kind in kinds:
        seen = set()
        e, c = make_items(kind, "explore", seen), make_items(kind, "confirm", seen)
        assert not ({x["sha"] for x in e} & {x["sha"] for x in c}), "explore/confirm overlap"
        items += e + c
    return items


def summary_sha(items):
    return canon_sha(sorted((x["id"], x["sha"], canon_sha(x["check"])) for x in items))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--out", required=True)
    g.add_argument("--kinds", default=",".join(KINDS))
    a = ap.parse_args(argv)
    items = generate(a.kinds.split(","))
    write_jsonl(a.out, items)
    print(len(items), "items; summary sha256", summary_sha(items))
    return 0


if __name__ == "__main__":
    sys.exit(main())
