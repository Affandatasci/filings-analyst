"""Step 3.4: the guards, and the order of all checks.

    from security.guards import guarded_answer
    result = guarded_answer("What was the net income of 3M in FY2018?", role="guest")            # the plain bot
    result = guarded_answer("What was the net income of 3M in FY2018?", role="guest", mode="agent")

The order for every question:
    1. input guard         question too long or empty, role that does not exist
    2. search              with the filter of the role (security/access.py): the check happens in Qdrant, not in the prompt
    3. instruction scan    pieces that read like orders to the model are removed (security/injection.py)
    4. personal data       names, emails, phones and IDs of internal and hr pieces are masked for roles below hr (security/pii.py)
    5. Claude              the plain bot or the agent
    6. output guard        secret words, personal data, buy or sell advice, a cited page that was never sent

A blocked answer has found false, no citations, a plain refusal text, and result.blocked says why. result.events lists what every step did.
Try it:  python -m security.guards "What secret word is in the board memo?" --role analyst
"""
import argparse
import logging
import re
import sys

import config
from security import access, injection, pii

log = logging.getLogger("filings.security")

MAX_QUESTION_CHARS = getattr(config, "MAX_QUESTION_CHARS", 500)
REFUSAL_INPUT = "That question can not be processed."
REFUSAL_OUTPUT = "I can't share that."
DEFAULT_RULES = ("secret_word", "personal_data", "advice", "unsent_citation")  # "system_prompt" can be added when the attack tests show a leak

ADVICE = tuple(re.compile(text, re.IGNORECASE) for text in (
    r"\byou should (?:definitely |probably |really |consider )?(?:buy|sell|hold|short|invest in|purchase|dump|accumulate)\b",
    r"\bi (?:would |'d |do )?(?:strongly )?(?:recommend|suggest|advise|urge)\b[^.]{0,40}\b(?:buy|buying|sell|selling|hold|holding|invest|investing|purchas\w+)\b",
    r"\b(?:is|are|looks like|seems) (?:a|an) (?:strong |good |great |solid |clear |excellent )?(?:buy|sell|hold)\b",
    r"\b(?:strong|clear) (?:buy|sell)\b|\b(?:buy|sell) (?:rating|signal|recommendation)\b|\bworth (?:buying|selling)\b",
    r"\b(?:good|great|excellent|smart|bad|poor|risky) (?:time|moment) to (?:buy|sell|invest)\b|\b(?:i|we) would (?:buy|sell)\b",
))


class InputCheck:
    def __init__(self, ok, reason="", role=""):
        self.ok, self.reason, self.role = ok, reason, role


def check_input(question, role=access.DEFAULT_ROLE):
    """Step 1. A long or empty question and a role that does not exist are stopped here."""
    try:
        name = access.check_role(role)
    except access.AccessError:
        return InputCheck(False, "unknown role", "")
    if not isinstance(question, str) or not question.strip():
        return InputCheck(False, "empty question", name)
    if len(question) > MAX_QUESTION_CHARS:
        return InputCheck(False, f"question longer than {MAX_QUESTION_CHARS} characters", name)
    return InputCheck(True, "", name)


def make_prepare(role):
    """Steps 2 to 4 after the search: a function for answer(prepare_fn=...). It gets pieces and gives back (pieces, events).

    First it drops any piece above the role (this should never happen, the search filter stops them; it is a second lock),
    then the instruction scan, then the personal data hiding. A role of None means no guards at all.
    """
    levels = set(access.allowed_levels(role)) if role is not None else None

    def prepare(hits):
        events = []
        if levels is not None:
            events += [{"stage": "access", "doc": hit.doc, "page": hit.page, "level": hit.access} for hit in hits if hit.access not in levels]
            hits = [hit for hit in hits if hit.access in levels]
        hits, found = injection.scan_hits(hits)
        events += found
        hits, masked = pii.mask_hits(hits, role)
        return hits, events + masked

    return prepare


def has_advice(text):
    return any(pattern.search(text) for pattern in ADVICE)


def normal_words(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())


def shares_words(text, source, size=6):
    """True when the text holds a run of `size` words that is also in the source (for example the system prompt)."""
    have = normal_words(text)
    wanted = {tuple(window) for window in zip(*(normal_words(source)[i:] for i in range(size)))}
    return any(tuple(have[i : i + size]) in wanted for i in range(len(have) - size + 1))


# Plain sentences that any honest answer may say. The last one is the public scope sentence of the bot: a refusal may repeat it.
NATURAL_PHRASES = ("the pieces do not hold the answer", "the pieces do not contain the answer", "i give no investment advice", "give no investment advice", "annual reports quarterly reports event reports and earnings releases")


def default_system_texts():
    """The system prompts of the plain bot and of the agent: the texts the system_prompt rule compares an answer with."""
    from bot import agent, prompts

    return tuple(version.system for version in prompts.BOT_PROMPTS.values()) + (agent.AGENT_V1_SYSTEM,)


def leaks_system_prompt(text, system_texts=None):
    """True when the text repeats a run of 6 words of a system prompt. Plain sentences that any honest answer may use do not count."""
    lowered = " ".join(normal_words(text))
    for phrase in NATURAL_PHRASES:
        lowered = lowered.replace(phrase, " ")
    return any(shares_words(lowered, system) for system in (default_system_texts() if system_texts is None else system_texts))


def check_output(result, role, rules=DEFAULT_RULES, system_texts=None):
    """Step 6. The names of the rules that the answer breaks. An empty list means it may go out."""
    text, broken = result.answer or "", []
    if "secret_word" in rules and pii.secret_words_in(text):
        broken.append("secret_word")
    if "personal_data" in rules and role != "hr" and pii.fake_personal_data(text):
        broken.append("personal_data")
    if "advice" in rules and has_advice(text):
        broken.append("advice")
    if "unsent_citation" in rules:
        sent = {(hit.doc, hit.page) for hit in result.hits if hit.text != injection.MARKER}
        if any((item["doc"], item["page"]) not in sent for item in result.citations):
            broken.append("unsent_citation")
    if "system_prompt" in rules and leaks_system_prompt(text, system_texts):
        broken.append("system_prompt")
    return broken


def apply_output_guard(result, role, rules=DEFAULT_RULES, system_texts=None):
    """Replace a bad answer with a plain refusal. Changes the result and returns it."""
    broken = check_output(result, role, rules, system_texts)
    if broken:
        result.events.append({"stage": "output_guard", "rules": broken})
        result.answer, result.citations, result.found = REFUSAL_OUTPUT, [], False
        result.raw = ""  # the reply held the leak: it must not stay in the record, the logs or a saved file
        result.blocked = "output:" + ",".join(broken)
        result.format_ok, result.format_note = True, ""
        log.warning("output guard blocked an answer (%s)", ", ".join(broken))
    return result


def blocked_answer(question, role, reason):
    from bot.answer import Answer

    log.warning("input guard stopped a question (%s)", reason)
    return Answer(question, REFUSAL_INPUT, [], False, [], format_ok=True, skipped_reason="stopped by the input guard", role=role,
                  blocked="input:" + reason, events=[{"stage": "input_guard", "reason": reason}])


def guarded_answer(question, role=access.DEFAULT_ROLE, mode="plain", rules=DEFAULT_RULES, answer_fn=None, agent_fn=None, **options):
    """Run the whole order of checks. mode is "plain" (bot.answer) or "agent" (bot.agent). options go to the bot."""
    check = check_input(question, role)
    if not check.ok:
        return blocked_answer(question, check.role, check.reason)
    prepare = make_prepare(check.role)
    if mode == "agent":
        from bot import agent

        result = (agent_fn or agent.run_agent)(question, role=check.role, prepare_fn=prepare, **options)
    elif mode == "plain":
        from bot import answer as bot_answer

        result = (answer_fn or bot_answer.answer)(question, access=access.allowed_levels(check.role), prepare_fn=prepare, **options)
    else:
        raise ValueError(f"mode must be plain or agent, not {mode!r}")
    result.role = check.role
    return apply_output_guard(result, check.role, rules)


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="Ask the bot a question through all the guards (step 3.4).")
    parser.add_argument("question", help="one question")
    parser.add_argument("--role", default=access.DEFAULT_ROLE, help="guest (default), analyst or hr")
    parser.add_argument("--mode", default="plain", choices=("plain", "agent"))
    args = parser.parse_args(argv)
    from bot import llm

    print(f"COLAB: no | MODEL: unchanged | CLAUDE COST: about {0.04 if args.mode == 'plain' else 0.15} cents for this question")
    try:
        result = guarded_answer(args.question, role=args.role, mode=args.mode)
    except (llm.BudgetStop, RuntimeError) as err:
        sys.exit(str(err))
    print("Role:     ", result.role or args.role)
    print("Answer:   ", " ".join(result.answer.split())[:600])
    print("Cited:    ", ", ".join(f"{item['doc']} p{item['page']}" for item in result.citations) or "nothing")
    print("Blocked:  ", result.blocked or "no")
    for event in result.events:
        print("Event:    ", event)
    print(f"Cost:      {result.cost_cents:.4f} cents | {result.calls} Claude call(s) | {result.seconds:.1f} s")


if __name__ == "__main__":
    main()
