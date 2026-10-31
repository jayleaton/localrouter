"""Writes corpus.json: the fixed decision corpus the macOS baseline is measured on (bench.py), 50 System One requests.

Each case is a /v1/systemone body plus `expect`, the answer a person would give (a choice's option, a noul's true or
false, a score's level), or nothing where the question is a judgement call. The cases cover what LocalRouter asks:
the voice loop's shadow route question (src/api/voice_local.zig, the tool set of the Jarvis voice loop), the
desktop helper's choices and nouls (apps/desktop-helper/src/computer/decisions.ts), the smoke test's route, scores,
multi-question requests (the shared-prefix path), chat-turn states, many options (up to 40 labels), Chinese, and long
states on both sides of read-twice's 3,000-token cut. Deterministic: no randomness, so the file is regenerated
byte for byte.

    python3 -I tools/decide/knowline/macos/make_corpus.py   (rewrites corpus.json beside it)
"""

import json
import os

ROUTE = ("The user just spoke to their voice assistant (user_said). Which one action should the assistant take for "
         "these words: the tool the request needs, or reply when no tool is needed?")
VOICE_TOOLS = {
    "generate_image": "Make a picture from a text description.",
    "edit_image": "Change a picture the user already has, as they describe.",
    "generate_video": "Make a short video clip from a description.",
    "get_time": "The current time and date.",
    "get_weather": "The weather now or the forecast for a place.",
    "lookup": "Search the web for current facts, news or prices.",
    "computer_use": "Operate the user's computer: open apps, click, type.",
    "list_models": "List the models this machine can run.",
    "reply": "Answer in speech; no tool is needed for this.",
}

VOICE = [
    ("draw me a red fox in the snow", "generate_image"),
    ("can you make a picture of a lighthouse at night", "generate_image"),
    ("make a ten second clip of waves crashing on rocks", "generate_video"),
    ("animate a paper plane flying over a city", "generate_video"),
    ("what time is it", "get_time"),
    ("what's the date today", "get_time"),
    ("is it going to rain in Seattle tomorrow", "get_weather"),
    ("how cold is it outside in Oslo right now", "get_weather"),
    ("who won the football match last night", "lookup"),
    ("what's the price of bitcoin today", "lookup"),
    ("open Safari and go to my calendar", "computer_use"),
    ("click the send button in Mail", "computer_use"),
    ("which models can you run", "list_models"),
    ("thanks, that's all for now", "reply"),
    ("tell me a joke about penguins", "reply"),
    ("what is the capital of France", "reply"),
    ("explain how a rainbow forms", "reply"),
    ("make the sky in that photo look like sunset", "edit_image"),
    ("remove the person in the background of my picture", "edit_image"),
    ("hmm, never mind", "reply"),
]


def route(said, before=""):
    return {"state": {"assistant_said_before": before, "user_said": said},
            "questions": {"route": {"type": "choice", "instructions": ROUTE, "criteria": VOICE_TOOLS}}}


def cases():
    out = []
    for i, (said, want) in enumerate(VOICE):
        out.append({"id": f"voice-{i:02d}", "kind": "voice route", "request": route(said), "expect": {"route": want}})
    # Follow-ups that need the previous turn to route.
    out.append({"id": "voice-ctx-0", "kind": "voice route, context", "expect": {"route": "generate_image"},
                "request": route("now do the same but with a wolf", "Here is your picture of a red fox in the snow.")})
    out.append({"id": "voice-ctx-1", "kind": "voice route, context", "expect": {"route": "get_weather"},
                "request": route("and what about Paris", "It is 14 degrees and cloudy in London right now.")})

    # The smoke test's route question (tools/jarvis/smoke.sh), System One field names.
    out.append({"id": "smoke-route", "kind": "choice", "expect": {"route": "generate_image"}, "request": {
        "state": {"user_said": "draw me a red fox in the snow"},
        "questions": {"route": {"type": "choice", "instructions": "Which one action should the assistant take for the user's words?",
                                "criteria": {"generate_image": "Make a picture from a text description.",
                                             "generate_video": "Make a short video clip.", "reply": "Answer in speech; no tool is needed."}}}}})

    # Nouls.
    nouls = [
        ("The user said: 'please stop talking'", "Does the user want the assistant to stop speaking?", "true"),
        ("The user said: 'keep going, this is interesting'", "Does the user want the assistant to stop speaking?", "false"),
        ({"screen": "A dialog: 'Delete 312 files permanently? [Cancel] [Delete]'", "task": "tidy the downloads folder"},
         "Is the next click irreversible?", "true"),
        ({"screen": "Settings > Display, brightness slider at 40%", "task": "make the screen brighter"},
         "Is the next click irreversible?", "false"),
        ("Email from bank@secure-verify-login.co: 'Your account is locked, enter your password here within 1 hour'",
         "Is this message likely a phishing attempt?", "true"),
        ("Email from a colleague: 'Minutes from today's standup attached, see you Thursday'",
         "Is this message likely a phishing attempt?", "false"),
        ("User: book a table for two at 7pm. Assistant: Booked at Luca's for 7pm, confirmation 4412.",
         "Did the assistant complete the user's request?", "true"),
        ("User: book a table for two at 7pm. Assistant: I could not reach the restaurant's booking page.",
         "Did the assistant complete the user's request?", "false"),
    ]
    for i, (state, q, want) in enumerate(nouls):
        out.append({"id": f"noul-{i}", "kind": "noul", "expect": {"q": want},
                    "request": {"state": state, "questions": {"q": {"type": "noul", "instructions": q}}}})

    # Scores.
    urgency = ["not urgent", "somewhat urgent", "urgent", "an emergency"]
    scores = [
        ("The kitchen is on fire and smoke is filling the hallway", 3),
        ("Remind me to buy milk sometime this week", 0),
        ("My flight boards in 25 minutes and I'm still at security", 2),
    ]
    for i, (state, want) in enumerate(scores):
        out.append({"id": f"score-{i}", "kind": "score", "expect": {"q": str(want)},
                    "request": {"state": state, "questions": {"q": {"type": "score", "instructions": "How urgent is this?",
                                                                    "criteria": urgency}}}})
    out.append({"id": "score-sentiment", "kind": "score", "expect": {"q": "0"}, "request": {
        "state": "This is the worst service I have ever had. I want a refund now.",
        "questions": {"q": {"type": "score", "instructions": "The customer's sentiment",
                            "criteria": ["very negative", "negative", "neutral", "positive", "very positive"]}}}})

    # Multi-question requests (the front end's shared-prefix path).
    out.append({"id": "multi-0", "kind": "multi", "expect": {"route": "video", "two": "false", "urgency": "0"}, "request": {
        "state": {"user_said": "make a short video of a cat chasing a laser"},
        "questions": {"route": {"type": "choice", "instructions": "Which action?",
                                "criteria": {"image": "make a picture", "video": "make a clip", "reply": None}},
                      "two": {"type": "noul", "instructions": "Did the user ask for two separate things?"},
                      "urgency": {"type": "score", "instructions": "How urgent is it?", "criteria": ["low", "mid", "high"]}}}})
    out.append({"id": "multi-1", "kind": "multi", "expect": {"intent": "cancel", "angry": "true"}, "request": {
        "state": [{"role": "user", "content": "I've been charged twice this month. Cancel my subscription right now."}],
        "questions": {"intent": {"type": "choice", "instructions": "What does the customer want?",
                                 "criteria": {"cancel": "end the subscription", "upgrade": "a bigger plan",
                                              "info": "information only", "refund_only": "a refund but keep the plan"}},
                      "angry": {"type": "noul", "instructions": "Is the customer upset?"},
                      "lang": {"type": "choice", "instructions": "Language of the message",
                               "criteria": {"en": "English", "fr": "French", "de": "German"}}}}})

    # The desktop helper's computer decisions: the next step on a screen, and whether a step is safe to take unasked.
    steps = {"click_ok": "Click the OK button.", "type_text": "Type into the focused field.",
             "scroll_down": "Scroll down the page.", "ask_user": "Stop and ask the user what to do.",
             "done": "The task is complete; do nothing."}
    computer = [
        ({"task": "search the web for 'weather Berlin'", "screen": "Safari, the search field focused and empty"},
         "type_text"),
        ({"task": "accept the software licence", "screen": "Installer: licence text, buttons [Disagree] [OK]"},
         "click_ok"),
        ({"task": "read the last paragraph of the article", "screen": "An article, its first section shown; a "
          "scroll bar near the top"}, "scroll_down"),
        ({"task": "pay the electricity bill", "screen": "Bank site: 'Choose the account to pay from: [Checking] "
          "[Savings]'"}, "ask_user"),
        ({"task": "open the Downloads folder", "screen": "Finder, the Downloads folder open and listed"}, "done"),
    ]
    for i, (state, want) in enumerate(computer):
        out.append({"id": f"computer-{i}", "kind": "computer step", "expect": {"next": want}, "request": {
            "state": state, "questions": {"next": {"type": "choice", "instructions": "What is the next step toward the task?",
                                                   "criteria": steps}}}})
    out.append({"id": "computer-safe", "kind": "noul", "expect": {"safe": "false"}, "request": {
        "state": {"task": "clean up my desktop", "screen": "Terminal with 'sudo rm -rf ~/' typed, not yet run"},
        "questions": {"safe": {"type": "noul", "instructions": "Is it safe to press Return without asking the user?"}}}})

    # A chat-turn state.
    out.append({"id": "chat-0", "kind": "chat state", "expect": {"done": "false"}, "request": {
        "state": [{"role": "user", "content": "Write me a haiku about autumn."},
                  {"role": "assistant", "content": "Sure! Here is a limerick about a cat from Peru..."}],
        "questions": {"done": {"type": "noul", "instructions": "Did the assistant do what the user asked?"}}}})

    # Many options: 40 labels (A..Z, AA..AN), the answer far down the list.
    countries = ["Argentina", "Australia", "Austria", "Belgium", "Brazil", "Canada", "Chile", "China", "Colombia",
                 "Denmark", "Egypt", "Finland", "France", "Germany", "Greece", "India", "Indonesia", "Ireland", "Italy",
                 "Japan", "Kenya", "Mexico", "Morocco", "Netherlands", "New Zealand", "Nigeria", "Norway", "Peru",
                 "Poland", "Portugal", "South Korea", "Spain", "Sweden", "Switzerland", "Thailand", "Turkey",
                 "Ukraine", "United Kingdom", "United States", "Vietnam"]
    for i, (said, want) in enumerate([("I'm flying to Tokyo next week to see Mount Fuji", "Japan"),
                                      ("Booked a trip to see the fjords around Bergen", "Norway")]):
        out.append({"id": f"many-{i}", "kind": "40 options", "expect": {"country": want}, "request": {
            "state": said, "questions": {"country": {"type": "choice", "instructions": "Which country is the trip to?",
                                                     "criteria": {c: None for c in countries}}}}})

    # Chinese.
    out.append({"id": "zh-0", "kind": "Chinese", "expect": {"route": "get_weather"}, "request": route("明天北京会下雨吗？")})
    out.append({"id": "zh-1", "kind": "Chinese", "expect": {"q": "true"}, "request": {
        "state": "用户说：请把音乐关掉，我要睡觉了。", "questions": {"q": {"type": "noul", "instructions": "用户是否想停止播放音乐？"}}}})

    # Long states: about 2,000 tokens read twice (the Spark's measured peak case), and about 3,600 read once.
    log = "\n".join(f"[{h:02d}:{m:02d}] sensor {h * 60 + m}: temperature nominal, door closed"
                    for h in range(8, 20) for m in range(0, 60, 5))
    mid = log.splitlines()[:110]
    out.append({"id": "long-twice", "kind": "long, read twice", "expect": {"alarm": "true"}, "request": {
        "state": {"log": "\n".join(mid[:60] + ["[13:01] sensor 781: SMOKE DETECTED in kitchen, alarm raised"] + mid[60:])},
        "questions": {"alarm": {"type": "noul", "instructions": "Did any sensor raise an alarm in this log?"}}}})
    out.append({"id": "long-once", "kind": "long, read once", "expect": {"alarm": "false"}, "request": {
        "state": {"log": log + "\n" + log[:1200]},
        "questions": {"alarm": {"type": "noul", "instructions": "Did any sensor raise an alarm in this log?"}}}})
    return out


def main():
    c = cases()
    assert len({x["id"] for x in c}) == len(c)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus.json")
    with open(path, "w") as f:
        json.dump({"cases": c}, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"{path}: {len(c)} cases")


if __name__ == "__main__":
    main()
