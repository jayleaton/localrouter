# Drama 3's named voices

Six fixed voices for `fishaudio/drama-3`, so that every sentence of a reply, and every reply, is spoken by the same
speaker. Drama 3's built-in voice is not one speaker: each prompt lands on its own, so a voice loop that speaks one
sentence a request changes voice every sentence.

Each file is a `drama3-voice/1` reference: a clip's exact transcript and its codec codes, spoken into the prompt
before the text. `fetch.sh` installs them, by the sha256 in `pins.json`, into `ROOT/voices/`, the server's `--voices`
directory.

## Where they come from

Every voice was **invented by Drama 3 itself**. No person's voice was recorded or imitated: they are the model's own
speakers, as the licence's Section IV(b) allows ("a voice the model invented"). Each file's `source` says so
(`model-invented`).

The model's built-in voice spoke 24 assistant-style passages of 10 to 20 s (LocalRouter on DGX-01, 2026-10-22,
drama-tensorfold `1a53b8a`'s native server through the Jarvis daemon). Takes were dropped when:

- Whisper large-v3-turbo's transcript differed from the text by any word, or
- the silence guard had cut them.

Ten takes were left. Six were then picked: the three of each pitch range (F0 under 130 Hz, over 170 Hz) with the
lowest CAM++ speaker-embedding similarity to each other. Each was encoded with upstream's
`tools/zig/drama_voice.py --native` (the pinned engine's `voice encode`, BF16 on the GB10) with its passage as the
transcript.

| Voice | Take | Median F0 | Frames (s) | Passage begins | File sha256 |
|---|---|---|---|---|---|
| Arlo | 10 | 111 Hz | 280 (13.0) | "Let me read that back to make sure I've got it right." | `46598d8ff922cb34ca99952a96d4a63479fe23465a4278f7e3ea92d453a58b8e` |
| Ezra | 13 | 99 Hz | 308 (14.3) | "I'm sorry, I couldn't find anything matching that name in your contacts." | `73ccdf19a2cb014148e1feda9f168dcb2d2fe4ba2ec5801bb555df1a2a2c5265` |
| Theo | 20 | 113 Hz | 431 (20.0) | "Breathe in slowly through your nose, filling your chest and then your stomach." | `6652cc95b220858aab99bb4498c9bb8ebc4365ccb27eb3b808ebd5db14c62acc` |
| Mae | 24 | 177 Hz | 296 (13.7) | "That's a lovely idea." | `f0882e7af17aaf85f62403c0332e830b0028472ee84de06f47e61e273558ef50` |
| Nora | 14 | 215 Hz | 349 (16.2) | "Here's an interesting fact for you. Honey never really spoils." | `a6bb78e495dbac589c6213616a1c54fc7cc9d2b983a10b51b697ae87059991de` |
| Wren | 19 | 224 Hz | 426 (19.8) | "In the late nineteenth century, the town was little more than a fishing harbour" | `d972d6c33fb6bf51f4b0bd012c0d847cd9f0d66bfb64d9385ba50b343c64d6d7` |

The names are invented and avoid Kitten TTS 2's voice names, so one `voice` names one speaker across the box's
speech models.

These files are Drama 3 outputs: speech codes, not weights. The Fish Audio Community License covers them as it covers
any output of the model; keep its attribution ("Built with Fish Audio") wherever they are used.
