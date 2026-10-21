#!/usr/bin/env bash
# Whole replies from the native Drama 3 server for several texts, each with its silent stretches (ffmpeg silencedetect,
# -45 dB, 0.4 s): a long gap inside a sentence is a degenerate reply, not speech, and inflates the audio length.
#   drama_silence.sh NAME [extra serve flags...]
set -euo pipefail
name=${1:?NAME}; shift
J=$HOME/jarvis; S=$J/scripts; O=$J/out/drama-silence-$name; mkdir -p "$O"
PORT=${PORT:-28793}; URL=http://127.0.0.1:$PORT; MODEL=fishaudio/drama-3
export HF_HOME=$J/hf HF_HUB_CACHE=$J/hf/hub
trap 'systemctl --user stop jarvis-drama 2> /dev/null || true' EXIT
start_s=$(date +%s)
systemd-run --user --unit=jarvis-drama -q -p MemoryMax=64G -p RuntimeMaxSec=3600 -E HF_HOME -E HF_HUB_CACHE \
    "$J/bin/tensorfold" serve $MODEL --name $MODEL --host 127.0.0.1 --port "$PORT" --no-update-check \
    --context 8192 --parallel 1 --no-drafts --prompt-cache-gib 0 --keep-warm 0 "$@"
python3 -I "$S/chat_bench.py" $URL $MODEL ready 900
texts=("The train leaves at seven fifteen."
       "The morning train leaves the station at seven fifteen, crosses the river by the old stone bridge, and reaches the coast just before nine."
       "Good evening. Your calendar is clear tomorrow morning, but you have a dentist appointment at two and dinner with Sam at seven."
       "I found three flights to Lisbon next Friday; the cheapest leaves at six in the morning and costs two hundred and ten euros.")
for i in "${!texts[@]}"; do
    python3 -I "$S/speech_bench.py" $URL $MODEL whole 1 "$O/t$i" "${texts[$i]}" | tee -a "$O/whole.jsonl"
    "$J/bin/ffmpeg" -hide_banner -nostats -i "$O/t$i/whole0.wav" -af silencedetect=n=-45dB:d=0.4 -f null - 2>&1 |
        { grep -oE "silence_duration: [0-9.]+" || true; } | awk -v t="$i" '{s+=$2; if($2>m)m=$2} END{printf "{\"text\": %d, \"silence_s\": %.2f, \"longest_gap_s\": %.2f}\n", t, s, m}' |
        tee -a "$O/silence.jsonl"
done
journalctl --user -u jarvis-drama --since "@$start_s" --no-pager -o cat > "$O/server.log" 2>&1 || true
