# Realtime with your own tools

`client.py` is a harness in one file (Python 3 standard library): it opens `/v1/realtime`, asks for client tool
execution, registers one local function and runs it when the model calls it. The box does the listening, thinking and
speaking; the tool runs on the device that owns it.

```sh
python3 -I client.py --url http://<box>:8190                 # one question, its call, the spoken answer
python3 -I client.py --url http://<box>:8190 --turns 3 --json out.json --wav answer.wav
LOCALROUTER_TOKEN=<token> python3 -I client.py --url ...     # with device pairing on
```

What it sends and receives:

```text
-> session.update        {localrouter: {tool_execution: "client"}, tools: [get_device_time], turn_detection: null}
<- session.updated       {localrouter: {tool_execution: "client"}, ...}
-> conversation.item.create (the question), response.create
<- response.function_call_arguments.done {call_id, name: "get_device_time", arguments}
<- response.done
   (runs get_device_time here, once)
-> conversation.item.create {type: "function_call_output", call_id, output}, response.create
<- response.audio.delta ... response.done              (the answer, spoken)
```

It then sends a second output for the same call and one for a call nobody made, and expects `duplicate_call_output`
and `unknown_call_id`. Per turn it prints the time from sending the output to the first audio of the answer: the cost
of the round trip through your tool, on top of the tool's own run time.

A real harness also listens for `localrouter.function_call.cancelled`: the user spoke again or another response began
before the output came, so the server will refuse it. Stop the work then. The protocol is in
[VOICE.md](../../docs/dev/VOICE.md#client-tool-execution).
