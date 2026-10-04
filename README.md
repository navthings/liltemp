# liltemp

is there a law for the best sampling temperature?

small models seem to want a colder temperature than big ones (lilchat went from 8/12 sane replies at 0.7 to 12/12 at 0.4). this measures the best temperature for a bunch of models (pythia 70m to 2.8b, smollm2, lilbase, plus pythia checkpoints mid training) and checks if it lines up with the model's loss.

"best" = the temperature where the model's text is as surprising as real human text, scored in bits per byte by a bigger reference model (qwen2.5-3b). second check: repeated 4-grams vs human text.

## run

```
HOURS=15.5 caffeinate -dims python -u liltemp.py > run.log 2>&1 &
tail -f run.log
```

it goes round by round (100 prompts per model per round), saves after every model, and picks up where it left off if you rerun it. results land in `runs/`.

knobs: `HOURS`, `OUT`, `ONLY=pythia-70m,lilbase`, `CHUNK`, `TEMP_STEP`, `REF`.
