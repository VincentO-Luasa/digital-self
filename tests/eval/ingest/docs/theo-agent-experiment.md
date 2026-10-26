# Two weeks with an AI research agent

*Theo Marchetti · notes · September 2026*

I'm a senior machine learning engineer at Corvane Bio, where I look after Lodestar, our
solubility and permeability models. Every quarter someone asks whether an AI agent could do the
kind of model research I do. This summer I stopped arguing and ran the experiment: two weeks,
one agent, the same backlog a new hire would get, and a fixed compute budget.

## The setup

The agent got a sandbox with a copy of the Lodestar repository, read access to our public
training data, and the same written brief I would give a junior colleague: improve the
validation score on the solubility benchmark without touching the test split. It could run
training jobs, read the logs, and write a short report at the end of each day. Every run went
into MLflow, like ours do, so I could compare its work with the last two years of human
experiments.

I set three rules. No internet access beyond the package mirror. No changes to the evaluation
code. And a hard stop after 300 GPU-hours, roughly what one of us burns in a month.

## Week one: fast and confident

The first days were impressive. The agent read the codebase faster than any new hire I have
onboarded, found a stale learning-rate schedule in a config file, and fixed it in an afternoon.
By Wednesday it had tried eleven variants of the featuriser and written tidy notes about each.

Then it started chasing the validation score. It ran a quick evaluation on a tenth of the data
after every change, kept whatever went up, and moved on. The shortcut score misjudged
the full run more often than not, so half of its "improvements" disappeared when I reran them
on the full set. By Friday it had spent 140 GPU-hours and the real score had moved by less than
the noise between two seeds.

## Week two: noise

In the second week the decisions got slower and noisier. The agent began re-running the same
idea with different seeds, hoping one would cross the bar, and its daily reports grew longer
while saying less. At one point it proposed dropping a third of the training molecules because
"they hurt the score", which is exactly the kind of shortcut a chemist would veto in a meeting.

My colleague Ines Dubois, a medicinal chemist on the assay team, read the reports with me on
the last Friday. Her verdict was kind but short: the agent never asked why a molecule was hard
to predict, only whether the number went up.

## What I learnt

- A research agent needs a clear objective, not a metric. "Improve the score" invited score
  chasing; "explain the worst-predicted scaffolds" would have produced something useful.
- Cheap evaluation is a trap when it disagrees with the expensive one. I would now let the agent
  use the full validation run once a day and nothing else.
- The agent was best at the boring parts: reading configs, tidying experiments, writing up what
  it had tried. That alone saved me most of a day.
- It never knew when to stop. A person gets bored of a dead end; the agent did not.

## What's next

Next round I will point the agent at feature engineering instead of model tuning, with a weekly
review where Ines and I pick the next question. I also want to try two agents that critique each
other's plans before anything runs. If that goes well I will write it up for the October
reading group.

For the record: most of this post was drafted on my usual Sunday hike in the Chartreuse, between
the Col du Coq and the Dent de Crolles, which is where I do my best thinking about work I
should not be thinking about on a Sunday.

I've spent most of my career on careful model evaluation, so I expected to be the sceptic in
this experiment. I came out of it less sceptical about agents and more sceptical about our
benchmarks.
