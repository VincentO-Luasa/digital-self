# Alex Rivera — about me

This is the long version of my story, written for friends and for future me. Everything here
is about Alex Rivera, a fictional data engineer who lives in Porto.

## Career

I have spent my whole working life around data: first as an intern counting things in
spreadsheets, then as an analyst, and now as an engineer who builds the platforms other people
count things with.

### Northwind Labs

I joined Northwind Labs in Porto in February 2023 as a Data Engineer on the Data Platform team.
My first year was mostly about the ingestion framework for the product analytics team: a set of
connectors, a schema registry and a small library that every new pipeline uses to declare its
sources, owners and freshness targets. Before it existed, each analyst wrote their own loader
and nobody knew which numbers were current.

In March 2025 I was promoted to Senior Data Engineer. I now run the platform team's on-call
rotation, review most pipeline designs before they ship, and mentor two junior engineers.
I report to the head of data platform.

What I like most about Northwind Labs is the writing culture: every significant change starts
as a short design note, and decisions are kept in a shared log. I have written more design
notes there than in the rest of my career put together.

- Stack: Python, SQL, Apache Spark, Airflow and dbt on a managed cloud warehouse.
- Team: six engineers and one product manager.
- Office: two days a week in the Porto office, the rest from home.

### Fabrikam Retail

From September 2019 to January 2023 I was a Data Analyst at Fabrikam Retail in Lisbon. I owned
the weekly sales reporting for the Iberian stores and moved it from a tangle of spreadsheets to
version-controlled SQL models, which cut the Monday report from a full day of work to twenty
minutes.

I also built the first stock-forecasting dashboard for the store managers and ran a monthly
"data clinic" where anyone in the company could bring a question about numbers. Fabrikam is
where I learned that a report nobody trusts is worse than no report at all, and it is the job
that pushed me from analysis towards engineering.

### Contoso Analytics

My first job was a summer internship at Contoso Analytics in Madrid in 2017, during my
bachelor's degree. I cleaned survey data and wrote my first Python scripts there, mostly to
avoid copying cells by hand.

## Projects

### Atlas Migration

Since 2026 I have been leading the Atlas Migration at Northwind Labs. We are moving every
nightly batch pipeline from a legacy Hadoop cluster to a managed cloud warehouse, with Airflow
for orchestration and dbt for transformations. The deadline is the end of 2026, when the old
cluster's support contract runs out.

The hard part is not the code but the trust: each migrated pipeline runs in parallel with the
old one for two weeks, and a comparison job checks that both produce the same numbers before
the old one is switched off. So far 140 of roughly 300 pipelines have moved. The finance team
is the main stakeholder, because their month-end close depends on a dozen of these jobs.

### Tidepool

Tidepool is a small open-source library for data quality checks that I started in 2022. It
lets a pipeline declare expectations (row counts, null rates, value ranges) next to the code
that produces the data. I maintain it with my friend Sam Okafor, who wrote most of the
documentation and the plugin system. It has about forty regular users, mostly small teams that
find the big data-quality platforms too heavy.

## Education

I studied Computer Science at Calder College in Seville from 2014 to 2018 (BSc). My final
project was a timetable planner for the college's bus routes.

I then did an MSc in Data Science at Westbridge University from 2018 to 2019. My thesis was
about detecting broken sensors in city air-quality data, and it is where I first used Spark.

In 2024 I earned a cloud data engineering certification, mostly to have a structured reason
to learn the warehouse we were about to migrate to.

## Life outside work

I grew up in Valencia, by the sea, in a family that argued about football at every dinner.
I moved to Seville for college, to Lisbon for my first real job in 2019, and to Porto in 2024,
when Northwind Labs offered me the move. Porto feels like the right size: big enough for good
coffee, small enough to walk everywhere.

Bouldering is my main sport. I climb three evenings a week at a gym near the river, and most
weekends in spring I go to the granite crags in the north with Sam Okafor, who moved to Porto
a year after me. My best grade so far is a 6C, and my goal for 2027 is to climb a 7A outdoors.

I also bake sourdough bread every weekend. My starter is six years old and has survived two
moves. Baking is the one hobby where I follow a recipe exactly, which my friends find funny
for someone who spends the week questioning other people's numbers.

Every morning starts with twenty minutes of calisthenics: push-ups, pull-ups and a few
handstand attempts. I have kept that habit for ten years, through every job and every move.

Since 2024 I have been learning Portuguese with a weekly tutor and a lot of podcasts. I can now
follow a meeting and argue about a restaurant bill, but phone calls are still hard.

On Saturday mornings I volunteer at a coding club for teenagers at the Porto public library,
where we teach Python with small games. It started as a favour to a librarian and became the
part of the week I would least like to give up.

My family is still in Valencia. My older sister teaches maths at a secondary school there, and
every August I spend two weeks at my parents' flat by the beach, mostly eating and losing at
cards to my nephews. We have a family group chat that is ninety percent photos of food.

Travel is the other thing I save up for. Every autumn Sam and I pick one country we have never
visited and spend ten days walking through it with small backpacks: Scotland in 2023, Slovenia
in 2024 and Japan in 2025. The rule is no laptops, which is harder for me than for Sam.

I play the guitar badly and enthusiastically. I took lessons for a year in Lisbon, and now I
mostly play old Spanish pop songs when friends come over for dinner, which is usually a
Friday, and usually involves too much bread.

## What I care about

I value reliable systems, clear writing and teams where knowledge is shared rather than hoarded.
I would rather ship a boring pipeline that runs every night than a clever one nobody can fix.
In the next few years I want to grow into a staff engineer role, write a short book on data
quality for small teams, and spend a month climbing in the Alps.
