# Working on Shiko

Shiko is an Exile Research project exploring how an agent can remember experiences across sessions, discover associations, and revise what it believes. These instructions apply throughout this repository.

## The Weirdness Charter

**Don’t dismiss the nonsense. Let’s get weird.**

Give a speculative idea room to develop. Biological analogies, strange experiments, and art are welcome here. When moving from an idea to a claim about the software, identify what was imagined, what was implemented, and what was observed. If a result surprises you, preserve the conditions that produced it and investigate. An explanation can make the result more interesting.

## Design commitments

- Build for one main agent per runtime, with its own Shiko engine and data root. A delegated report retains its original actor and source. Keep this model portable across deployment layouts.
- Keep the memory core usable from ordinary Python. Hermes is the first host. Verify lifecycle assumptions against the Hermes revision used for integration and keep host-specific behaviour in the adapter.
- Preserve original evidence through correction, consolidation, and recall. A memory becoming easier to retrieve does not increase the number of independent observations supporting it.
- Use generative models, bounded decision models, or deterministic methods where they suit the operation. Apply durable changes through explicit policy. Let routine updates proceed automatically and preserve uncertainty when the evidence is inconclusive.
- Make failures visible through the public API and diagnostics. Continue useful work where possible. A caller must be able to tell an empty recall result from an unavailable retrieval path and a durable write from a failed one.

## Implementation and evidence

When the maintainer’s local plans are present, follow the numbered milestones in `notes/plans/`. Complete each phase against its goal and acceptance criteria. Run the phase test and the completed regression tests, then append the commands, environment, observed results, and failures to `notes/plans/evidence/milestone-N.md`. Check off a phase after recording its evidence. Record a blocker, dependency, or unresolved choice before moving past affected work.

Keep `notes/` gitignored and out of distributions. The maintainer chooses what evidence to share. Public builds and tests must work from a clean checkout without those files. In public documentation, distinguish demonstrated capabilities from research targets and keep the README focused on the experience of using Shiko.
