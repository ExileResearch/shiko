# Shiko

_思考 / 試行 — Thought and experimentation_

**An adaptive memory engine for AI agents · Exile Research, Softworks Division**

What would you want an agent to remember three weeks after you spent an evening designing a robotic arm together?

Suppose you open a fresh session and say, *“I found a camera for the wrist.”* You never mention motors. The question you’d like to hear back is *“How heavy is it? The wrist stalled with the last camera, and we haven’t tested it since you changed the gearing.”*

That’s the experience Shiko is being built for. An agent should be able to pick up the relevant thread on its own, remember what actually happened, and recognize which part is still an open question.

## What makes Shiko different?

- **One experience has several ways back in.** A part number, a similar failure on another machine, or an unfinished test could all bring the same episode to mind before the agent answers.
- **Changing your mind leaves a history.** When you replace a motor, Shiko is designed to keep the original choice, why you made it, and what prompted the switch. “What’s installed?” and “Why did we try the smaller one?” deserve different answers.
- **A memory carries its source.** The agent can distinguish a load you estimated, a load a tool measured, and a measurement a helper relayed. Retelling the same estimate five times still leaves one estimate.
- **Some connections can form between conversations.** One experiment asks whether Shiko can revisit a week’s worth of design decisions and discover that they all depend on a payload nobody ever specified.

## Where it lives

Shiko is in early development. The first target is a Hermes memory plugin, backed by a portable Python package for custom agent runtimes. Each main agent owns its memory. The design lets you change the model answering your questions while keeping the experiences that give those questions context.

You’re welcome to [download the source](https://github.com/robjvan/shiko/archive/refs/heads/main.zip) and try Shiko as working builds become available. If it recalls something surprising, misses an obvious connection, or breaks, we’d like to hear what happened. [Report a bug or leave feedback](https://github.com/robjvan/shiko/issues).

A user manual will follow. An architecture preprint is planned for Zenodo and aiXiv when the software is ready.
