// Unit tests for src/utils/workflow-generator.ts and the retry loop in
// src/commands/workflow.ts. The LLM is fully mocked — no network calls.

import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
	buildMessages,
	buildSystemPrompt,
	stripCodeFences,
	generateWorkflow,
	type LlmClient,
	type LlmMessage,
} from "../src/utils/workflow-generator.ts";
import { runGenerate } from "../src/commands/workflow.ts";
import type { SkillMetadata } from "../src/skills.ts";

const SKILLS: SkillMetadata[] = [
	{ name: "gmail", description: "Read and send email", directory: "/x/skills/gmail", filePath: "/x/skills/gmail/SKILL.md" },
	{ name: "slack", description: "Post to Slack", directory: "/x/skills/slack", filePath: "/x/skills/slack/SKILL.md" },
	{ name: "summarize", description: "Summarize text", directory: "/x/skills/summarize", filePath: "/x/skills/summarize/SKILL.md" },
];

const VALID_YAML = `name: morning-digest
description: Summarize unread emails and post to Slack each morning.
steps:
  - skill: gmail
    prompt: Fetch unread emails.
  - skill: summarize
    prompt: Compose a digest.
  - skill: slack
    prompt: Post the digest.
    channel: "#daily-digest"
`;

const INVALID_YAML_MISSING_NAME = `description: no name here
steps:
  - skill: gmail
    prompt: hi
`;

// ── buildSystemPrompt / buildMessages ──────────────────────────────────

test("buildSystemPrompt embeds the schema text and the skill list", () => {
	const sys = buildSystemPrompt(SKILLS);
	assert.ok(sys.includes("<schema>"), "system prompt missing <schema> tag");
	assert.ok(sys.includes('"$id": "https://gitagent.dev/spec/workflow.schema.json"'), "system prompt missing schema $id");
	assert.ok(sys.includes("- gmail: Read and send email"), "system prompt missing gmail skill");
	assert.ok(sys.includes("- slack: Post to Slack"), "system prompt missing slack skill");
	assert.ok(sys.includes("Output ONLY valid YAML"), "system prompt missing rule about raw YAML");
});

test("buildSystemPrompt handles empty skill list with a fallback hint", () => {
	const sys = buildSystemPrompt([]);
	assert.ok(sys.includes("no installed skills detected"), "system prompt missing empty-skills fallback");
});

test("buildMessages includes two few-shot pairs and the user prompt", () => {
	const messages = buildMessages({ prompt: "Do the thing", skills: SKILLS });
	assert.equal(messages[0].role, "system");
	assert.equal(messages[1].role, "user");
	assert.equal(messages[2].role, "assistant");
	assert.equal(messages[3].role, "user");
	assert.equal(messages[4].role, "assistant");
	assert.equal(messages[5].role, "user");
	assert.equal(messages[5].content, "Do the thing");
});

test("buildMessages wraps refine-mode prompts with the previous YAML and instruction", () => {
	const messages = buildMessages({
		prompt: "Add an approval step before the Slack post.",
		skills: SKILLS,
		previousWorkflow: VALID_YAML,
	});
	const last = messages[messages.length - 1];
	assert.equal(last.role, "user");
	assert.ok(last.content.includes("Here is the current workflow"));
	assert.ok(last.content.includes("Add an approval step before the Slack post."));
	assert.ok(last.content.includes("morning-digest"));
	assert.ok(last.content.includes("Return the complete updated workflow as YAML — not a diff."));
});

// ── stripCodeFences ────────────────────────────────────────────────────

test("stripCodeFences removes generic fenced code", () => {
	const input = "```\nname: foo\n```\n";
	assert.equal(stripCodeFences(input), "name: foo");
});

test("stripCodeFences removes yaml-tagged fences", () => {
	const input = "```yaml\nname: foo\n```";
	assert.equal(stripCodeFences(input), "name: foo");
});

test("stripCodeFences leaves unfenced YAML alone", () => {
	const input = "name: foo\n";
	assert.equal(stripCodeFences(input), "name: foo");
});

test("stripCodeFences extracts the block when the LLM wraps it in prose", () => {
	const input = [
		"Here is your workflow:",
		"",
		"```yaml",
		VALID_YAML.trimEnd(),
		"```",
		"",
		"Let me know if you want an approval step added!",
	].join("\n");
	const out = stripCodeFences(input);
	assert.equal(out.startsWith("name: morning-digest"), true, out);
	assert.equal(out.includes("Here is your workflow"), false, out);
	assert.equal(out.includes("Let me know"), false, out);
	assert.equal(out.includes("```"), false, out);
});

// ── generateWorkflow with an injected LLM ──────────────────────────────

test("generateWorkflow returns the LLM output after stripping fences", async () => {
	const captured: { messages: LlmMessage[] } = { messages: [] };
	const llm: LlmClient = async (messages) => {
		captured.messages = messages;
		return "```yaml\n" + VALID_YAML + "```";
	};
	const out = await generateWorkflow({
		prompt: "summarize emails and post to slack",
		skills: SKILLS,
		llm,
	});
	assert.equal(out.trim().startsWith("name: morning-digest"), true);
	assert.equal(captured.messages.length, 6);
	assert.equal(captured.messages[0].role, "system");
});

test("generateWorkflow throws if prompt is empty", async () => {
	await assert.rejects(
		() => generateWorkflow({ prompt: "   ", skills: SKILLS, llm: async () => VALID_YAML }),
		/prompt is required/,
	);
});

// ── Retry loop in runGenerate ──────────────────────────────────────────

test("runGenerate retries when validation fails, then writes the file when the second attempt is valid", async () => {
	const dir = await mkdtemp(join(tmpdir(), "gitagent-test-"));
	try {
		let calls = 0;
		const llm: LlmClient = async (messages) => {
			calls++;
			const userMsg = messages[messages.length - 1].content;
			if (calls === 1) return INVALID_YAML_MISSING_NAME;
			// Second attempt: ensure the retry prompt included the validation error.
			assert.ok(userMsg.includes("schema validation"), `retry user message did not mention validation: ${userMsg}`);
			return VALID_YAML;
		};
		const result = await runGenerate({
			flags: {
				dir,
				prompt: "summarize unread emails and post to Slack",
				dryRun: false,
			},
			llm,
		});
		assert.equal(calls, 2);
		assert.ok(result.filePath, "expected a written file path");
		assert.equal(result.filePath!.endsWith(join("workflows", "morning-digest.yaml")), true, result.filePath);
		const written = await readFile(result.filePath!, "utf-8");
		assert.ok(written.includes("name: morning-digest"));
	} finally {
		await rm(dir, { recursive: true, force: true });
	}
});

test("runGenerate honours --dry-run by returning YAML without writing", async () => {
	const dir = await mkdtemp(join(tmpdir(), "gitagent-test-"));
	try {
		const llm: LlmClient = async () => VALID_YAML;
		const result = await runGenerate({
			flags: { dir, prompt: "x", dryRun: true },
			llm,
		});
		assert.equal(result.filePath, undefined);
		assert.ok(result.yaml.includes("name: morning-digest"));
		// workflows/ must not have been created.
		await assert.rejects(() => readFile(join(dir, "workflows", "morning-digest.yaml"), "utf-8"));
	} finally {
		await rm(dir, { recursive: true, force: true });
	}
});

test("runGenerate gives up after MAX_RETRIES and throws", async () => {
	const dir = await mkdtemp(join(tmpdir(), "gitagent-test-"));
	try {
		let calls = 0;
		const llm: LlmClient = async () => {
			calls++;
			return INVALID_YAML_MISSING_NAME;
		};
		await assert.rejects(
			() => runGenerate({ flags: { dir, prompt: "x", dryRun: true }, llm }),
			/Validation failed after retries/,
		);
		assert.equal(calls, 3); // 1 initial + 2 retries
	} finally {
		await rm(dir, { recursive: true, force: true });
	}
});

test("runGenerate refuses to overwrite an existing workflow unless --force is passed", async () => {
	const dir = await mkdtemp(join(tmpdir(), "gitagent-test-"));
	try {
		const llm: LlmClient = async () => VALID_YAML;
		const first = await runGenerate({ flags: { dir, prompt: "x", dryRun: false }, llm });
		assert.ok(first.filePath);

		await assert.rejects(
			() => runGenerate({ flags: { dir, prompt: "x", dryRun: false }, llm }),
			/already exists/,
		);

		const forced = await runGenerate({ flags: { dir, prompt: "x", dryRun: false, force: true }, llm });
		assert.equal(forced.filePath, first.filePath);
	} finally {
		await rm(dir, { recursive: true, force: true });
	}
});

test("runGenerate rejects a --refine path outside the agent directory", async () => {
	const dir = await mkdtemp(join(tmpdir(), "gitagent-test-"));
	try {
		const llm: LlmClient = async () => VALID_YAML;
		await assert.rejects(
			() => runGenerate({ flags: { dir, prompt: "x", refine: "/etc/hostname", dryRun: true }, llm }),
			/must be inside the agent directory/,
		);
		await assert.rejects(
			() => runGenerate({ flags: { dir, prompt: "x", refine: "../../etc/hostname", dryRun: true }, llm }),
			/must be inside the agent directory/,
		);
	} finally {
		await rm(dir, { recursive: true, force: true });
	}
});

// ── Installed-skill cross-check ────────────────────────────────────────

const UNKNOWN_SKILL_YAML = `name: morning-weather-summary
description: Check the weather each morning and send a text summary.
steps:
  - id: fetch_weather
    skill: weather
    prompt: Fetch today's forecast.
  - id: send_summary
    skill: sms
    prompt: Text a one-sentence summary.
    depends_on: [fetch_weather]
`;

// Materializes real skills/<name>/SKILL.md files so discoverSkills() finds them.
async function withInstalledSkills(names: string[], fn: (dir: string) => Promise<void>): Promise<void> {
	const { mkdir, writeFile } = await import("node:fs/promises");
	const dir = await mkdtemp(join(tmpdir(), "gitagent-skills-"));
	try {
		for (const name of names) {
			await mkdir(join(dir, "skills", name), { recursive: true });
			await writeFile(
				join(dir, "skills", name, "SKILL.md"),
				`---\nname: ${name}\ndescription: Test skill ${name}\n---\n\nDo ${name} things.\n`,
				"utf-8",
			);
		}
		await fn(dir);
	} finally {
		await rm(dir, { recursive: true, force: true });
	}
}

test("runGenerate retries when a step names a skill that is not installed", async () => {
	await withInstalledSkills(["gmail", "slack", "summarize"], async (dir) => {
		let calls = 0;
		let retryPrompt = "";
		const llm: LlmClient = async (messages) => {
			calls++;
			if (calls === 1) return UNKNOWN_SKILL_YAML;
			retryPrompt = messages[messages.length - 1].content;
			return VALID_YAML;
		};
		const result = await runGenerate({ flags: { dir, prompt: "text me the weather", dryRun: true }, llm });
		assert.equal(calls, 2);
		assert.ok(retryPrompt.includes('"weather" is not an installed skill'), retryPrompt);
		assert.ok(retryPrompt.includes('"sms" is not an installed skill'), retryPrompt);
		assert.ok(result.yaml.includes("name: morning-digest"));
	});
});

test("runGenerate fails rather than writing a workflow that references missing skills", async () => {
	await withInstalledSkills(["gmail", "slack", "summarize"], async (dir) => {
		const llm: LlmClient = async () => UNKNOWN_SKILL_YAML;
		await assert.rejects(
			() => runGenerate({ flags: { dir, prompt: "text me the weather", dryRun: false }, llm }),
			/Validation failed after retries/,
		);
		await assert.rejects(() => readFile(join(dir, "workflows", "morning-weather-summary.yaml"), "utf-8"));
	});
});

test("runGenerate --allow-missing-skills writes the workflow anyway", async () => {
	await withInstalledSkills(["gmail", "slack", "summarize"], async (dir) => {
		let calls = 0;
		const llm: LlmClient = async () => {
			calls++;
			return UNKNOWN_SKILL_YAML;
		};
		const result = await runGenerate({
			flags: { dir, prompt: "text me the weather", dryRun: false, allowMissingSkills: true },
			llm,
		});
		assert.equal(calls, 1, "should not retry when the check is disabled");
		assert.ok(result.filePath);
		const written = await readFile(result.filePath!, "utf-8");
		assert.ok(written.includes("skill: weather"));
	});
});

test("runGenerate skips the skill check when no skills are installed", async () => {
	// An agent dir with no skills/ folder is the documented escape hatch: the
	// system prompt tells the model to invent sensible names, so rejecting them
	// here would make generation impossible on a fresh project.
	const dir = await mkdtemp(join(tmpdir(), "gitagent-test-"));
	try {
		let calls = 0;
		const llm: LlmClient = async () => {
			calls++;
			return UNKNOWN_SKILL_YAML;
		};
		const result = await runGenerate({ flags: { dir, prompt: "text me the weather", dryRun: true }, llm });
		assert.equal(calls, 1);
		assert.ok(result.yaml.includes("skill: weather"));
	} finally {
		await rm(dir, { recursive: true, force: true });
	}
});

test("approval is accepted as a pseudo-skill even though it is not installed", async () => {
	await withInstalledSkills(["analytics", "email"], async (dir) => {
		const approvalYaml = `name: daily-sales-report
description: Pull sales data, require approval, then send.
steps:
  - id: pull_data
    skill: analytics
    prompt: Pull yesterday's totals.
  - id: approve
    skill: approval
    prompt: Review the data before it goes out.
    requires_approval: true
    depends_on: [pull_data]
  - skill: email
    prompt: Send the approved report.
    depends_on: [approve]
`;
		let calls = 0;
		const llm: LlmClient = async () => {
			calls++;
			return approvalYaml;
		};
		const result = await runGenerate({ flags: { dir, prompt: "sales report with sign-off", dryRun: true }, llm });
		assert.equal(calls, 1, "approval step should not trigger a retry");
		assert.ok(result.yaml.includes("skill: approval"));
	});
});

test("runGenerate refine mode reads previous YAML and passes it to the LLM", async () => {
	const dir = await mkdtemp(join(tmpdir(), "gitagent-test-"));
	try {
		const { writeFile, mkdir } = await import("node:fs/promises");
		await mkdir(join(dir, "workflows"), { recursive: true });
		const refinePath = join(dir, "workflows", "starter.yaml");
		await writeFile(refinePath, VALID_YAML, "utf-8");

		let observed = "";
		const llm: LlmClient = async (messages) => {
			observed = messages[messages.length - 1].content;
			return VALID_YAML;
		};
		await runGenerate({
			flags: {
				dir,
				prompt: "add an approval step before slack",
				refine: "workflows/starter.yaml",
				dryRun: true,
			},
			llm,
		});
		assert.ok(observed.includes("Here is the current workflow"));
		assert.ok(observed.includes("add an approval step before slack"));
	} finally {
		await rm(dir, { recursive: true, force: true });
	}
});
