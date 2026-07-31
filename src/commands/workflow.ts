import { existsSync } from "fs";
import { mkdir, readFile, writeFile } from "fs/promises";
import { join, resolve, sep } from "path";
import { discoverSkills } from "../skills.js";
import { validateSkillReferences, validateWorkflow } from "../utils/schemas.js";
import { generateWorkflow, type LlmClient } from "../utils/workflow-generator.js";

interface GenerateFlags {
	dir: string;
	prompt?: string;
	refine?: string;
	model?: string;
	apiKey?: string;
	dryRun: boolean;
	force?: boolean;
	allowMissingSkills?: boolean;
}

const RED = (s: string) => `\x1b[31m${s}\x1b[0m`;
const GREEN = (s: string) => `\x1b[32m${s}\x1b[0m`;
const DIM = (s: string) => `\x1b[2m${s}\x1b[0m`;
const BOLD = (s: string) => `\x1b[1m${s}\x1b[0m`;

const MAX_RETRIES = 2;

function printHelp(): void {
	console.log(`${BOLD("gitagent workflow")} — generate SkillFlow workflows from natural language

Usage:
  gitagent workflow generate [options]

Options:
  -d, --dir <path>        Agent directory (default: current directory)
  -p, --prompt <text>     Natural-language description of the workflow (required)
      --refine <file>     Refine an existing workflow YAML by applying --prompt as an instruction
                          (must be a path inside the agent directory)
  -m, --model <spec>      LLM model in provider:model form (default: openai:gpt-4o)
      --api-key <key>     API key for the provider (falls back to OPENAI_API_KEY or <PROVIDER>_API_KEY)
      --dry-run           Print the generated YAML to stdout instead of writing a file
  -f, --force             Overwrite workflows/<name>.yaml if it already exists
                          (without this flag an existing file is never replaced)
      --allow-missing-skills
                          Accept steps that reference skills which are not installed
                          (by default generation fails rather than writing a workflow
                          that cannot run)
  -h, --help              Show this help message

Examples:
  gitagent workflow generate -p "every morning summarize unread emails and post to Slack"
  gitagent workflow generate -p "add a human approval step before the Slack post" --refine workflows/morning-digest.yaml
`);
}

// Reads the value that follows a flag, erroring out when the flag is the last
// token. Without this, argv[++i] yields undefined and the failure surfaces far
// from the parse site (e.g. resolve(undefined) throwing a bare TypeError).
function valueFor(argv: string[], i: number, flag: string): string {
	const v = argv[i + 1];
	if (v === undefined) {
		console.error(RED(`${flag} requires a value`));
		process.exit(2);
	}
	return v;
}

function parseFlags(argv: string[]): GenerateFlags {
	const flags: GenerateFlags = { dir: process.cwd(), dryRun: false };
	for (let i = 0; i < argv.length; i++) {
		const a = argv[i];
		switch (a) {
			case "-d":
			case "--dir":
				flags.dir = valueFor(argv, i++, a);
				break;
			case "-p":
			case "--prompt":
				flags.prompt = valueFor(argv, i++, a);
				break;
			case "--refine":
				flags.refine = valueFor(argv, i++, a);
				break;
			case "-m":
			case "--model":
				flags.model = valueFor(argv, i++, a);
				break;
			case "--api-key":
				flags.apiKey = valueFor(argv, i++, a);
				break;
			case "--dry-run":
				flags.dryRun = true;
				break;
			case "-f":
			case "--force":
				flags.force = true;
				break;
			case "--allow-missing-skills":
				flags.allowMissingSkills = true;
				break;
			case "-h":
			case "--help":
				printHelp();
				process.exit(0);
				break;
			default:
				if (!a.startsWith("-") && flags.prompt === undefined) {
					flags.prompt = a;
				} else {
					console.error(RED(`Unknown option: ${a}`));
					process.exit(2);
				}
		}
	}
	return flags;
}

function slugify(name: string): string {
	const cleaned = name
		.toLowerCase()
		.trim()
		.replace(/[^a-z0-9-]+/g, "-")
		.replace(/^-+|-+$/g, "")
		.replace(/-+/g, "-");
	return cleaned || "workflow";
}

export interface RunGenerateOptions {
	flags: GenerateFlags;
	llm?: LlmClient;
}

export async function runGenerate(opts: RunGenerateOptions): Promise<{ filePath?: string; yaml: string; }> {
	const { flags } = opts;
	if (!flags.prompt || !flags.prompt.trim()) {
		throw new Error("--prompt is required");
	}

	const agentDir = resolve(flags.dir);
	const skills = await discoverSkills(agentDir);
	const skillNames = skills.map((s) => s.name);

	let previousWorkflow: string | undefined;
	if (flags.refine) {
		// resolve() ignores the base when the second argument is absolute, so an
		// absolute path or a ../ traversal would otherwise read (and ship to the
		// LLM) any file on disk. Keep --refine inside the agent directory.
		const refinePath = resolve(agentDir, flags.refine);
		if (refinePath !== agentDir && !refinePath.startsWith(agentDir + sep)) {
			throw new Error(`--refine path must be inside the agent directory: ${refinePath}`);
		}
		previousWorkflow = await readFile(refinePath, "utf-8");
	}

	let promptForLlm = flags.prompt.trim();
	let lastErrors: string[] = [];
	let yaml = "";

	for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
		console.error(DIM(attempt === 0 ? "Generating workflow..." : `Retry ${attempt}/${MAX_RETRIES} — fixing validation errors...`));
		yaml = await generateWorkflow({
			prompt: promptForLlm,
			skills,
			previousWorkflow,
			model: flags.model,
			apiKey: flags.apiKey,
			llm: opts.llm,
		});
		const result = validateWorkflow(yaml);
		// A schema-valid workflow can still name skills that aren't installed,
		// which would only surface as a run-time failure. Fold that into the same
		// retry loop so the model gets a chance to pick real skills.
		const skillErrors = result.valid && !flags.allowMissingSkills
			? validateSkillReferences(result.data!, skillNames)
			: [];
		if (result.valid && skillErrors.length === 0) {
			lastErrors = [];
			break;
		}
		lastErrors = [...result.errors, ...skillErrors];
		if (attempt < MAX_RETRIES) {
			promptForLlm =
				`${flags.prompt.trim()}\n\nThe previous attempt failed schema validation or referenced skills that are not installed. Fix these errors and return the full YAML again:\n` +
				lastErrors.map((e) => `- ${e}`).join("\n");
		}
	}

	if (lastErrors.length > 0) {
		console.error(RED("\nWorkflow validation failed after retries:"));
		for (const e of lastErrors) console.error(RED(`  - ${e}`));
		if (lastErrors.some((e) => e.includes("is not an installed skill"))) {
			console.error(DIM(`\nInstalled skills: ${skillNames.join(", ") || "(none)"}`));
			console.error(
				DIM(
					"Create the missing skill(s) under skills/ first, or re-run with --allow-missing-skills to write the workflow anyway.",
				),
			);
		}
		console.error(DIM("\nLast generated YAML:\n"));
		console.error(yaml);
		throw new Error("Validation failed after retries");
	}

	if (flags.dryRun) {
		process.stdout.write(yaml.endsWith("\n") ? yaml : yaml + "\n");
		return { yaml };
	}

	// Parse the validated YAML to get the workflow name for the file path.
	const validated = validateWorkflow(yaml).data!;
	const slug = slugify(validated.name);
	const workflowsDir = join(agentDir, "workflows");
	const filePath = join(workflowsDir, `${slug}.yaml`);
	if (!flags.force && existsSync(filePath)) {
		throw new Error(
			`${filePath} already exists. Re-run with --force to overwrite it, or use --dry-run to preview the generated workflow.`,
		);
	}
	await mkdir(workflowsDir, { recursive: true });
	await writeFile(filePath, yaml.endsWith("\n") ? yaml : yaml + "\n", "utf-8");
	console.error(GREEN(`\nWrote workflow to ${filePath}`));
	return { filePath, yaml };
}

export async function handleWorkflowCommand(argv: string[]): Promise<void> {
	// argv is the raw process.argv tail starting at the 'workflow' token.
	// argv[0] === "workflow"; argv[1] is the sub-command.
	const sub = argv[1];
	if (!sub || sub === "-h" || sub === "--help") {
		printHelp();
		return;
	}
	if (sub !== "generate") {
		console.error(RED(`Unknown subcommand: ${sub}`));
		printHelp();
		process.exit(2);
	}
	const flags = parseFlags(argv.slice(2));
	try {
		await runGenerate({ flags });
	} catch (err: any) {
		console.error(RED(`\nError: ${err?.message ?? String(err)}`));
		process.exit(1);
	}
}
