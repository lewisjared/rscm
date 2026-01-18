---
name: OpenSpec: Proposal
description: Investigate a feature idea thoroughly, then scaffold an OpenSpec change after approval.
category: OpenSpec
tags: [openspec, change, proposal]
---
<!-- OPENSPEC:START -->
**Guardrails**

- Favor straightforward, minimal implementations first and add complexity only when it is requested or clearly required.
- Keep changes tightly scoped to the requested outcome.
- Refer to `openspec/AGENTS.md` for additional OpenSpec conventions or clarifications.
- Do not write any code during the proposal stage. Only create design documents (proposal.md, tasks.md, design.md, and spec deltas). Implementation happens in the apply stage after approval.

---

## Phase 1: Investigation

**Goal:** Deeply understand the problem space and codebase before committing to any specification. This phase produces an investigation document for user review.

**Steps:**

1. **Orient to the request**
   - Read `openspec/project.md` for project context and constraints
   - Run `openspec list` and `openspec list --specs` to understand current state
   - Identify any immediately obvious ambiguities and ask 1-2 clarifying questions if critical

2. **Deep codebase exploration**
   Use the Explore agent (via Task tool with subagent_type=Explore) to thoroughly investigate:
   - Existing patterns and conventions relevant to the feature
   - Code that will be affected or extended
   - Similar implementations that could serve as templates
   - Potential edge cases or conflicts with existing functionality

   Search for related requirements: `rg -n "Requirement:|Scenario:" openspec/specs`

3. **Document findings**
   Create `openspec/changes/<change-id>/investigation.md` with:

   ```markdown
   # Investigation: <feature idea>

   ## Context
   [What the user is asking for and why it matters]

   ## Codebase Analysis

   ### Relevant Existing Code
   - `path/to/file.rs:123` - [what it does, why it's relevant]
   - ...

   ### Patterns to Follow
   [Existing patterns this feature should align with]

   ### Potential Conflicts or Concerns
   [Anything that might complicate implementation]

   ## Approaches Considered

   ### Approach A: <name>
   **Description:** [1-2 sentences]
   **Pros:**
   - ...
   **Cons:**
   - ...
   **Estimated scope:** [small/medium/large]

   ### Approach B: <name>
   ...

   ### Approach C: <name> (if applicable)
   ...

   ## Open Questions
   - [ ] Question requiring user input?
   - [ ] Design decision that needs clarification?

   ## Recommendation
   [Which approach and why, or "need user input on X before recommending"]
   ```

4. **Present investigation to user**
   - Summarize key findings
   - Present the approaches with clear trade-offs
   - List open questions that need answers
   - Ask user to choose an approach or provide guidance

   **STOP HERE.** Do not proceed to Phase 2 until the user has:
   - Reviewed the investigation
   - Answered open questions
   - Approved an approach direction

---

## Phase 2: Specification

**Goal:** Transform the approved approach into formal OpenSpec artifacts.

**Prerequisites:** User has approved an approach from Phase 1.

**Steps:**

5. **Create proposal.md**
   Based on the approved approach from investigation.md:

   ```markdown
   # Change: <Brief description>

   ## Why
   [1-2 sentences on problem/opportunity - from investigation context]

   ## What Changes
   - [Bullet list of changes]
   - [Mark breaking changes with **BREAKING**]

   ## Chosen Approach
   [Which approach was selected and key rationale - reference investigation.md]

   ## Impact
   - Affected specs: [list capabilities]
   - Affected code: [key files/systems - from investigation]
   ```

6. **Create design.md** (if needed)
   Include design.md if the change involves:
   - Cross-cutting concerns (multiple modules/crates)
   - New architectural patterns
   - External dependencies or data model changes
   - Security, performance, or migration complexity

   Structure:
   ```markdown
   ## Context
   [From investigation - constraints, stakeholders]

   ## Decision
   [Chosen approach and detailed rationale]

   ## Alternatives Rejected
   [Other approaches from investigation and why not chosen]

   ## Trade-offs Accepted
   [Explicit acknowledgment of cons from chosen approach]

   ## Migration Plan
   [If applicable]
   ```

7. **Create spec deltas**
   In `openspec/changes/<change-id>/specs/<capability>/spec.md`:
   - Map the change to concrete requirements
   - Use `## ADDED|MODIFIED|REMOVED|RENAMED Requirements`
   - Include at least one `#### Scenario:` per requirement
   - Cross-reference related capabilities when relevant

8. **Create tasks.md**
   Ordered list of small, verifiable work items:
   - Each task should deliver visible progress
   - Include validation steps (tests, checks)
   - Note dependencies between tasks
   - Mark parallelizable work

   ```markdown
   ## 1. <Category>
   - [ ] 1.1 <Specific task>
   - [ ] 1.2 <Specific task>

   ## 2. <Category>
   - [ ] 2.1 <Specific task>
   ...
   ```

9. **Validate**
   Run `openspec validate <change-id> --strict --no-interactive`
   Resolve all issues before presenting the complete proposal.

10. **Present complete proposal for final approval**
    Summarize the full proposal and ask user to confirm before implementation.

---

**Reference**

- Use `openspec show <id> --json --deltas-only` to inspect delta details
- Use `openspec show <spec> --type spec` to view existing specs
- Explore agent: `Task tool with subagent_type=Explore` for codebase investigation
- Search requirements: `rg -n "Requirement:|Scenario:" openspec/specs`
<!-- OPENSPEC:END -->
