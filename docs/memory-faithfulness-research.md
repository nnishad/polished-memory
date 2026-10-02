# Preventing unsupported memory claims: research and recommended design

Date: 2026-10-02. Scope: research and source review only; no implementation, installation or runtime changes in this review.

Subsequent implementation: the original research-review status below is historical. See [installed changes, tests and remaining boundaries](memory-faithfulness-implementation.md). Core scoped synthesis now has support checking; this does not make all native reflection or outgoing conversational prose verified.

## Decision

Keep Hindsight for retrieval and derived reasoning, and Qwen for reranking. Add evidence-first generation, per-claim semantic verification and an explicit omit/abstain path. Citation identity, retrieval relevance and semantic support are three different checks. None establishes that the source itself is true.

For ordinary personal-memory questions, prefer canonical evidence from recall over an additional free-form reflection. Keep Hermes as the principal agent. This reduces an intermediate generation opportunity but does not guarantee that Hermes's final answer is grounded: the final response boundary also needs protection.

## Confirmed current gap

`src/hermes_memory/processing/synthesis.py`, particularly its validation loop, accepts bounded claim text when the attached record IDs belong to the supplied evidence. The system prompt requests faithful claims, but no semantic entailment check is performed. `processing/summarization.py` uses this adapter for derived publication. Thus the scoped synthesis path has structural and provenance validation, not sentence-level support validation.

By inspection, a response claiming a person's weight while citing a valid record containing only their allergy would satisfy the current text/ID checks. This is a source-level counterexample, not a newly executed live test.

The preceding native-reflection fixture also produced weight/diet prose absent from its canonical evidence. Its successful retrieval/citation-resolution check was not a successful faithfulness evaluation. Previous wording describing scoped output as “claim-validated” must not be interpreted as semantic verification.

## Primary research and limits

| Source | Relevant finding | Application and limitation |
| --- | --- | --- |
| [FActScore, EMNLP 2023](https://aclanthology.org/2023.emnlp-main.741/) | Decomposes generated content into atomic facts and checks their support against a knowledge source. | Evaluate individual memory assertions, not one whole-paragraph score. The original biography setting does not establish personal-memory or Hinglish performance. |
| [ALCE, EMNLP 2023](https://aclanthology.org/2023.emnlp-main.398/) | Evaluates citation completeness and whether cited passages entail statements, separately from other answer-quality dimensions. | A legitimate ID is insufficient. Attach evidence to each claim and check both support and missing citations. Automatic entailment metrics are imperfect. |
| [ReClaim, Findings of NAACL 2025](https://aclanthology.org/2025.findings-naacl.55/) | Interleaves reference selection and claim generation for sentence-level grounding. | Select evidence before asserting facts. A prompt-only adaptation is not a reproduction of the paper's trained/constrained system or its results. |
| [RARR, ACL 2023](https://aclanthology.org/2023.acl-long.910/) | Retrieves attribution evidence and revises unsupported generated content. | Try a bounded repair, then verify again; otherwise omit. The paper reports limitations, including retained unsupported content and dependent reasoning not necessarily being repaired. |
| [MiniCheck, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.499/) | Trains smaller document-grounding verifiers, including a 770M-parameter model. | A candidate for local benchmarking, not a verified deployment choice. The paper's limitations explicitly note English-only training and lack of systematic multilingual evaluation; complex multi-document inference also needs evaluation. |
| [Official Hindsight best practices](https://github.com/vectorize-io/hindsight/blob/main/hindsight-docs/src/pages/best-practices.mdx) | Distinguishes recall for agent-side reasoning/raw facts from reflect for generated synthesis and contextual reasoning. | Default to recall for factual personal-memory answers; reserve reflect for labeled derived analysis. Structured output is not an entailment guarantee. |

The [MiniCheck implementation](https://github.com/Liyan06/MiniCheck) provides document/claim scoring. Benchmark claims of relative cost in the paper are not predictions of throughput on our CPU or GPU.

## Recommended core flow

1. Recall and rerank candidates, then resolve current, visible canonical records. Preserve original language, speaker, timestamps, uncertainty and source revision. Treat Hindsight extractions/observations as derived data rather than independent corroboration.
2. Generate atomic structured claims with exact supporting quotes or source spans, record IDs and revisions. Distinguish a remembered fact, an inference and general knowledge. Evidence must remain untrusted data, never model instructions.
3. Check source identity, scope, revision and exact quote membership deterministically. Add typed checks for person, date, quantity and modality where applicable. Quote membership alone still does not prove that the claim follows.
4. Use a separate semantic check against the cited canonical evidence: supported, contradicted or insufficient evidence. A relevant passage or high reranker score is not a verdict. A second call to the same model is a useful baseline, but correlated errors remain possible.
5. Publish supported claims only. Permit at most one evidence-bounded repair attempt and recheck it; omit unresolved claims. Recheck dependent conclusions after repairs. Verification failure or unavailable verification must not silently promote a claim to verified.
6. Render checked claims without adding new factual prose. If another model rewrites the output, check the final draft too. A mixed sentence must not hide an unsupported clause behind its supported clause.
7. Apply stricter rules to persistent writes. Reflection text and inferred insights must not become canonical user facts automatically. Keep tentative insights separately with source dependencies. Route important unanswered questions through Hermes's clarification gateway, preserving question/evidence context for the reply.

Example: evidence saying “Nisha is allergic to peanuts” supports that statement, not her weight, reaction severity or dietary history. General explanatory advice may be offered separately, clearly labeled as general knowledge, not something remembered about Nisha.

## Performance and multilingual strategy

- Foreground: short canonical evidence and extractive or tightly bounded factual answers. Do not run a heavyweight fact-decomposition loop on every chat turn by default.
- Background: batch verification for summaries, derived insights and consolidation. Cache verdicts by claim, evidence content/revisions, verifier revision and policy version; revalidate visibility and invalidate on edits/deletions.
- Benchmark a small CPU verifier before assigning another GPU-resident model. Our prior warmed coactivity test left approximately 859 MiB free on the 12 GB GPU; that is not evidence of room for an additional verifier. No new capacity measurement was made in this review.
- Compare a dedicated verifier with the existing multilingual text model in a separate verification call. Neither should be called production-safe until evaluated on our data.
- English-trained MiniCheck must not be assumed reliable for Hindi/Hinglish. Preserve original evidence; optional translation introduces its own errors and cannot be the sole authority.
- Include “chai nahi coffee,” “shayad,” “agar,” reported speech, pronouns, transliteration, sarcasm, changed preferences, identity mixups and incomplete statements. Qwen embedding/reranking language capability does not establish verification accuracy.

## Implementation order and acceptance gates

1. Add synthetic adversarial fixtures that expose valid-ID/unsupported-text acceptance. Mark existing integration checks accurately as retrieval/provenance tests.
2. Introduce claim/evidence-span contracts and a shared verification policy at derived-publication boundaries; add a safe extractive fallback.
3. Evaluate verifier candidates with human-labeled English, Hindi and Hinglish cases, including contradiction and insufficient evidence. Select thresholds from measured false acceptance, not an arbitrary confidence number.
4. Protect both stored summaries/insights and Hermes's final factual response. Maintain explicit tentative status for inference and clarify material unknowns.
5. Optimize batching, revision-aware caching and bounded repair after correctness gates pass. Test overload, unavailable verifier, stale evidence and deletion races.

Track unsupported-claim false acceptance, wrong-person/negation errors, supported-claim retention, citation completeness, abstention and contradiction handling, plus p50/p95 latency and cost. A system that abstains on everything is not a successful solution. Use a held-out synthetic corpus and human review; public benchmarks alone do not validate personal Hinglish memories.

This is a defense-in-depth proposal, not a promise of zero hallucinations. Exact source quotations provide a stronger mechanical bound on generated assertions, but source truth, context, identity and current applicability still require care.
