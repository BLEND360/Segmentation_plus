"""LLM persona generation and business objective grounding."""
from __future__ import annotations

import logging
import re

import pandas as pd

from .config import PipelineConfig
from .ollama_client import OllamaClient
from .types import (
    BusinessGrounding,
    DataSchema,
    EvaluationResult,
    ExplainabilityReport,
    PersonaResult,
)

log = logging.getLogger("segplus.persona_generation")

CLUSTER_LETTERS = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H"}

PERSONA_SYSTEM_PROMPT = """You are a senior customer analytics expert and strategic advisor.
You analyse customer cluster profiles and generate precise, actionable personas.
Always respond with ONLY valid JSON - no preamble, no explanation, no markdown fences.

CRITICAL NAMING RULES:
- The archetype MUST be a unique, descriptive 2-5 word business label.
- Base the name on the ACTUAL DATA VALUES shown (high/low/above/below average), not just feature column names.
- NEVER use generic names like 'Segment A', 'Cluster Profile', 'Credit Score Optimizers', or names that merely repeat feature column names.
- NEVER include raw column names like 'credit_score' or 'risk_score' directly in the archetype.
- Good examples: 'Premium Low-Risk Elite', 'Budget-Conscious Rebuilders', 'High-Value Dormant Accounts', 'Creditworthy Loyalists', 'At-Risk High-Spenders'
- The name should instantly convey WHO these customers are to a business stakeholder.

The JSON must contain exactly these keys:
  archetype: string (2-5 word evocative business label reflecting this cluster's actual value profile)
  description: string (2-3 sentences describing who this customer is, referencing specific data patterns)
  key_traits: list of 4 strings (concise behavioural characteristics derived from the data)
  business_recommendations: list of 3 strings (specific, actionable strategies tied to this cluster's profile)
"""

GROUNDING_SYSTEM_PROMPT = """You are a Chief Strategy Officer synthesising customer segmentation results
into an executive-level strategic report.
Respond with ONLY valid JSON containing:
  executive_summary: string (3-4 sentences - overall findings and strategic implication)
  cluster_priority: list of objects, each with:
    cluster: string
    archetype: string
    priority: string ("High" | "Medium" | "Low")
    strategic_action: string (one concrete next action)
  cluster_actions: list of objects, each with:
    cluster: string
    grounded_persona_name: string (business-ready segment label)
    grounded_recommendations: list of 3 strings (specific actions for this cluster)
  quick_wins: list of 3 strings (immediate actions any team can take this week)
"""


def build_cluster_profile_text(
    cluster_letter: str,
    profile_row: pd.Series,
    cluster_pct: float,
    top_features: dict[str, float],
    top_feature_names: list[str],
    business_objective: str,
    global_means: pd.Series | None = None,
    global_stds: pd.Series | None = None,
) -> str:
    """Build a rich textual summary of a cluster's statistical profile with population comparison."""
    lines = [
        f"CLUSTER {cluster_letter} PROFILE",
        f"  Size: {int(cluster_pct * 100)}% of total customers",
        "",
        f"  Business Objective: {business_objective}",
        "",
        "  Key metrics (cluster mean vs population mean):",
    ]
    for feat in top_feature_names[:8]:
        if feat in profile_row.index:
            val = profile_row[feat]
            if global_means is not None and feat in global_means.index:
                pop_mean = float(global_means[feat])
                if pop_mean != 0:
                    pct_diff = ((val - pop_mean) / abs(pop_mean)) * 100
                    direction = "above" if val > pop_mean else "below"
                    lines.append(
                        f"    - {feat}: {val:.2f} ({direction} pop. avg {pop_mean:.2f}, {pct_diff:+.0f}%)"
                    )
                else:
                    lines.append(f"    - {feat}: {val:.2f} (pop. avg {pop_mean:.2f})")
            else:
                lines.append(f"    - {feat}: {val:.2f}")

    lines += ["", "  Feature importances (contribution to cluster separation):"]
    for feat, imp in list(top_features.items())[:5]:
        lines.append(f"    - {feat}: {imp:.4f}")

    return "\n".join(lines)


class PersonaGenerator:
    """Generate LLM-powered personas for each cluster and ground in business objective."""

    def __init__(self, client: OllamaClient, config: PipelineConfig):
        self.client = client
        self.config = config
        self._is_available: bool | None = None

    def _check_availability(self) -> bool:
        if self._is_available is None:
            self._is_available = self.client.health_check()
        return self._is_available

    def generate_personas(
        self,
        evaluation: EvaluationResult,
        explainability: ExplainabilityReport,
        raw_df: pd.DataFrame,
        schema: DataSchema,
    ) -> list[PersonaResult]:
        """For each cluster: build profile -> call LLM -> parse JSON -> create PersonaResult."""
        use_fallback = not self._check_availability()
        labels = evaluation.labels
        total = int((labels != -1).sum())
        cluster_ids = sorted(c for c in set(labels) if c != -1)

        num_cols = [c for c in schema.numeric_cols if c in raw_df.columns]
        global_means = raw_df[num_cols].mean() if num_cols else pd.Series(dtype=float)
        global_stds = raw_df[num_cols].std().replace(0, 1.0) if num_cols else pd.Series(dtype=float)
        self._global_means = global_means
        self._global_stds = global_stds
        personas: list[PersonaResult] = []

        for cid in cluster_ids:
            mask = labels == cid
            size = int(mask.sum())
            pct = size / total
            letter = CLUSTER_LETTERS.get(cid, str(cid))
            persona_name = f"Cluster {letter}"

            # Compute profile from raw data
            profile_row = raw_df[mask][num_cols].mean()
            top_features_dict = {
                k: v for k, v in explainability.feature_importances.items() if v > 0
            }
            cluster_top_features = self._cluster_specific_feature_order(
                profile_row=profile_row,
                global_means=global_means,
                global_stds=global_stds,
                candidate_features=list(top_features_dict.keys()),
            )

            if use_fallback:
                persona = self._build_fallback_persona(
                    cid, persona_name, size, pct, profile_row, top_features_dict, cluster_top_features
                )
            else:
                persona = self._generate_llm_persona(
                    cid, letter, persona_name, size, pct,
                    profile_row, top_features_dict, cluster_top_features,
                )

            personas.append(persona)
            log.info("Persona generated: %s -> %s", persona_name, persona.archetype)

        return self._ensure_unique_archetypes(personas)

    def _generate_focused_name(self, profile_descriptor: str, letter: str) -> str:
        """Second-pass LLM call: given a data profile, generate a catchy 2-4 word business name."""
        prompt = (
            f"A customer segment has this data profile: {profile_descriptor}\n\n"
            f"Business context: {self.config.business_objective}\n\n"
            f"Generate ONE catchy 2-4 word business persona name for this segment.\n"
            f"Rules:\n"
            f"- Must be a memorable business label, e.g. 'Premium Loyalists', 'Budget Pragmatists', "
            f"'Rising Stars', 'At-Risk Big Spenders', 'Creditworthy Savers'\n"
            f"- Do NOT use generic words like 'Segment', 'Cluster', 'Group', 'Profile', 'Optimizers'\n"
            f"- Do NOT just repeat data column names or metrics\n"
            f"- Respond with ONLY the persona name, nothing else. No quotes, no explanation."
        )
        try:
            raw = self.client.generate(prompt, temperature=0.6, max_tokens=30)
            name = raw.strip().strip('"').strip("'").strip()
            # Remove any JSON wrapper if LLM over-formats
            if "{" in name or "}" in name:
                return ""
            # Validate length
            words = name.split()
            if 2 <= len(words) <= 5 and not self._is_generic_archetype(name):
                log.info("Focused naming for Cluster %s: '%s'", letter, name)
                return name
            return ""
        except Exception as e:
            log.debug("Focused naming failed for Cluster %s: %s", letter, e)
            return ""

    def _generate_llm_persona(
        self,
        cid: int,
        letter: str,
        persona_name: str,
        size: int,
        pct: float,
        profile_row: pd.Series,
        top_features_dict: dict[str, float],
        top_feature_names: list[str],
    ) -> PersonaResult:
        """Generate persona via Ollama LLM with two-pass naming."""
        # Always compute the data-driven profile descriptor
        profile_descriptor = self._derive_data_driven_archetype(profile_row, top_feature_names, letter)

        profile_text = build_cluster_profile_text(
            cluster_letter=letter,
            profile_row=profile_row,
            cluster_pct=pct,
            top_features=top_features_dict,
            top_feature_names=top_feature_names,
            business_objective=self.config.business_objective,
            global_means=self._global_means,
            global_stds=self._global_stds,
        )

        prompt = (
            f"{profile_text}\n\n"
            f"Generate a customer persona for this cluster. "
            f"Respond ONLY with valid JSON."
        )

        try:
            raw = self.client.generate(prompt, system=PERSONA_SYSTEM_PROMPT, temperature=0.3)
            parsed = self.client.extract_json(raw)
            llm_archetype = parsed.get("archetype", f"Segment {letter}")
            naming_rationale = (
                f"LLM-generated persona name using cluster-specific drivers: "
                f"{', '.join(top_feature_names[:3])}."
            )

            # If the first pass returned a generic name, try a focused second pass
            if self._is_generic_archetype(llm_archetype):
                focused_name = self._generate_focused_name(profile_descriptor, letter)
                if focused_name:
                    llm_archetype = focused_name
                    naming_rationale = (
                        f"Second-pass focused LLM naming from profile: {profile_descriptor}."
                    )
                else:
                    llm_archetype = self._business_fallback_name(top_feature_names, letter)
                    naming_rationale = (
                        f"LLM returned generic name; using business-style fallback from drivers: "
                        f"{', '.join(top_feature_names[:2])}."
                    )
            elif self._needs_business_rewrite(llm_archetype):
                focused_name = self._generate_focused_name(profile_descriptor, letter)
                if focused_name:
                    llm_archetype = focused_name
                    naming_rationale = (
                        f"LLM returned metric-literal name; rewritten via focused naming from profile: "
                        f"{profile_descriptor}."
                    )
                else:
                    llm_archetype = self._business_fallback_name(top_feature_names, letter)
                    naming_rationale = (
                        f"LLM returned metric-literal name; rewritten to business-style fallback "
                        f"from drivers: {', '.join(top_feature_names[:2])}."
                    )

            return PersonaResult(
                cluster_id=cid,
                persona_name=persona_name,
                archetype=llm_archetype,
                description=parsed.get("description", ""),
                key_traits=parsed.get("key_traits", []),
                business_recommendations=parsed.get("business_recommendations", []),
                cluster_size=size,
                cluster_pct=pct,
                top_features={
                    k: round(float(profile_row.get(k, 0)), 2)
                    for k in top_feature_names[:5]
                    if k in profile_row.index
                },
                naming_rationale=naming_rationale,
                categorization_basis=top_feature_names[:5],
                profile_descriptor=profile_descriptor,
            )
        except Exception as e:
            log.warning("LLM persona generation failed for cluster %s: %s", letter, e)
            return self._build_fallback_persona(
                cid, persona_name, size, pct, profile_row, top_features_dict, top_feature_names
            )

    def _is_generic_archetype(self, name: str) -> bool:
        n = (name or "").strip().lower()
        if not n:
            return True
        generic_tokens = ["segment", "cluster", "profile", "group", "persona", "a", "b", "c", "d"]
        if n in {"segment a", "segment b", "segment c", "cluster a", "cluster b", "cluster c"}:
            return True
        if sum(1 for t in generic_tokens if t in n) >= 2:
            return True
        # Reject names that are just column names glued together (e.g. "Credit Score Risk Score Optimizers")
        raw_col_tokens = set()
        for tok in n.replace("-", " ").replace("_", " ").split():
            raw_col_tokens.add(tok)
        filler_words = {"optimizers", "drivers", "segment", "cluster", "profile", "group", "and", "&", "the"}
        meaningful_words = raw_col_tokens - filler_words
        if len(meaningful_words) < 2:
            return True
        # Reject literal metric-style labels; force focused renaming pass
        if "&" in n and ("high " in n or "low " in n):
            return True
        if " score" in n and ("high " in n or "low " in n):
            return True
        return False

    def _needs_business_rewrite(self, name: str) -> bool:
        """Detect literal metric-style names that should be rewritten as business labels."""
        n = (name or "").strip().lower()
        if not n:
            return True
        bad_patterns = [
            r"\bhigh\b.*\blow\b",
            r"\blow\b.*\bhigh\b",
            r"\bcredit score\b",
            r"\brisk score\b",
            r"&",
        ]
        return any(re.search(p, n) for p in bad_patterns)

    def _business_fallback_name(self, top_feature_names: list[str], letter: str) -> str:
        """Create a compact business-style name from top drivers when LLM naming is literal/generic."""
        roots = []
        for f in self._deduplicate_feature_roots(top_feature_names):
            pretty = f.replace("_score", "").replace("_cluster", "").replace("_", " ").title().strip()
            if pretty:
                roots.append(pretty.split()[0])
            if len(roots) >= 2:
                break
        if len(roots) >= 2:
            return f"{roots[0]} {roots[1]} Strategists"
        if len(roots) == 1:
            return f"{roots[0]} Navigators"
        return f"Business Segment {letter}"

    @staticmethod
    def _deduplicate_feature_roots(features: list[str]) -> list[str]:
        """Remove semantically duplicate features (e.g. credit_score vs credit_score_cluster)."""
        seen_roots: set[str] = set()
        unique: list[str] = []
        for f in features:
            # Strip common suffixes that create near-duplicate feature names
            root = re.sub(
                r'_(cluster|group|bin|cat|flag|encoded|scaled|norm|raw|[xy]|bucket|band)$',
                '', f.lower(),
            )
            root = re.sub(r'_+$', '', root)
            if root not in seen_roots:
                seen_roots.add(root)
                unique.append(f)
        return unique

    def _classify_feature_level(self, feat: str, profile_row: pd.Series) -> str:
        """Classify a feature value as High/Moderate/Low relative to global population."""
        if not hasattr(self, '_global_means') or self._global_means is None:
            return ""
        if feat not in self._global_means.index or feat not in self._global_stds.index:
            return ""
        if feat not in profile_row.index:
            return ""

        value = float(profile_row[feat])
        mean = float(self._global_means[feat])
        std = float(self._global_stds[feat])
        if std < 1e-10:
            return ""

        z = (value - mean) / std
        if z >= 0.5:
            return "High"
        elif z <= -0.5:
            return "Low"
        else:
            return "Moderate"

    def _derive_data_driven_archetype(
        self,
        profile_row: pd.Series,
        top_feature_names: list[str],
        letter: str,
    ) -> str:
        """Derive a descriptive archetype using feature VALUES (not just names)."""
        usable = [f for f in top_feature_names if f in profile_row.index]
        # Remove semantically duplicate features (credit_score vs credit_score_cluster)
        usable = self._deduplicate_feature_roots(usable)
        if not usable:
            return f"Strategic Segment {letter}"

        parts: list[str] = []
        for feat in usable[:3]:
            pretty = feat.replace("_", " ").title()
            level = self._classify_feature_level(feat, profile_row)
            if level:
                parts.append(f"{level} {pretty}")
            else:
                parts.append(pretty)
            if len(parts) >= 2:
                break

        if len(parts) >= 2:
            return f"{parts[0]} & {parts[1]}"
        if len(parts) == 1:
            return f"{parts[0]} Segment"
        return f"Strategic Segment {letter}"

    def _build_fallback_persona(
        self,
        cid: int,
        persona_name: str,
        size: int,
        pct: float,
        profile_row: pd.Series,
        top_features_dict: dict[str, float],
        top_feature_names: list[str] | None = None,
    ) -> PersonaResult:
        """Rule-based fallback when Ollama is unavailable."""
        letter = CLUSTER_LETTERS.get(cid, str(cid))

        # Derive data-driven descriptor from top feature values
        top_feats = top_feature_names if top_feature_names else list(top_features_dict.keys())[:5]
        if top_feats:
            profile_descriptor = self._derive_data_driven_archetype(profile_row, top_feats, letter)
        else:
            profile_descriptor = f"Cluster {letter} Profile"

        # Try focused LLM naming even in fallback (persona JSON failed but naming might work)
        archetype = profile_descriptor
        naming_rationale = (
            "Fallback rule-based name from cluster-specific top movers: "
            + ", ".join(top_feats[:3])
        )
        if self._check_availability():
            focused_name = self._generate_focused_name(profile_descriptor, letter)
            if focused_name:
                archetype = focused_name
                naming_rationale = f"Focused LLM naming from profile: {profile_descriptor}."

        return PersonaResult(
            cluster_id=cid,
            persona_name=persona_name,
            archetype=archetype,
            description=f"Cluster {letter} represents {pct:.0%} of customers. "
                        f"Key differentiators: {', '.join(top_feats[:3])}.",
            key_traits=[f"Distinctive {f.replace('_', ' ').title()}" for f in top_feats[:4]],
            business_recommendations=[
                "Run Ollama to generate detailed recommendations",
                f"Investigate {top_feats[0].replace('_', ' ') if top_feats else 'key features'} patterns",
                "Conduct qualitative research with cluster members",
            ],
            cluster_size=size,
            cluster_pct=pct,
            top_features={
                k: round(float(profile_row.get(k, 0)), 2)
                for k in top_feats[:5]
                if k in profile_row.index
            },
            naming_rationale=naming_rationale,
            categorization_basis=top_feats[:5],
            profile_descriptor=profile_descriptor,
        )

    def _cluster_specific_feature_order(
        self,
        profile_row: pd.Series,
        global_means: pd.Series,
        global_stds: pd.Series,
        candidate_features: list[str],
    ) -> list[str]:
        """Sort features by absolute standardized movement for this cluster vs total population."""
        scored: list[tuple[str, float]] = []
        for f in candidate_features:
            if f in profile_row.index and f in global_means.index and f in global_stds.index:
                z_move = abs(float(profile_row[f] - global_means[f]) / float(global_stds[f]))
                scored.append((f, z_move))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [f for f, _ in scored]

    def ground_in_business_objective(
        self,
        personas: list[PersonaResult],
    ) -> BusinessGrounding:
        """LLM synthesises executive summary mapping personas to business objective."""
        if not self._check_availability():
            return BusinessGrounding(
                executive_summary="Ollama offline. Start Ollama and re-run for LLM-generated strategic grounding.",
                cluster_priorities=[
                    {"cluster": p.persona_name, "archetype": p.archetype,
                     "priority": "TBD", "strategic_action": "Run Ollama to generate"}
                    for p in personas
                ],
                quick_wins=[
                    "Start Ollama: ollama serve",
                    f"Pull model: ollama pull {self.config.ollama_model}",
                    "Re-run notebook",
                ],
                cluster_actions=[
                    {
                        "cluster": p.persona_name,
                        "grounded_persona_name": p.archetype,
                        "grounded_recommendations": p.business_recommendations[:3],
                    }
                    for p in personas
                ],
            )

        persona_summaries = []
        for p in personas:
            persona_summaries.append(
                f"  {p.persona_name} - {p.archetype}\n"
                f"    Size: {p.cluster_size:,} ({p.cluster_pct * 100:.1f}%)\n"
                f"    Description: {p.description[:140]}\n"
                f"    Top traits: {'; '.join(p.key_traits[:2])}\n"
                f"    Drivers: {', '.join(p.categorization_basis[:3])}"
            )

        prompt = (
            f"Business Objective:\n{self.config.business_objective}\n\n"
            f"Identified Customer Personas:\n" + "\n\n".join(persona_summaries) + "\n\n"
            f"Ground these personas in the stated business objective. "
            f"Identify priorities and strategic actions. Respond ONLY with valid JSON."
        )

        try:
            log.info("Grounding personas in business objective...")
            raw = self.client.generate(
                prompt,
                system=GROUNDING_SYSTEM_PROMPT,
                temperature=0.2,
                max_tokens=900,
            )
            parsed = self.client.extract_json(raw)
            return BusinessGrounding(
                executive_summary=parsed.get("executive_summary", ""),
                cluster_priorities=parsed.get("cluster_priority", []),
                quick_wins=parsed.get("quick_wins", []),
                cluster_actions=parsed.get("cluster_actions", []),
            )
        except Exception as e:
            log.warning("Business grounding failed: %s", e)
            return BusinessGrounding(
                executive_summary=f"LLM grounding failed: {e}",
                cluster_priorities=[],
                quick_wins=[],
                cluster_actions=[],
            )

    def apply_grounding_to_personas(
        self,
        personas: list[PersonaResult],
        grounding: BusinessGrounding,
    ) -> list[PersonaResult]:
        """Apply grounded persona names and recommendations returned by the grounding LLM step."""
        if not grounding.cluster_actions:
            return self._ensure_unique_archetypes(personas)

        action_map = {}
        for a in grounding.cluster_actions:
            cluster_key = str(a.get("cluster", "")).strip().lower()
            if cluster_key:
                action_map[cluster_key] = a

        updated: list[PersonaResult] = []
        for p in personas:
            key = str(p.persona_name).strip().lower()
            action = action_map.get(key)
            if action:
                grounded_name = action.get("grounded_persona_name")
                grounded_recs = action.get("grounded_recommendations", [])
                if isinstance(grounded_name, str) and grounded_name.strip():
                    p.archetype = grounded_name.strip()
                    p.naming_rationale = (
                        p.naming_rationale + " "
                        + "Business grounding renamed this persona to align with objective."
                    ).strip()
                if isinstance(grounded_recs, list) and grounded_recs:
                    p.business_recommendations = [str(x) for x in grounded_recs[:3]]
            updated.append(p)
        return self._ensure_unique_archetypes(updated)

    def _ensure_unique_archetypes(self, personas: list[PersonaResult]) -> list[PersonaResult]:
        """Enforce unique archetype names using distinguishing cluster characteristics."""
        groups: dict[str, list[PersonaResult]] = {}
        for p in personas:
            base = self._clean_archetype_text(p.archetype)
            p.archetype = base
            key = self._archetype_key(base)
            groups.setdefault(key, []).append(p)

        for _, group in groups.items():
            if len(group) <= 1:
                continue
            self._differentiate_duplicate_group(group)

        # Final hard guarantee of uniqueness
        seen_exact: dict[str, int] = {}
        for p in personas:
            k = p.archetype.strip().lower()
            seen_exact[k] = seen_exact.get(k, 0) + 1
            if seen_exact[k] > 1:
                cluster_tag = str(p.persona_name).replace("Cluster ", "").strip() or str(seen_exact[k])
                p.archetype = f"{p.archetype} ({cluster_tag})"
                p.naming_rationale = (
                    p.naming_rationale + " "
                    + f"Appended cluster tag '{cluster_tag}' to guarantee unique persona naming."
                ).strip()
        return personas

    def _differentiate_duplicate_group(self, group: list[PersonaResult]) -> None:
        """Rename duplicate archetypes using the feature that most distinguishes them."""
        # Collect all features present across the group's top_features
        all_feats: set[str] = set()
        for p in group:
            all_feats.update(p.top_features.keys())

        # Deduplicate feature roots to avoid credit_score vs credit_score_cluster
        deduped_feats = self._deduplicate_feature_roots(list(all_feats))

        # Find the feature with the largest value spread across the duplicate group
        best_feat, best_spread = None, -1.0
        for feat in deduped_feats:
            values = [p.top_features.get(feat, 0.0) for p in group]
            spread = max(values) - min(values)
            if spread > best_spread:
                best_spread = spread
                best_feat = feat

        if best_feat is None or best_spread < 1e-10:
            # No distinguishing feature found; fall back to cluster letters
            for p in group:
                letter = str(p.persona_name).replace("Cluster ", "").strip()
                p.archetype = f"{p.archetype} ({letter})"
                p.naming_rationale += f" Differentiated by cluster label '{letter}'."
            return

        # Rank clusters by the distinguishing feature and assign descriptive modifiers
        ranked = sorted(group, key=lambda p: p.top_features.get(best_feat, 0.0), reverse=True)
        pretty_feat = best_feat.replace("_", " ").title()
        n = len(ranked)

        for i, p in enumerate(ranked):
            val = p.top_features.get(best_feat, 0.0)
            if n == 2:
                modifier = "Higher" if i == 0 else "Lower"
            else:
                if i == 0:
                    modifier = "High"
                elif i == n - 1:
                    modifier = "Low"
                else:
                    modifier = "Mid"
            p.archetype = f"{p.archetype} – {modifier} {pretty_feat}"
            p.naming_rationale += (
                f" Differentiated by {best_feat} (value={val:.2f}, "
                f"ranked {i + 1}/{n} across duplicate names)."
            )

    def _clean_archetype_text(self, name: str) -> str:
        n = (name or "").strip()
        n = re.sub(r"\s+", " ", n)
        n = re.sub(r"[-\s]+$", "", n)  # remove trailing hyphen/space
        return n or "Strategic Segment"

    def _archetype_key(self, name: str) -> str:
        n = name.lower()
        n = re.sub(r"[^a-z0-9\s]", " ", n)
        n = re.sub(r"\s+", " ", n).strip()
        return n
