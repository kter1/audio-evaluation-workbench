"""Blind-safe progress and next actions; no references or detector scores."""
from .store import PHASES, now

PHASE_LABELS = {"calibration": "Calibration", "main-01": "Main study I",
                "main-02": "Main study II", "validation": "Fresh validation", "challenge": "Listening challenge"}


def progress(store, reviewer):
    queue = store.queue(reviewer)
    primary = [a for a in queue if a["role"] != "repeat"]
    drafts = {r["assignment_id"] for r in store.rows("""SELECT DISTINCT e.assignment_id FROM annotation_revisions e
        JOIN assignments a ON a.id=e.assignment_id WHERE a.reviewer=? AND a.submitted_at IS NULL AND e.event='draft'""", (reviewer,))}
    batches = []
    for phase in PHASES:
        items = [a for a in primary if a["phase"] == phase]
        if not items:
            continue
        repeats = [a for a in queue if a["phase"] == phase and a["role"] == "repeat"]
        batches.append({"phase": phase, "label": PHASE_LABELS[phase], "total": len(items),
                        "done": sum(bool(a["submitted_at"]) for a in items),
                        "revealed": all(a["revealed_at"] for a in items),
                        "drafts": sum(a["id"] in drafts for a in items),
                        "repeats": len(repeats), "repeat_done": sum(bool(a["submitted_at"]) for a in repeats)})
    next_step = {"screen": "Findings & Export", "phase": None, "label": "Open your findings",
                 "description": "Turn reviewed cases into evidence and a research handoff."}
    if not queue:
        next_step = {"screen": "Review", "phase": None, "label": "Open review",
                     "description": "No recordings are assigned in this workspace yet. See the setup guide."}
    deferred = None
    for batch in batches:
        phase = batch["phase"]
        if batch["done"] < batch["total"]:
            next_step = {"screen": "Review", "phase": phase,
                         "label": ("Resume " if batch["drafts"] else "Start ") + ("calibration" if phase == "calibration" else batch["label"].lower()),
                         "description": ("Listen to this first batch, then compare your judgments with the references. Calibration feedback stays separate from study results."
                                         if phase == "calibration" else "Continue blind listening and document the evidence behind your decisions.")}
            break
        if batch["revealed"] or store.setting("reviewer_mode"):
            continue
        repeats = [a for a in queue if a["phase"] == phase and a["role"] == "repeat" and not a["submitted_at"]]
        ready = [a for a in repeats if not a["due_at"] or a["due_at"] <= now()]
        if ready:
            next_step = {"screen": "Review", "phase": phase, "label": "Complete your blind repeats",
                         "description": "The waiting period has passed. Review these clips without seeing your earlier answers."}
            break
        if repeats:
            deferred = {"screen": "Quality Review", "phase": phase, "label": "Check repeat availability",
                        "description": "Your repeat reviews are waiting for the 48-hour interval. You can continue another batch meanwhile."}
            continue
        next_step = {"screen": "Quality Review", "phase": phase,
                     "label": "Check calibration" if phase == "calibration" else "Run quality checks",
                     "description": "This batch is submitted. Complete any assigned second reviews, then reveal the comparison."}
        break
    else:
        if deferred:
            next_step = deferred
        elif store.setting("reviewer_mode") and queue:
            next_step = {"screen": "Quality Review", "phase": None, "label": "Return your independent reviews",
                         "description": "Your assignments are complete. Download the frozen results for the study owner."}
    return {"batches": batches, "done": sum(bool(a["submitted_at"]) for a in primary),
            "total": len(primary), "draft_ids": drafts, "next": next_step}
