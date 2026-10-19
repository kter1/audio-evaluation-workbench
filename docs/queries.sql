-- Work on a private research database. These queries are not blind-view endpoints.
-- Original submissions only; corrections do not overwrite study evidence.

-- Revealed escalations, ready for research handoff.
SELECT r.trial_id, a.reviewer, a.phase, a.guideline,
       json_extract(e.payload, '$.escalation_reason') AS question,
       json_extract(e.payload, '$.observations') AS evidence
FROM annotation_revisions e
JOIN assignments a ON a.id = e.assignment_id
JOIN recordings r ON r.id = a.recording_id
WHERE e.event = 'submitted' AND a.revealed_at IS NOT NULL
  AND json_extract(e.payload, '$.escalated') = 1;

-- Independent disagreement with the original primary review.
SELECT r.trial_id, p.reviewer AS primary_reviewer, q.reviewer AS independent_reviewer,
       json_extract(ep.payload, '$.judgment') AS primary_judgment,
       json_extract(eq.payload, '$.judgment') AS independent_judgment
FROM assignments p
JOIN assignments q ON p.recording_id = q.recording_id AND q.role = 'independent'
JOIN recordings r ON r.id = p.recording_id
JOIN annotation_revisions ep ON ep.assignment_id = p.id AND ep.event = 'submitted'
JOIN annotation_revisions eq ON eq.assignment_id = q.id AND eq.event = 'submitted'
WHERE p.role = 'primary' AND p.revealed_at IS NOT NULL AND q.revealed_at IS NOT NULL
  AND p.guideline = q.guideline
  AND json_extract(ep.payload, '$.judgment') <> json_extract(eq.payload, '$.judgment');

-- Timestamped history for a recording; originals, drafts, and corrections stay distinct.
SELECT r.trial_id, a.reviewer, e.event, e.created_at, e.payload
FROM annotation_revisions e
JOIN assignments a ON a.id = e.assignment_id
JOIN recordings r ON r.id = a.recording_id
WHERE a.revealed_at IS NOT NULL
ORDER BY r.trial_id, e.id;

-- One original primary submission per phase, never count repeats as unique work.
SELECT a.phase, COUNT(*) AS unique_completed
FROM assignments a
WHERE a.role = 'primary' AND a.submitted_at IS NOT NULL
GROUP BY a.phase;
