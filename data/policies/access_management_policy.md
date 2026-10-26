# Access Management Policy

Owner: IT
Effective date: 2026-06-01

## Purpose and precedence

This policy governs how employees and contractors request and receive access to
NovaOps systems. Security and contract limits override this general policy — if the
Security and MFA Policy or a vendor contract says otherwise, that document wins.

## Approval tiers

Access requirements scale with sensitivity:

- Standard role-based access (the tools a role normally needs) requires manager approval.
- Privileged or admin access requires IT Manager approval, on top of the manager's.
- AWS admin specifically requires both IT Manager and Finance approval, since it
  carries billing exposure as well as infrastructure risk.
- Contractors need both manager and IT approval for GitHub access, regardless of
  which repositories are involved.
- BambooHR is HR-only, with the single exception of employee self-service (an
  employee viewing their own profile, PTO balance, or pay stubs).

## Requesting access

Access requests should name the specific system and the business reason, not just
"same as my teammate" — role comparisons drift over time and can over-grant.
Approvals are recorded against the request so there is always a record of who
authorized what and why.

## Limits beyond eligibility

Being eligible for a system does not guarantee access on request day: eligible
users can still be blocked by seat limits or an active renewal freeze on that
software. When that happens, the request is not denied outright — it is queued and
the requester is told why, so it can be granted the moment capacity frees up.

## Periodic review

IT reviews privileged and admin grants on a quarterly basis and revokes any that no
longer map to an active business need, independent of the offboarding process.
